"""Servicio privado de traducción de glosas; no registra el texto recibido."""

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
import sys
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator
from starlette.responses import JSONResponse

from verify_package import verify_package

MAX_BODY_BYTES = 8192
MAX_TEXT_CHARACTERS = 2000


class InputTooLong(ValueError):
    pass


class TranslationInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1, max_length=MAX_TEXT_CHARACTERS)
    source: Literal["recognizer"] | None = None
    is_question: StrictBool | None = None

    @field_validator("text")
    @classmethod
    def nonempty_text(cls, value):
        if not value.strip() or "\x00" in value:
            raise ValueError("Invalid text")
        return value


class BodyLimitMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/translate":
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > MAX_BODY_BYTES:
                await JSONResponse({"detail": "request_too_large"}, status_code=413)(scope, receive, send)
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        async def limited_receive():
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, limited_receive, send)


def load_translator():
    package = Path(os.environ.get("MODEL_PACKAGE_DIR", "/opt/asl-model")).resolve()
    manifest = verify_package(package)
    sys.path.insert(0, str(package))
    from onnx_translator import MAX_INPUT, OnnxGlossTranslator
    from tokenizers import Tokenizer

    class CheckedTranslator(OnnxGlossTranslator):
        def __init__(self):
            threads = int(os.environ.get("TRANSLATOR_THREADS", "2"))
            if not 1 <= threads <= 16:
                raise RuntimeError("TRANSLATOR_THREADS must be between 1 and 16")
            super().__init__(package / "model", threads=threads)
            self.variant = manifest["variant"]
            self.input_validator = Tokenizer.from_file(str(package / "model" / "tokenizer.json"))
            self.input_validator.no_truncation()
            self.input_validator.no_padding()

        def generate(self, text):
            # Incluye el prefijo y las reglas reales, sin modificar el tokenizer del modelo.
            if len(self.input_validator.encode(text).ids) > MAX_INPUT:
                raise InputTooLong("input_token_limit")
            return super().generate(text)

    return CheckedTranslator()


def create_app(translator_factory=load_translator, timeout_seconds=None):
    timeout = float(timeout_seconds if timeout_seconds is not None
                    else os.environ.get("TRANSLATOR_TIMEOUT_SECONDS", "10"))
    if not 0 < timeout <= 120:
        raise RuntimeError("TRANSLATOR_TIMEOUT_SECONDS must be greater than 0 and at most 120")

    @asynccontextmanager
    async def lifespan(application):
        application.state.translator = await asyncio.to_thread(translator_factory)
        application.state.slot = asyncio.Semaphore(1)
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    application.add_middleware(BodyLimitMiddleware)

    @application.exception_handler(RequestValidationError)
    async def validation_error(_request, _error):
        # FastAPI incluye por defecto la entrada inválida; aquí evitamos devolverla.
        return JSONResponse({"detail": "invalid_translation_input"}, status_code=422)

    @application.get("/health")
    async def health():
        if not getattr(application.state, "ready", False):
            raise HTTPException(status_code=503, detail="not_ready")
        return {"status": "ok", "model": "asl_gloss_to_english",
                "variant": getattr(application.state.translator, "variant", "fp32")}

    @application.post("/translate")
    async def translate(payload: TranslationInput):
        if not getattr(application.state, "ready", False):
            raise HTTPException(status_code=503, detail="not_ready")
        slot = application.state.slot
        if slot.locked():
            raise HTTPException(status_code=503, detail="translator_busy", headers={"Retry-After": "1"})
        await slot.acquire()

        def complete(task):
            slot.release()
            if not task.cancelled():
                task.exception()

        task = asyncio.create_task(asyncio.to_thread(
            application.state.translator, payload.text,
            source=payload.source, is_question=payload.is_question,
        ))
        task.add_done_callback(complete)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except InputTooLong:
            raise HTTPException(status_code=422, detail="input_token_limit") from None
        except asyncio.TimeoutError:
            # No se libera el cupo hasta que ONNX termine: cancelar la espera no cancela CPU nativa.
            raise HTTPException(status_code=504, detail="translation_timeout") from None
        except Exception:
            raise HTTPException(status_code=503, detail="translation_failed") from None

    return application


app = create_app()
