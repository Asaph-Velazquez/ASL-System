"""Comparación local reproducible con HTTP y lecturas de recursos Docker."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def percentile(values, fraction):
    """Percentil interpolado linealmente; no descarta errores para fingir menor latencia."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def request(base, path, payload=None):
    address = urllib.parse.urlparse(base)
    if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Only local HTTP endpoints are accepted")
    if address.username or address.password or address.path not in {"", "/"}:
        raise ValueError("Invalid local endpoint")
    req = urllib.request.Request(base.rstrip("/") + path,
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            status, body = response.status, response.read(65537)
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read(65537)
    except (OSError, TimeoutError) as error:
        return {"status": None, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": type(error).__name__}
    elapsed = (time.perf_counter() - started) * 1000
    try:
        result = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {"status": status, "elapsed_ms": elapsed, "error": "invalid_json"}
    if path == "/translate" and (status != 200 or not isinstance(result, dict)
                                  or not isinstance(result.get("english"), str)
                                  or not result["english"].strip()
                                  or result.get("gloss") != payload["text"]):
        return {"status": status, "elapsed_ms": elapsed, "error": "invalid_translation",
                "response": result}
    return {"status": status, "elapsed_ms": elapsed, "response": result}


def docker_json(docker, *arguments):
    return json.loads(subprocess.check_output([docker, *arguments], text=True, timeout=20))


def cpu_seconds(docker, container):
    text = subprocess.check_output([docker, "exec", container, "cat", "/sys/fs/cgroup/cpu.stat"],
                                   text=True, timeout=20)
    stats = dict(line.split() for line in text.splitlines())
    return int(stats["usage_usec"]) / 1000000


def memory_bytes(value):
    match = re.fullmatch(r"\s*([0-9.]+)\s*([A-Za-z]+)\s*", value)
    if not match:
        raise ValueError("Invalid Docker memory size")
    number, unit = match.groups()
    units = {"B": 1, "kB": 1000, "KB": 1000, "MB": 1000**2, "GB": 1000**3,
             "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return float(number) * units[unit]


def parse_stats_line(line):
    # Docker para Windows intercala controles ANSI y líneas vacías aun con stdout redirigido.
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
    return json.loads(cleaned) if cleaned else None


class Sampler:
    def __init__(self, docker, containers):
        self.samples = []
        self.errors = []
        self.process = subprocess.Popen([docker, "stats", "--format", "{{json .}}", *containers],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.process.stdout:
            try:
                item = parse_stats_line(line)
                if item is None:
                    continue
                item["sampled_at_monotonic"] = time.monotonic()
                item["memory_bytes"] = memory_bytes(item["MemUsage"].split("/")[0])
                self.samples.append(item)
            except (ValueError, KeyError) as error:
                self.errors.append(type(error).__name__)

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        self.thread.join(timeout=5)
        stderr = self.process.stderr.read()
        if stderr.strip():
            self.errors.append(stderr.strip())


def describe_container(docker, name):
    info = docker_json(docker, "inspect", name)[0]
    environment = dict(value.split("=", 1) for value in info["Config"]["Env"] if "=" in value)
    if environment.get("TRANSLATOR_THREADS") != "2":
        raise ValueError(f"{name}: TRANSLATOR_THREADS must equal 2")
    image = docker_json(docker, "image", "inspect", info["Image"])[0]
    model_files = json.loads(subprocess.check_output([
        docker, "exec", name, "python", "-c",
        'import json,pathlib; p=pathlib.Path("/opt/asl-model/model"); '
        'print(json.dumps({f.name:f.stat().st_size for f in p.glob("*.onnx")}))',
    ], text=True, timeout=20))
    return {"name": name, "id": info["Id"], "image_id": info["Image"],
            "image_bytes": image["Size"], "model_files": model_files,
            "model_bytes": sum(model_files.values()), "threads": 2}


def summary(records):
    valid = [record["elapsed_ms"] for record in records if "error" not in record]
    return {"requests": len(records), "successes": len(valid), "errors": len(records) - len(valid),
            "latency_success_ms": {"mean": statistics.mean(valid) if valid else None,
                                   "p50": percentile(valid, .50), "p95": percentile(valid, .95)}}


def markdown(report):
    lines = ["# Comparación local FP32 e INT8", "", f"Fecha UTC: {report['created_utc']}", "",
             "Corpus sintético de hotel, sin usar el conjunto de entrenamiento como evaluación. "
             "Las coincidencias con FP32 no son una medida de exactitud ni una validación humana.", "",
             "| Medición | FP32 | INT8 |", "|---|---:|---:|"]
    for label, key in [("ONNX (bytes)", "model_bytes"), ("Imagen lógica (bytes)", "image_bytes"),
                       ("RAM en reposo, muestra (bytes)", "idle_memory_bytes"),
                       ("RAM máxima muestreada durante carga (bytes)", "sampled_load_memory_max_bytes"),
                       ("CPU del contenedor por petición (s)", "cpu_seconds_per_request")]:
        lines.append(f"| {label} | {report['variants']['fp32'].get(key)} | {report['variants']['int8'].get(key)} |")
    for key in ["mean", "p50", "p95"]:
        values = [report["variants"][v]["summary"]["latency_success_ms"][key] for v in ["fp32", "int8"]]
        lines.append(f"| Latencia {key} de éxitos (ms) | {values[0]} | {values[1]} |")
    lines += ["", f"Casos con salida inicial idéntica: {report['exact_matches']} / {len(report['comparisons'])}.",
              "", "CPU: diferencia de uso acumulado de cgroup durante toda la carga alternada, "
              "incluye el pequeño coste de leer las estadísticas. RAM: muestras de Docker; "
              "el máximo muestreado puede omitir picos breves. No es el máximo del kernel. "
              "Imagen: tamaño lógico, no espacio físico adicional tras deduplicación. "
              "Latencias de éxitos; los errores permanecen contados y registrados.", "",
              "Los contenedores comparten host y permanecen cargados; las peticiones son seriales, "
              "alternando el orden de las variantes en cada caso y repetición. La medición "
              "describe este equipo local, no garantiza el rendimiento de Azure.", "",
              "## Traducciones para revisión humana", "", "| Entrada | FP32 | INT8 | Coincide |",
              "|---|---|---|---|"]
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    for case in report["comparisons"]:
        lines.append("| " + " | ".join(escape(case[k]) for k in ["text", "fp32", "int8", "equal"]) + " |")
    lines += ["", "Se deben revisar especialmente negaciones, cantidades, números de habitación, "
              "destinos y urgencia. Una diferencia puede ser equivalente, mejor o peor. "
              "El corpus no tiene referencias humanas independientes y no permite calcular precisión.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docker", default=shutil.which("docker") or "docker")
    parser.add_argument("--fp32-container", default="asl-system-asl-gloss-translator-1")
    parser.add_argument("--int8-container", default="asl-system-asl-gloss-translator-int8-1")
    parser.add_argument("--fp32-url", default="http://127.0.0.1:8002")
    parser.add_argument("--int8-url", default="http://127.0.0.1:8003")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(".dev-logs/gloss-int8-comparison"))
    args = parser.parse_args()
    if args.rounds < 3:
        parser.error("At least three rounds are required")
    corpus = json.loads(Path(__file__).with_name("corpus.json").read_text(encoding="utf-8"))
    names = {"fp32": args.fp32_container, "int8": args.int8_container}
    urls = {"fp32": args.fp32_url, "int8": args.int8_url}
    variants = {key: describe_container(args.docker, value) for key, value in names.items()}
    warmup = []
    for key in names:
        health = request(urls[key], "/health")
        expected = {"fp32"} if key == "fp32" else {"int8", "int8-dynamic"}
        if health.get("response", {}).get("variant") not in expected or health["status"] != 200:
            raise RuntimeError(f"Incorrect health contract for {key}: {health}")
        variants[key]["health"] = health["response"]
        for text in corpus[:3]:
            record = request(urls[key], "/translate", {"text": text})
            warmup.append({"variant": key, "text": text, **record})
            if "error" in record:
                raise RuntimeError(f"Warmup failed for {key}: {record}")
    idle = docker_json(args.docker, "stats", "--no-stream", "--format", "{{json .}}", names["fp32"])
    variants["fp32"]["idle_memory_bytes"] = memory_bytes(idle["MemUsage"].split("/")[0])
    idle = docker_json(args.docker, "stats", "--no-stream", "--format", "{{json .}}", names["int8"])
    variants["int8"]["idle_memory_bytes"] = memory_bytes(idle["MemUsage"].split("/")[0])
    before = {key: cpu_seconds(args.docker, name) for key, name in names.items()}
    sampler = Sampler(args.docker, list(names.values()))
    records = []
    started = time.monotonic()
    try:
        for repeat in range(args.rounds):
            for index, text in enumerate(corpus):
                for key in (["fp32", "int8"] if (repeat + index) % 2 == 0 else ["int8", "fp32"]):
                    records.append({"variant": key, "case": index, "round": repeat,
                                    **request(urls[key], "/translate", {"text": text})})
    finally:
        finished = time.monotonic()
        sampler.close()
    for key, name in names.items():
        selected = [record for record in records if record["variant"] == key]
        variants[key]["summary"] = summary(selected)
        variants[key]["cpu_seconds_workload"] = cpu_seconds(args.docker, name) - before[key]
        variants[key]["cpu_seconds_per_request"] = variants[key]["cpu_seconds_workload"] / len(selected)
        samples = [s["memory_bytes"] for s in sampler.samples if s["Name"] == name
                   and started <= s["sampled_at_monotonic"] <= finished]
        variants[key]["sampled_load_memory_max_bytes"] = max(samples) if samples else None
        variants[key]["load_memory_sample_count"] = len(samples)
    comparisons = []
    for index, text in enumerate(corpus):
        outputs = {key: [r.get("response", {}).get("english") for r in records
                         if r["variant"] == key and r["case"] == index] for key in names}
        comparisons.append({"text": text, "fp32": outputs["fp32"][0], "int8": outputs["int8"][0],
                            "equal": outputs["fp32"][0] is not None and outputs["fp32"][0] == outputs["int8"][0],
                            "stable": {key: len(set(values)) == 1 for key, values in outputs.items()}})
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "rounds": args.rounds,
              "corpus": corpus, "corpus_provenance": "synthetic-hotel-prompts-not-held-out-ground-truth",
              "variants": variants, "records": records, "warmup": warmup,
              "resource_samples": sampler.samples, "resource_sampling_errors": sampler.errors,
              "comparisons": comparisons, "exact_matches": sum(c["equal"] for c in comparisons)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output / "COMPARACION.md").write_text(markdown(report), encoding="utf-8")
    incomplete = sampler.errors or any(v["summary"]["errors"]
                                      or v["load_memory_sample_count"] == 0 for v in variants.values())
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
