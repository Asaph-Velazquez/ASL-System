"""Comprueba los cálculos y la clasificación de fallos del comparador."""

import unittest
from unittest.mock import patch
import urllib.error

from compare import memory_bytes, parse_stats_line, percentile, request, summary


class BenchmarkTests(unittest.TestCase):
    def test_stats_stream_strips_docker_windows_terminal_controls(self):
        self.assertEqual(parse_stats_line('\x1b[H{"Name":"test","MemUsage":"2.035GiB / 15GiB"}\x1b[K\n')["Name"], "test")
        self.assertIsNone(parse_stats_line("\x1b[K\n"))
        with self.assertRaises(ValueError):
            parse_stats_line("not-json")

    def test_percentile_interpolation_and_empty_samples(self):
        self.assertIsNone(percentile([], .95))
        self.assertEqual(percentile([3], .95), 3)
        self.assertEqual(percentile([40, 10, 30, 20], .50), 25)
        self.assertAlmostEqual(percentile([10, 20, 30, 40], .95), 38.5)

    def test_memory_units_match_actual_docker_output(self):
        self.assertEqual(memory_bytes("2.035GiB"), 2.035 * 1024**3)
        self.assertEqual(memory_bytes("123 MiB"), 123 * 1024**2)
        with self.assertRaises(ValueError):
            memory_bytes("unknown")

    def test_failure_remains_counted_but_is_not_success_latency(self):
        result = summary([{"elapsed_ms": 10}, {"elapsed_ms": 20},
                          {"elapsed_ms": 15000, "error": "timeout"}])
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["errors"], 1)
        self.assertEqual(result["latency_success_ms"]["mean"], 15)

    def test_remote_target_is_rejected_before_network_access(self):
        with patch("urllib.request.urlopen") as send:
            with self.assertRaises(ValueError):
                request("https://example.com", "/translate", {"text": "WATER"})
            send.assert_not_called()

    def test_http_failure_and_network_error_are_recorded(self):
        import io
        failure = urllib.error.HTTPError("local", 503, "busy", {}, io.BytesIO(b'{"detail":"busy"}'))
        with patch("urllib.request.urlopen", side_effect=failure):
            self.assertEqual(request("http://127.0.0.1:8003", "/translate", {"text": "WATER"})["error"],
                             "invalid_translation")
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            self.assertEqual(request("http://127.0.0.1:8003", "/translate", {"text": "WATER"})["error"],
                             "TimeoutError")


if __name__ == "__main__":
    unittest.main()
