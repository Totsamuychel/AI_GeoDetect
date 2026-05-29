"""
server.py — Легкий веб-сервер демо геолокації (без зовнішніх фреймворків).

Використовує лише стандартну бібліотеку Python (http.server). API приймає
зображення у вигляді base64 (data-URL) у JSON-тілі — це уникає потреби в
парсингу multipart та сторонніх залежностях.

Запуск (з conda-середовища diploma):
    python webapp/server.py
    python webapp/server.py --port 8000 --host 0.0.0.0

Потім відкрийте http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Гарантуємо UTF-8 вивід у консоль Windows.
if getattr(sys.stdout, "encoding", "") and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from registry import decode_image, get_registry  # noqa: E402

logger = logging.getLogger("server")

STATIC_DIR = Path(__file__).resolve().parent / "static"

MIME = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg":  "image/svg+xml",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
}

MAX_UPLOAD = 25 * 1024 * 1024  # 25 МБ


class Handler(BaseHTTPRequestHandler):
    server_version = "GeoDemo/1.0"

    # Тихіший лог
    def log_message(self, fmt, *args):  # noqa: A003
        logger.info("%s - %s", self.address_string(), fmt % args)

    # ── Допоміжні відповіді ────────────────────────────────────────────────
    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._send_json({"error": "Не знайдено"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):  # noqa: N802
        path = self.path.split("?", 1)[0]

        if path == "/" or path == "/index.html":
            self._send_file(STATIC_DIR / "index.html")
            return

        if path == "/api/models":
            try:
                self._send_json({"models": get_registry().available()})
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": str(e)}, 500)
            return

        # Статика (захист від виходу за межі каталогу)
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() in target.parents or target == STATIC_DIR.resolve():
                self._send_file(target)
            else:
                self._send_json({"error": "Заборонено"}, 403)
            return

        self._send_json({"error": "Не знайдено"}, 404)

    # ── POST ───────────────────────────────────────────────────────────────────
    def do_POST(self):  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/predict":
            self._send_json({"error": "Не знайдено"}, 404)
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_UPLOAD:
            self._send_json({"error": "Невірний або завеликий запит (макс. 25 МБ)."}, 413)
            return

        try:
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            arch = payload.get("model", "streetclip")
            image_data = payload.get("image")
            if not image_data:
                self._send_json({"error": "Не передано зображення."}, 400)
                return
            image = decode_image(image_data)
            result = get_registry().predict(arch, image)
            self._send_json(result)
        except FileNotFoundError as e:
            self._send_json({"error": f"Модель недоступна: {e}"}, 503)
        except Exception as e:  # noqa: BLE001
            logger.exception("Помилка інференсу")
            self._send_json({"error": f"Помилка обробки: {e}"}, 500)


def main():
    p = argparse.ArgumentParser(description="Веб-демо геолокації (Варшава/Прага/Будапешт)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--preload", action="store_true",
                   help="Завантажити всі моделі одразу при старті (повільніший старт).")
    args = p.parse_args()

    reg = get_registry()
    if args.preload:
        for m in reg.available():
            if m["available"]:
                try:
                    reg._ensure_loaded(m["id"])  # noqa: SLF001
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Не вдалося завантажити {m['id']}: {e}")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print("=" * 60)
    print(f"  Веб-демо геолокації запущено: {url}")
    print(f"  Моделі: {', '.join(m['id'] for m in reg.available() if m['available'])}")
    print("  Перше передбачення кожною моделлю — повільніше (завантаження +")
    print("  калібрування OOD). Натисніть Ctrl+C для зупинки.")
    print("=" * 60)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nЗупинка сервера…")
        httpd.shutdown()


if __name__ == "__main__":
    main()
