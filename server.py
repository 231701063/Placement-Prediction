import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from config import PORT, PUBLIC_DIR
from model_service import MODEL_BUNDLE, build_overview_payload, format_percent, predict_placement


class PlacementHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/overview":
            self.send_json(200, build_overview_payload())
            return

        if parsed.path == "/":
            self.serve_file(PUBLIC_DIR / "index.html")
            return

        file_path = (PUBLIC_DIR / parsed.path.lstrip("/")).resolve()
        if not str(file_path).startswith(str(PUBLIC_DIR.resolve())):
            self.send_json(403, {"error": "Forbidden"})
            return
        self.serve_file(file_path)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length).decode("utf-8")
            payload = json.loads(raw_body) if raw_body else {}
            result = predict_placement(payload)
            self.send_json(200, result)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid JSON body"})
        except ValueError as error:
            self.send_json(400, {"error": str(error)})

    def serve_file(self, file_path):
        if not file_path.exists() or not file_path.is_file():
            self.send_json(404, {"error": "File not found"})
            return

        content_type = self.get_content_type(file_path.suffix.lower())
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def get_content_type(self, suffix):
        mapping = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }
        return mapping.get(suffix, "text/plain; charset=utf-8")

    def log_message(self, format_string, *args):
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), PlacementHandler)
    print(f"Placement predictor is running on http://localhost:{PORT}")
    print(
        "Accuracy: "
        f"{format_percent(MODEL_BUNDLE['metrics']['accuracy'])} | "
        f"F1: {format_percent(MODEL_BUNDLE['metrics']['f1Score'])}"
    )
    server.serve_forever()
