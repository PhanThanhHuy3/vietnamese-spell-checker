"""Serve the spelling checker and verified research artifacts on the same origin."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from urllib.parse import parse_qsl, urlsplit

from src.data.noise_synthesizer import NOISE_LABELS, VietnameseNoiseSynthesizer
from src.data.prepare_vsec import ROOT
from src.inference import CorrectionService, ModelUnavailable
from src.training.metrics import evaluate_edits
from src.web_data import WebsiteData

ASSETS = {
    "/": ("index.html", "text/html"),
    "/index.html": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/research.js": ("research.js", "text/javascript"),
}
PAGE_ROUTES = {"/editor", "/guide", "/about", "/lab", "/lab/compare", "/lab/dataset",
               "/lab/augmentation", "/lab/experiments", "/lab/models"}
MAX_TEXT = 4000
MAX_BODY = 32768


def checked_text(data):
    text = data.get("text")
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise ValueError(f"Nhập văn bản từ 1 đến {MAX_TEXT} ký tự.")
    text.encode("utf-8")
    return text


class SpellCheckerRequestHandler(BaseHTTPRequestHandler):
    def respond(self, status, payload, content_type="application/json"):
        body = (json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
                if content_type == "application/json" else payload)
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except ConnectionError:
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        path, query = url.path, dict(parse_qsl(url.query))
        service = self.server.service
        data = self.server.data
        if path in ASSETS or path in PAGE_ROUTES:
            name, mime = ASSETS.get(path, ASSETS["/"])
            self.respond(200, (ROOT / "web" / name).read_bytes(), mime)
            return
        try:
            result = None
            if path in ("/api/models", "/api/status"):
                models = service.status()
                result = {"models": models, "recommendation": service.recommendation(models),
                          "max_text_length": MAX_TEXT}
            elif path.startswith("/api/models/"):
                name = path.removeprefix("/api/models/")
                result = service.check_checkpoint(name) if query.get("probe") == "1" else service.inventory(name)
            elif path in ("/api/dataset/stats", "/api/dataset"):
                result = {"token_histograms": data.token_histograms} if query.get("tokens") == "1" else data.stats
            elif path == "/api/dataset/samples":
                result = data.samples(query)
            elif path == "/api/dataset/integrity":
                result = data.integrity()
            elif path == "/api/experiments":
                result = {"runs": data.experiments()}
            elif path.startswith("/api/experiments/"):
                parts = path.removeprefix("/api/experiments/").split("/")
                if len(parts) == 1 or len(parts) == 2 and parts[1] in ("predictions", "history"):
                    result = data.experiment(parts[0], parts[1] if len(parts) > 1 else None, query)
            if result is None:
                self.respond(404, {"error": "Không tìm thấy dữ liệu."})
            else:
                self.respond(200, result)
        except (ValueError, ModelUnavailable) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception:
            logging.exception("Không đọc được %s", path)
            self.respond(500, {"error": "Không thể tải dữ liệu."})

    def do_POST(self):
        path = urlsplit(self.path).path
        action = {
            "/api/check": "check", "/api/correct_spelling": "check",
            "/api/compare": "compare", "/api/compare_engines": "compare",
            "/api/augment": "augment", "/api/generate_noise": "augment",
        }.get(path)
        if not action:
            self.respond(404, {"error": "Không tìm thấy đường dẫn."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BODY:
                self.respond(413, {"error": "Kích thước yêu cầu không hợp lệ."})
                return
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("Dữ liệu phải là đối tượng JSON.")
            text = checked_text(data)
            result = getattr(self, action)(text, data)
        except ModelUnavailable as exc:
            self.respond(503, {"error": str(exc)})
        except (ValueError, UnicodeError) as exc:
            self.respond(400, {"error": str(exc)})
        except Exception:
            logging.exception("Không xử lý được %s", path)
            self.respond(500, {"error": "Không thể xử lý yêu cầu."})
        else:
            self.respond(200, result)

    def check(self, text, data):
        model = data.get("model", data.get("mode", "auto"))
        if not isinstance(model, str):
            raise ValueError("Tên mô hình không hợp lệ.")
        return self.server.service.check(text, {"rule": "baseline", "ai": "vit5"}.get(model, model))

    def compare(self, text, data):
        service = self.server.service
        reference = data.get("reference")
        if reference is not None:
            reference = checked_text({"text": reference})
        models = service.status()
        requested = data.get("models", [name for name, info in models.items() if info["available"]])
        if not isinstance(requested, list) or not requested or any(type(name) is not str or name not in models for name in requested):
            raise ValueError("Chọn mô hình hợp lệ để chạy.")
        results = {}
        for model in dict.fromkeys(requested):
            if not models[model]["available"]:
                results[model] = {"error": models[model]["reason"]}
                continue
            try:
                results[model] = service.check(text, model, assist=False)
                if reference is not None:
                    metrics = evaluate_edits([results[model]["predicted_text"]], [text], [reference])
                    results[model]["reference_metrics"] = {key: metrics[key] for key in ("detection", "correction")}
            except ModelUnavailable as exc:
                results[model] = {"error": str(exc)}
        completed = sum("error" not in item for item in results.values())
        return {"original_text": text, "comparison": completed >= 2, "results": results,
                "message": None if completed >= 2 else "Cần ít nhất hai mô hình chạy thành công để so sánh."}

    def augment(self, text, data):
        probability, count, seed = data.get("noise_prob", .3), data.get("count", 1), data.get("seed", 42)
        mode = data.get("type")
        if type(probability) not in (int, float) or not 0 <= probability <= 1:
            raise ValueError("Xác suất phải nằm trong khoảng 0 đến 1.")
        if type(count) is not int or not 1 <= count <= 20 or type(seed) is not int or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Số biến thể hoặc seed không hợp lệ.")
        if mode is not None and (not isinstance(mode, str) or mode not in NOISE_LABELS):
            raise ValueError("Loại lỗi không hợp lệ.")
        generator = VietnameseNoiseSynthesizer(seed)
        variants = []
        for _ in range(count):
            noisy, changes = generator.generate(text, probability, mode)
            variants.append({"input_text": noisy, "target_text": text, "differences": changes,
                             "errors_count": len(changes), "noise_type": mode or "mixed", "seed": seed})
        return {"clean_text": text, "noise_prob": probability, "variants": variants,
                "noisy_text": variants[0]["input_text"], "differences": variants[0]["differences"],
                "errors_count": variants[0]["errors_count"]}


def make_server(host="127.0.0.1", port=7860, root=ROOT):
    service = CorrectionService(root)
    server = ThreadingHTTPServer((host, port), SpellCheckerRequestHandler)
    server.service = service
    server.data = WebsiteData(root, service)
    return server


def run_app(port=7860, host="127.0.0.1"):
    with make_server(host, port) as server:
        print(f"VietSpell AI · http://{host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    run_app(args.port, args.host)
