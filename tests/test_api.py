"""API tests using a fake detector, so they run without model weights. Run: pytest -q"""
import io
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main  # noqa: E402
from app.reasoning import ReasoningConfig, ReasoningEngine  # noqa: E402
from tests.test_reasoning import FakeDetector, det, sharp_image  # noqa: E402


def jpeg_bytes() -> bytes:
    buf = io.BytesIO()
    sharp_image().save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "weights/does_not_exist.pt")
    monkeypatch.delenv("WEIGHTS_URL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with TestClient(main.app) as c:
        fake = FakeDetector([det(500, 0.9), det(100, 0.8, x1=300, x2=400)])
        fake.names = {0: "10", 1: "100", 2: "500"}
        fake.weights_name = "fake.pt"
        main.STATE["detector"] = fake
        main.STATE["engine"] = ReasoningEngine(fake, None, ReasoningConfig())
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_detect_ok(client):
    r = client.post("/detect", files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and body["detections"][0]["value"] == 500
    assert "X-Request-ID" in r.headers


def test_detect_rejects_non_image(client):
    r = client.post("/detect", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_detect_rejects_large_upload(client):
    main.STATE["settings"].max_upload_bytes = 100
    r = client.post("/detect", files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 413


def test_detect_503_without_model(client):
    main.STATE["detector"] = None
    r = client.post("/detect", files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 503


def test_ask_total(client):
    r = client.post("/ask", data={"question": "How much money is here?"},
                    files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["route"] == "detector" and body["evidence"]["confirmed_total"] == 600


def test_ask_out_of_scope_skips_detector(client):
    r = client.post("/ask", data={"question": "Is this note fake?"},
                    files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    body = r.json()
    assert body["status"] == "insufficient_information" and body["detector_called"] is False


def test_ask_without_image(client):
    r = client.post("/ask", data={"question": "How many notes?"})
    assert r.status_code == 200 and r.json()["status"] == "insufficient_information"


def test_ask_requires_question(client):
    r = client.post("/ask", files={"file": ("notes.jpg", jpeg_bytes(), "image/jpeg")})
    assert r.status_code == 422
