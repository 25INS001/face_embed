"""Fixtures for the face_embed suite.

One endpoint, one job: take an uploaded image and return an embedding. The
interesting surface is not the maths — it is the four ways the request can fail
and the status code each one maps to, which views.py spells out explicitly:

    no 'image' in request.FILES  -> 400 No image provided
    cv2.imdecode returns None    -> 400 Invalid image
    detect_face returns None     -> 404 No face detected
    extract_face_embedding None  -> 500 Embedding extraction failed

Each fixture below lets a test choose one of those branches without loading a
model.
"""

import io

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="run tests against a running face_embed with the real model loaded",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--live"):
        return
    skip = pytest.mark.skip(reason="needs a running face_embed; pass --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


EMBEDDING_DIMENSION = 512


@pytest.fixture
def embedding():
    """A deterministic 512-float embedding, the shape auth-service expects."""
    return [round(i / EMBEDDING_DIMENSION, 6) for i in range(EMBEDDING_DIMENSION)]


@pytest.fixture
def detects_a_face(monkeypatch, embedding):
    """The happy path: a face is found and an embedding is produced."""
    import numpy as np

    from detection import views

    monkeypatch.setattr(views, "detect_face", lambda image: np.zeros((112, 112, 3), dtype=np.uint8))
    monkeypatch.setattr(views, "extract_face_embedding", lambda face: list(embedding))
    return embedding


@pytest.fixture
def finds_no_face(monkeypatch):
    from detection import views

    monkeypatch.setattr(views, "detect_face", lambda image: None)
    return None


@pytest.fixture
def embedding_fails(monkeypatch):
    import numpy as np

    from detection import views

    monkeypatch.setattr(views, "detect_face", lambda image: np.zeros((112, 112, 3), dtype=np.uint8))
    monkeypatch.setattr(views, "extract_face_embedding", lambda face: None)
    return None


@pytest.fixture
def undecodable_image(monkeypatch):
    """Make cv2.imdecode return None, the 'Invalid image' branch."""
    from detection import views

    monkeypatch.setattr(views.cv2, "imdecode", lambda buf, flags: None)


@pytest.fixture
def image_file():
    """An uploadable file object. The bytes are never really decoded."""

    def make(content=b"\xff\xd8\xff\xe0fake-jpeg-bytes", name="face.jpg"):
        buffer = io.BytesIO(content)
        buffer.name = name
        return buffer

    return make


@pytest.fixture(scope="session")
def live_base_url():
    import os

    return os.getenv("FACE_EMBED_BASE_URL", "http://localhost:8000").rstrip("/")
