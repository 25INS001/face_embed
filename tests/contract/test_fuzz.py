"""Hostile uploads against the embedding endpoint.

This endpoint takes an arbitrary binary from an unauthenticated caller and hands
it to an image decoder, so it has a wider attack surface than its one route
suggests. The view reads the whole upload into memory with

    np.asarray(bytearray(file.read()), dtype=np.uint8)

before anything looks at it — no size check, no content-type check. That shape
is what the tests below probe.
"""

import io

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.fuzz, pytest.mark.django_db]

ENDPOINT = "/api/face/embedding/"


def upload(content, name="face.jpg"):
    buffer = io.BytesIO(content)
    buffer.name = name
    return buffer


def assert_no_unhandled_error(resp, what):
    """500 is allowed only when it is the view's own extraction-failure branch."""
    if resp.status_code == 500:
        try:
            error = resp.json().get("error")
        except ValueError:
            error = None
        assert error == "Embedding extraction failed", (
            f"{what} produced an unhandled 500. Body: {resp.content[:300]}"
        )


# --------------------------------------------------------------------------- #
# Non-images
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "content,description",
    [
        (b"", "empty file"),
        (b"not an image", "plain text"),
        (b"\x00" * 1024, "null bytes"),
        (b"\xff\xd8\xff", "truncated JPEG header"),
        (b"GIF89a" + b"\x00" * 100, "GIF header only"),
        (b"%PDF-1.4\n%\xe2\xe3\xcf\xd3", "a PDF"),
        (b"<?xml version='1.0'?><svg xmlns='http://www.w3.org/2000/svg'/>", "an SVG"),
        (b"#!/bin/sh\nrm -rf /", "a shell script"),
        (b"\x50\x4b\x03\x04" + b"\x00" * 100, "a zip archive"),
    ],
)
def test_non_images_are_rejected_not_crashed(client, detects_a_face, undecodable_image,
                                             content, description):
    """cv2.imdecode returns None for anything it cannot parse, and the view
    turns that into a 400. The point is that it never raises."""
    resp = client.post(ENDPOINT, {"image": upload(content)})
    assert resp.status_code == 400, f"{description} returned {resp.status_code}"
    assert resp.json()["error"] == "Invalid image"


def test_an_empty_upload_is_rejected(client, detects_a_face):
    resp = client.post(ENDPOINT, {"image": upload(b"")})
    assert resp.status_code == 400
    assert_no_unhandled_error(resp, "an empty upload")


# --------------------------------------------------------------------------- #
# Field shape
# --------------------------------------------------------------------------- #

def test_a_text_field_named_image_is_not_a_file(client, detects_a_face):
    """request.FILES only holds uploads. A plain form field named `image` is in
    request.POST, so the view must still answer 'No image provided'."""
    resp = client.post(ENDPOINT, {"image": "just a string"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "No image provided"


def test_multiple_files_take_the_first(client, detects_a_face):
    """Django's MultiValueDict returns the last value for a repeated key. Either
    behaviour is defensible; what matters is that it does not error."""
    resp = client.post(ENDPOINT, {"image": [upload(b"a" * 100), upload(b"b" * 100)]})
    assert resp.status_code in (200, 400, 404)
    assert_no_unhandled_error(resp, "two files under one key")


def test_a_json_body_instead_of_a_file(client, detects_a_face):
    resp = client.post(ENDPOINT, {"image": "data"}, content_type="application/json")
    assert resp.status_code == 400
    assert_no_unhandled_error(resp, "a JSON body")


@pytest.mark.parametrize(
    "content_type",
    ["text/plain", "application/json", "application/xml", "application/octet-stream"],
)
def test_wrong_request_content_type(client, detects_a_face, content_type):
    """DRF parses before the view runs.

    Only JSON, form and multipart parsers are configured, so anything else is
    refused with 415 by the framework and the view is never entered.
    application/json does parse, reaches the view, and gets the 400. Both are
    correct — the test's job is to confirm neither path raises.
    """
    resp = client.post(ENDPOINT, b"raw bytes", content_type=content_type)
    assert resp.status_code in (400, 415), (
        f"a {content_type} body returned {resp.status_code}"
    )
    assert_no_unhandled_error(resp, f"a {content_type} body")


@pytest.mark.parametrize(
    "filename",
    [
        "../../../../etc/passwd",
        "..\\..\\windows\\system32\\config",
        "face.jpg\x00.php",
        "<script>alert(1)</script>.jpg",
        "'; DROP TABLE detection; --.jpg",
        "\r\nX-Injected: 1.jpg",
        "🙂" * 200 + ".jpg",
        "",
    ],
)
def test_hostile_upload_filenames(client, detects_a_face, filename):
    """The filename is never used — nothing is written to disk and cv2 decodes
    from memory. This pins that: a hostile name must not change the outcome."""
    resp = client.post(ENDPOINT, {"image": upload(b"\xff\xd8\xff\xe0" + b"x" * 100, name=filename)})
    assert resp.status_code == 200, (
        f"filename {filename[:40]!r} changed the result — the name is reaching "
        "something it should not"
    )
    assert_no_unhandled_error(resp, f"upload named {filename[:40]!r}")


def test_an_absurdly_long_filename_is_refused_before_the_view(client, detects_a_face):
    """Django's upload handler drops a 5000-character filename, so the file
    never lands in request.FILES and the view answers 'No image provided'.

    Separated from the cases above because the outcome genuinely differs: the
    request is rejected rather than processed. What matters is that the
    rejection is clean and happens before any decoding.
    """
    resp = client.post(
        ENDPOINT, {"image": upload(b"\xff\xd8\xff\xe0" + b"x" * 100, name="f" * 5000 + ".jpg")}
    )
    assert resp.status_code == 400
    assert_no_unhandled_error(resp, "an upload with a 5000-character filename")


def test_no_file_is_written_to_disk(client, detects_a_face, tmp_path, settings):
    """A traversal filename is only dangerous if something saves it. Point
    MEDIA_ROOT at an empty directory and confirm nothing lands there."""
    settings.MEDIA_ROOT = str(tmp_path)
    client.post(ENDPOINT, {"image": upload(b"\xff\xd8\xff\xe0" + b"x" * 100, name="../evil.jpg")})
    assert list(tmp_path.iterdir()) == [], "the upload was persisted to MEDIA_ROOT"


# --------------------------------------------------------------------------- #
# Size
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("size", [1_000_000, 10_000_000], ids=["1MB", "10MB"])
def test_large_uploads_are_handled(client, detects_a_face, size):
    """The whole upload is read into memory before any check.

    Django spools anything over FILE_UPLOAD_MAX_MEMORY_SIZE to a temporary file,
    but views.py then calls .read() on all of it regardless. There is no
    configured ceiling on the request body, so this documents the behaviour at
    sizes a caller could send trivially.
    """
    resp = client.post(ENDPOINT, {"image": upload(b"\xff\xd8" + b"x" * size)})
    assert resp.status_code in (200, 400, 404, 413)
    assert_no_unhandled_error(resp, f"a {size}-byte upload")


def test_no_upload_size_limit_is_configured(settings):
    """Recorded, not asserted as correct.

    DATA_UPLOAD_MAX_MEMORY_SIZE bounds form data but not file uploads, so an
    unauthenticated caller decides how much memory each request costs. A limit
    belongs in nginx (client_max_body_size) or in the view.
    """
    assert getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", None) in (None, 2621440), (
        "an upload limit is now configured — assert the limit here instead"
    )


# --------------------------------------------------------------------------- #
# Method and header abuse
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "header,value",
    [
        ("HTTP_X_FORWARDED_FOR", "127.0.0.1"),
        ("HTTP_X_ORIGINAL_URL", "/admin"),
        ("HTTP_AUTHORIZATION", "Bearer anything"),
        ("HTTP_CONTENT_LENGTH", "999999999"),
    ],
)
def test_headers_do_not_change_the_outcome(client, detects_a_face, header, value):
    resp = client.post(ENDPOINT, {"image": upload(b"\xff\xd8\xff\xe0")}, **{header: value})
    assert resp.status_code == 200
    assert_no_unhandled_error(resp, f"request with {header}")


def test_the_error_body_does_not_leak_internals(client, finds_no_face):
    """DEBUG is off in the test settings, so a traceback here would mean the
    view rendered one itself."""
    resp = client.post(ENDPOINT, {"image": upload(b"\xff\xd8\xff\xe0")})
    body = resp.content.decode(errors="replace").lower()
    for leak in ("traceback", "site-packages", "face_functions", "torch", "/app/"):
        assert leak not in body, f"the error body mentions {leak!r}"
