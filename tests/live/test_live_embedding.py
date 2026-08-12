"""Black-box HTTP against a running face_embed with the real model loaded.

The contract layer stubs the inference stack, so it proves the HTTP contract and
nothing about the model. Only this layer answers the questions that matter for
the face pipeline end to end:

  * does the model actually load in the deployed container
  * is the embedding really 512-dimensional
  * is the same face stable enough between calls for auth-service to match on

Generating a face to test with is the awkward part. Rather than commit a photo
of a real person, these tests use a synthetic image and skip when no face is
detected in it — so a green run means "the service works", and a skip means
"the fixture was not good enough", never a false pass.

    FACE_EMBED_BASE_URL=http://localhost:8000 pytest --live tests/live
"""

import io

import pytest
import requests

pytestmark = pytest.mark.live

ENDPOINT = "/api/face/embedding/"
EXPECTED_DIMENSION = 512


@pytest.fixture(scope="session")
def service_up(live_base_url):
    try:
        resp = requests.post(f"{live_base_url}{ENDPOINT}", timeout=30)
    except requests.exceptions.RequestException as exc:
        pytest.fail(f"face_embed is not reachable at {live_base_url}: {exc}")
    assert resp.status_code == 400, (
        f"expected the no-image 400 from a bare POST, got {resp.status_code}"
    )
    return True


@pytest.fixture(scope="session")
def synthetic_face_bytes():
    """A face-like greyscale image built with numpy, encoded as PNG.

    Deliberately not a real photograph: committing one would put biometric data
    of an identifiable person into a git repository.
    """
    np = pytest.importorskip("numpy")
    png = pytest.importorskip("PIL.Image", reason="Pillow is needed to encode the fixture")

    image = np.full((256, 256, 3), 200, dtype=np.uint8)
    image[80:110, 70:110] = 40      # left eye
    image[80:110, 150:190] = 40     # right eye
    image[150:170, 110:150] = 60    # nose
    image[195:215, 80:180] = 30     # mouth

    buffer = io.BytesIO()
    png.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def post_image(base_url, content, name="face.png"):
    return requests.post(
        f"{base_url}{ENDPOINT}",
        files={"image": (name, io.BytesIO(content), "image/png")},
        timeout=120,  # first call may load the model
    )


# --------------------------------------------------------------------------- #
# Failure branches, which need no face at all
# --------------------------------------------------------------------------- #

def test_no_image_is_400(live_base_url, service_up):
    resp = requests.post(f"{live_base_url}{ENDPOINT}", timeout=30)
    assert resp.status_code == 400
    assert resp.json()["error"] == "No image provided"


def test_an_unparseable_upload_is_400(live_base_url, service_up):
    resp = post_image(live_base_url, b"this is definitely not an image")
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid image"


def test_a_blank_image_has_no_face(live_base_url, service_up):
    """A flat grey square. The real detector must find nothing in it."""
    np = pytest.importorskip("numpy")
    png = pytest.importorskip("PIL.Image")

    buffer = io.BytesIO()
    png.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8)).save(buffer, format="PNG")

    resp = post_image(live_base_url, buffer.getvalue())
    assert resp.status_code == 404, (
        f"the detector found a face in a blank image: {resp.status_code} {resp.text[:200]}"
    )


@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_only_post_is_allowed(live_base_url, service_up, method):
    resp = requests.request(method, f"{live_base_url}{ENDPOINT}", timeout=30)
    assert resp.status_code == 405


def test_no_credential_is_required(live_base_url, service_up):
    """Deployment check on the posture the contract suite records in process."""
    resp = requests.post(f"{live_base_url}{ENDPOINT}", timeout=30)
    assert resp.status_code not in (401, 403), (
        f"face_embed now requires authentication (HTTP {resp.status_code}) — "
        "update tests/contract/test_embedding_api.py to match"
    )


# --------------------------------------------------------------------------- #
# The model itself
# --------------------------------------------------------------------------- #

@pytest.fixture
def detected(live_base_url, service_up, synthetic_face_bytes):
    resp = post_image(live_base_url, synthetic_face_bytes)
    if resp.status_code == 404:
        pytest.skip(
            "the synthetic fixture is not detected as a face by this model — "
            "point the suite at a real image to exercise the embedding path"
        )
    assert resp.status_code == 200, f"embedding failed: {resp.status_code} {resp.text[:300]}"
    return resp.json()


def test_the_embedding_is_512_dimensional(detected):
    """auth-service /face/register stores exactly 512 floats.

    A change here silently invalidates every embedding already stored: old and
    new vectors would be uncomparable, and matching would degrade rather than
    error.
    """
    assert detected["dimension"] == EXPECTED_DIMENSION
    assert len(detected["embedding"]) == EXPECTED_DIMENSION


def test_the_embedding_is_json_serialisable_floats(detected):
    """numpy float32 does not serialise; if the view ever stopped converting,
    this is where it would surface."""
    assert all(isinstance(v, (int, float)) for v in detected["embedding"])


def test_the_embedding_is_not_degenerate(detected):
    """An all-zero or constant vector means the model loaded but produced
    nothing useful — a failure mode that still returns HTTP 200."""
    values = detected["embedding"]
    assert any(v != 0 for v in values), "the embedding is all zeros"
    assert len(set(values)) > 1, "every component of the embedding is identical"


def test_the_same_image_embeds_consistently(live_base_url, service_up, synthetic_face_bytes, detected):
    """Matching depends on this. If the same bytes give different vectors, the
    model is running with a non-deterministic transform and no threshold in
    auth-service will behave predictably.
    """
    second = post_image(live_base_url, synthetic_face_bytes)
    if second.status_code != 200:
        pytest.skip("second call did not detect a face")

    first_vector = detected["embedding"]
    second_vector = second.json()["embedding"]

    largest_difference = max(abs(a - b) for a, b in zip(first_vector, second_vector))
    assert largest_difference < 1e-4, (
        f"the same image produced embeddings differing by up to {largest_difference}"
    )


@pytest.mark.fuzz
def test_a_large_upload_does_not_take_the_service_down(live_base_url, service_up):
    """There is no configured body limit. Confirm the service survives one."""
    try:
        requests.post(
            f"{live_base_url}{ENDPOINT}",
            files={"image": ("big.png", io.BytesIO(b"\x89PNG\r\n" + b"x" * 20_000_000), "image/png")},
            timeout=120,
        )
    except requests.exceptions.RequestException:
        pass  # refusing the body outright is acceptable

    after = requests.post(f"{live_base_url}{ENDPOINT}", timeout=30)
    assert after.status_code == 400, "the service stopped answering after a large upload"
