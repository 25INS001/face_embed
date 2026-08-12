"""POST /api/face/embedding/ — the only endpoint face_embed exposes.

This service is one half of the face pipeline. The other half is auth-service's
/face/register and /face/match, which store and compare what this returns. So
the response shape is a cross-service contract, not an internal detail: if the
dimension changes here, every embedding already in the auth-service database
becomes uncomparable.

The dimension is therefore asserted as a fixed number rather than as
len(embedding), which would pass no matter what the model returned.
"""

import pytest

pytestmark = [pytest.mark.contract, pytest.mark.django_db]

ENDPOINT = "/api/face/embedding/"
EXPECTED_DIMENSION = 512


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #

def test_an_image_with_a_face_returns_an_embedding(client, detects_a_face, image_file):
    resp = client.post(ENDPOINT, {"image": image_file()})
    assert resp.status_code == 200
    assert "embedding" in resp.json()


def test_the_embedding_has_the_dimension_auth_service_expects(client, detects_a_face, image_file):
    """auth-service stores a 512-float vector and rejects anything else.

    Hardcoded on purpose: reading the length off the response would make this
    test agree with any model, including a wrong one.
    """
    body = client.post(ENDPOINT, {"image": image_file()}).json()
    assert body["dimension"] == EXPECTED_DIMENSION
    assert len(body["embedding"]) == EXPECTED_DIMENSION


def test_the_reported_dimension_matches_the_vector(client, detects_a_face, image_file):
    body = client.post(ENDPOINT, {"image": image_file()}).json()
    assert body["dimension"] == len(body["embedding"])


def test_the_embedding_is_a_list_of_numbers(client, detects_a_face, image_file):
    """It is serialised straight to JSON and posted to auth-service, so numpy
    scalars or nested arrays would break the consumer, not this service."""
    body = client.post(ENDPOINT, {"image": image_file()}).json()
    assert isinstance(body["embedding"], list)
    assert all(isinstance(v, (int, float)) for v in body["embedding"])


def test_the_response_is_json(client, detects_a_face, image_file):
    resp = client.post(ENDPOINT, {"image": image_file()})
    assert resp["Content-Type"].startswith("application/json")


@pytest.mark.parametrize("filename", ["face.jpg", "face.png", "face.bmp", "noextension", "FACE.JPEG"])
def test_the_filename_does_not_matter(client, detects_a_face, image_file, filename):
    """Decoding is by content, not by name — cv2.imdecode never sees the name."""
    resp = client.post(ENDPOINT, {"image": image_file(name=filename)})
    assert resp.status_code == 200


# --------------------------------------------------------------------------- #
# The four failure branches
# --------------------------------------------------------------------------- #

def test_no_file_is_400(client, detects_a_face):
    resp = client.post(ENDPOINT, {})
    assert resp.status_code == 400
    assert resp.json()["error"] == "No image provided"


def test_a_differently_named_field_is_400(client, detects_a_face, image_file):
    """The view reads request.FILES['image'] specifically."""
    resp = client.post(ENDPOINT, {"file": image_file()})
    assert resp.status_code == 400
    assert resp.json()["error"] == "No image provided"


def test_an_undecodable_image_is_400(client, detects_a_face, undecodable_image, image_file):
    resp = client.post(ENDPOINT, {"image": image_file(b"this is not an image")})
    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid image"


def test_an_image_with_no_face_is_404(client, finds_no_face, image_file):
    resp = client.post(ENDPOINT, {"image": image_file()})
    assert resp.status_code == 404
    assert resp.json()["error"] == "No face detected"


def test_a_failed_extraction_is_500(client, embedding_fails, image_file):
    """Distinct from 'no face': the face was found, the model failed.

    A caller can retry a 500 and should not retry a 404, so collapsing the two
    would make the client's behaviour wrong in one direction or the other.
    """
    resp = client.post(ENDPOINT, {"image": image_file()})
    assert resp.status_code == 500
    assert resp.json()["error"] == "Embedding extraction failed"


def test_the_failure_branches_have_distinct_status_codes(client, image_file, detects_a_face,
                                                         finds_no_face):
    """finds_no_face is applied after detects_a_face, so this reads the 404."""
    assert client.post(ENDPOINT, {"image": image_file()}).status_code == 404
    assert client.post(ENDPOINT, {}).status_code == 400


# --------------------------------------------------------------------------- #
# Method and posture
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["get", "put", "patch", "delete"])
def test_only_post_is_allowed(client, method):
    resp = getattr(client, method)(ENDPOINT)
    assert resp.status_code == 405


def test_the_endpoint_is_unauthenticated(client, detects_a_face, image_file):
    """Like bucket, face_embed declares no permission_classes and settings.py
    has no REST_FRAMEWORK block, so DRF's AllowAny default applies.

    This one matters more than bucket's: the endpoint runs GPU inference. An
    unauthenticated caller can spend the model's time at will, and nginx routes
    /face/ from the public host.
    """
    resp = client.post(ENDPOINT, {"image": image_file()})
    assert resp.status_code == 200, (
        "the endpoint now requires a credential — replace this test with one "
        "that asserts an anonymous caller is refused"
    )


def test_no_default_permission_class_is_configured():
    from django.conf import settings

    assert "DEFAULT_PERMISSION_CLASSES" not in getattr(settings, "REST_FRAMEWORK", {})


def test_the_view_declares_no_permission_classes():
    from rest_framework.permissions import AllowAny

    from detection.views import FaceEmbeddingView

    effective = getattr(FaceEmbeddingView, "permission_classes", [AllowAny])
    assert list(effective) in ([AllowAny], [])
