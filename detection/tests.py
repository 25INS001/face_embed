from unittest.mock import patch
from io import BytesIO

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from PIL import Image
from rest_framework.test import APIClient


class FaceEmbeddingApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_requires_image_file(self):
        response = self.client.post("/api/face/embedding/", {}, format="multipart")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["error"], "No image provided")

    def test_rejects_invalid_image_bytes(self):
        bad_file = SimpleUploadedFile("not-an-image.txt", b"not an image", content_type="text/plain")

        response = self.client.post("/api/face/embedding/", {"image": bad_file}, format="multipart")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["error"], "Invalid image")

    @patch("detection.views.extract_face_embedding", return_value=[0.1, 0.2, 0.3])
    @patch("detection.views.detect_face", return_value=np.zeros((112, 112, 3), dtype=np.uint8))
    def test_returns_embedding_dimension(self, mock_detect, mock_extract):
        image_bytes = BytesIO()
        Image.new("RGB", (4, 4), color=(255, 255, 255)).save(image_bytes, format="PNG")
        image = SimpleUploadedFile("face.png", image_bytes.getvalue(), content_type="image/png")

        response = self.client.post("/api/face/embedding/", {"image": image}, format="multipart")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["embedding"], [0.1, 0.2, 0.3])
        self.assertEqual(response.data["dimension"], 3)
        mock_detect.assert_called_once()
        mock_extract.assert_called_once()
