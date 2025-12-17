import cv2
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .face_functions import detect_face, extract_face_embedding


class FaceEmbeddingView(APIView):
    """
    POST image → returns face embedding
    """

    def post(self, request):
        if 'image' not in request.FILES:
            return Response(
                {"error": "No image provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        file = request.FILES['image']

        # Read image into OpenCV format
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return Response(
                {"error": "Invalid image"},
                status=status.HTTP_400_BAD_REQUEST
            )

        face = detect_face(image)
        if face is None:
            return Response(
                {"error": "No face detected"},
                status=status.HTTP_404_NOT_FOUND
            )

        embedding = extract_face_embedding(face)
        if embedding is None:
            return Response(
                {"error": "Embedding extraction failed"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({
            "embedding": embedding,
            "dimension": len(embedding)
        })
