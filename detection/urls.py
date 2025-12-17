from django.urls import path
from .views import FaceEmbeddingView

urlpatterns = [
    path('face/embedding/', FaceEmbeddingView.as_view()),
]
