# app/detection/face_functions.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os
import sys
from transformers import AutoModel
from huggingface_hub import hf_hub_download

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------
# DOWNLOAD HF REPO (Fixed Repo ID)
# ---------------------------------------------------
def download(repo_id, path):
    os.makedirs(path, exist_ok=True)
    files = ['config.json', 'wrapper.py', 'model.safetensors']
    try:
        hf_hub_download(repo_id, 'files.txt', local_dir=path, local_dir_use_symlinks=False)
        with open(os.path.join(path, 'files.txt'), 'r') as f:
            extra_files = f.read().split('\n')
        files += extra_files
    except Exception:
        pass 
        
    for file in files:
        if file.strip() == "": continue
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            try:
                print(f"Downloading {file}...")
                hf_hub_download(repo_id, file, local_dir=path, local_dir_use_symlinks=False)
            except Exception as e:
                print(f"Warning: Could not download {file}: {e}")

def load_model_from_local_path(path):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)

    try:
        model = AutoModel.from_pretrained(path, trust_remote_code=True)
        model.eval().to(device)
    except Exception as e:
        print(f"CRITICAL: Failed to load AdaFace model from {path}")
        raise e
    finally:
        os.chdir(cwd)
        sys.path.pop(0)
    
    return model

def load_adaface_model():
    repo_id = "minchul/cvlface_adaface_ir101_webface12m"
    save_path = os.path.join(os.getcwd(), "models", "adaface_model")

    if not os.path.exists(save_path):
        print(f"Downloading AdaFace model from {repo_id}...")
        download(repo_id, save_path)

    return load_model_from_local_path(save_path)

adaface_model = None

def get_adaface_model():
    global adaface_model
    if adaface_model is None:
        print("Initializing Face Recognition Model...")
        adaface_model = load_adaface_model()
        print("Face Recognition Model Loaded.")
    return adaface_model

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---------------------------------------------------
# DETECT FACE (Optimized)
# ---------------------------------------------------
def detect_face(image_np):
    """
    Detects face in the provided image (person crop).
    Optimization: Only scans the top 50% of the image.
    """
    h_full, w_full = image_np.shape[:2]
    
    # OPTIMIZATION: People usually have heads in the top half of their body.
    # We slice the top 50% to reduce the area Haar Cascade has to scan.
    scan_h = int(h_full * 1.0)
    
    # Sanity check: if image is too small, scan whole thing
    if scan_h < 50: 
        scan_h = h_full

    roi = image_np[0:scan_h, 0:w_full]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    # Get the biggest face
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    # Add padding
    pad = int(0.2 * max(w, h))
    x, y = max(0, x - pad), max(0, y - pad)
    w, h = w + 2 * pad, h + 2 * pad
    
    # Ensure crop is within the ROI bounds (remember we are in 'roi', not 'image_np')
    x = max(0, min(w_full, x))
    y = max(0, min(scan_h, y))
    w = min(w_full - x, w)
    h = min(scan_h - y, h)

    if w == 0 or h == 0: return None

    # Crop from the ROI (which matches the top of image_np)
    face = roi[y:y + h, x:x + w]
    return face

# ---------------------------------------------------
# EXTRACT EMBEDDING
# ---------------------------------------------------
def extract_face_embedding(face_np):
    try:
        img = face_np
        face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = get_adaface_model()(face_tensor)
            emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().flatten().tolist()
    except Exception as e:
        print("AdaFace Inference Error:", e)
        return None
