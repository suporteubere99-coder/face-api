import os
import math
import base64
import logging
from io import BytesIO
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("face_api")

MODEL_NAME = "buffalo_l"
DETECTION_THRESHOLD = 0.25
DETECTION_SIZE = (640, 640)

app = FastAPI(title="Face Similarity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Request: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Response: %s %s -> %s", request.method, request.url.path, response.status_code)
    return response

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Erro interno em %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "status": "failure",
            "message": "internal_server_error",
            "detail": str(exc),
        },
    )

face_app = FaceAnalysis(
    name=MODEL_NAME,
    allowed_modules=["detection", "recognition"],
    providers=["CPUExecutionProvider"],
)
face_app.prepare(
    ctx_id=-1,
    det_thresh=DETECTION_THRESHOLD,
    det_size=DETECTION_SIZE,
)

haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

CALIBRATION_POINTS = [
    (-0.20, 0),
    (-0.10, 2),
    (-0.05, 4),
    (0.00, 7),
    (0.02, 9),
    (0.04, 12),
    (0.06, 15),
    (0.08, 19),
    (0.10, 23),
    (0.12, 28),
    (0.14, 33),
    (0.16, 38),
    (0.18, 43),
    (0.20, 48),
    (0.23, 55),
    (0.26, 62),
    (0.30, 70),
    (0.35, 79),
    (0.40, 86),
    (0.45, 91),
    (0.50, 94),
    (0.55, 96),
    (0.60, 97),
    (0.70, 99),
    (0.80, 99),
]

def normalize_name(name: Optional[str]) -> str:
    return (name or "").strip()

def read_image(data: bytes) -> np.ndarray:
    pil = Image.open(BytesIO(data))
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def to_data_url(data: bytes, mime: Optional[str]) -> str:
    mime = mime or "image/jpeg"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def size_text(data: bytes) -> str:
    kb = max(1, round(len(data) / 1024))
    return f"{kb}KB"

def normalize_embedding(emb: np.ndarray) -> np.ndarray:
    emb = emb.astype("float32")
    norm = float(np.linalg.norm(emb))
    if norm == 0.0:
        return emb
    return emb / norm

def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on", "sim"}

def bbox_to_list(face: Any) -> List[float]:
    try:
        return [round(float(v), 2) for v in face.bbox]
    except Exception:
        return []

def det_score_of(face: Any) -> float:
    try:
        return round(float(getattr(face, "det_score", 0.0)), 6)
    except Exception:
        return 0.0

def upscale_if_needed(
    img: np.ndarray,
    min_side: int = 900,
    max_scale: float = 2.0,
) -> np.ndarray:
    h, w = img.shape[:2]
    m = min(h, w)
    if m >= min_side:
        return img
    scale = min(max_scale, float(min_side) / float(max(1, m)))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)

def clahe_variant(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)

def sharpen_variant(img: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    return cv2.filter2D(img, -1, kernel)

def try_faces(img: np.ndarray):
    try:
        return face_app.get(img)
    except Exception as e:
        logger.exception("Erro no face_app.get: %s", e)
        return []

def pick_best_face(faces: List[Any], img_shape: Tuple[int, int, int]):
    h, w = img_shape[:2]
    img_area = float(max(1, w * h))
    cx, cy = w / 2.0, h / 2.0

    best_face = None
    best_score = -10**9

    for f in faces:
        try:
            x1, y1, x2, y2 = [float(v) for v in f.bbox]
        except Exception:
            continue

        area = max(1.0, (x2 - x1) * (y2 - y1))
        fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        det = float(getattr(f, "det_score", 0.0))
        area_ratio = area / img_area
        center_dist = math.hypot(fx - cx, fy - cy) / max(w, h)

        score = area_ratio * 0.65 + det * 0.35 - center_dist * 0.20

        if score > best_score:
            best_score = score
            best_face = f

    return best_face

def detect_with_insight(img: np.ndarray):
    candidates = [img]

    up = upscale_if_needed(img)
    if up.shape != img.shape:
        candidates.append(up)

    candidates.append(clahe_variant(up))
    candidates.append(sharpen_variant(up))

    for cand in candidates:
        faces = try_faces(cand)
        if faces:
            face = pick_best_face(faces, cand.shape)
            if face is not None:
                emb = getattr(face, "normed_embedding", None)
                if emb is None:
                    emb = getattr(face, "embedding", None)
                if emb is None:
                    continue
                emb = normalize_embedding(np.asarray(emb, dtype=np.float32))
                return emb, face, cand

    return None, None, None

def detect_with_haar_then_insight(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes = haar_cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(80, 80),
    )

    if len(boxes) == 0:
        return None, None, None

    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])

    margin = 0.28
    ih, iw = img.shape[:2]
    x1 = max(0, int(round(x - w * margin)))
    y1 = max(0, int(round(y - h * margin)))
    x2 = min(iw, int(round(x + w + w * margin)))
    y2 = min(ih, int(round(y + h + h * margin)))

    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None, None

    emb, face, used = detect_with_insight(crop)
    if emb is not None:
        return emb, face, used

    crop_up = upscale_if_needed(crop, min_side=1000, max_scale=2.0)
    emb, face, used = detect_with_insight(crop_up)
    if emb is not None:
        return emb, face, used

    return None, None, None

def get_single_embedding(img: np.ndarray):
    emb, face, used = detect_with_insight(img)
    if emb is not None:
        return emb, face, used

    emb, face, used = detect_with_haar_then_insight(img)
    if emb is not None:
        return emb, face, used

    return None, None, None

def get_embedding(img: np.ndarray):
    emb1, face1, used_img = get_single_embedding(img)
    if emb1 is None:
        return None, None, None

    embs = [emb1]

    try:
        flip = cv2.flip(used_img, 1)
        emb2, _, _ = get_single_embedding(flip)
        if emb2 is not None:
            embs.append(emb2)
    except Exception:
        logger.exception("Erro ao gerar embedding com flip")

    final_emb = normalize_embedding(np.mean(np.stack(embs, axis=0), axis=0))
    return final_emb, face1, used_img

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_embedding(a)
    b = normalize_embedding(b)
    return float(np.dot(a, b))

def score_from_cosine(raw_cosine: float) -> int:
    pts = sorted(CALIBRATION_POINTS, key=lambda x: x[0])

    if raw_cosine <= pts[0][0]:
        return max(0, min(99, int(round(pts[0][1]))))

    if raw_cosine >= pts[-1][0]:
        return max(0, min(99, int(round(pts[-1][1]))))

    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]

        if x1 <= raw_cosine <= x2:
            t = 0.0 if x2 == x1 else (raw_cosine - x1) / (x2 - x1)
            score = y1 + t * (y2 - y1)
            return max(0, min(99, int(round(score))))

    return 0

async def extract_one(file: UploadFile, include_image: bool = False) -> dict:
    filename = normalize_name(file.filename)

    try:
        file_bytes = await file.read()
        img = read_image(file_bytes)
    except Exception as e:
        logger.exception("Erro lendo imagem %s", filename)
        return {
            "name": filename,
            "status": "failure",
            "reason": f"read_error: {str(e)}",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": [],
            "det_score": 0.0,
            "size": "0KB",
            "image": None,
        }

    emb, face, _ = get_embedding(img)
    if emb is None:
        return {
            "name": filename,
            "status": "failure",
            "reason": "no_face_detected",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": [],
            "det_score": 0.0,
            "size": size_text(file_bytes),
            "image": to_data_url(file_bytes, file.content_type) if include_image else None,
        }

    return {
        "name": filename,
        "status": "success",
        "reason": None,
        "embedding": emb.tolist(),
        "embedding_dim": int(emb.shape[0]),
        "bbox": bbox_to_list(face),
        "det_score": det_score_of(face),
        "size": size_text(file_bytes),
        "image": to_data_url(file_bytes, file.content_type) if include_image else None,
    }

@app.get("/")
def home():
    return {
        "status": "ok",
        "message": "Face Similarity API online",
        "model_name": MODEL_NAME,
    }

@app.get("/health")
def health():
    loaded_models = {}
    try:
        for task, model in face_app.models.items():
            loaded_models[task] = getattr(model, "model_file", None)
    except Exception:
        pass

    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "file": __file__,
        "cwd": os.getcwd(),
        "loaded_models": loaded_models,
    }

@app.post("/extract-embedding")
async def extract_embedding(
    image: UploadFile = File(...),
    include_image: Optional[bool] = Form(False),
):
    include = parse_bool(include_image, default=False)
    result = await extract_one(image, include_image=include)

    if result["status"] != "success":
        return JSONResponse(
            status_code=200,
            content={
                "status": "failure",
                "message": "Nenhum rosto detectado na imagem.",
                **result,
            },
        )

    return {
        "status": "success",
        **result,
    }

@app.post("/extract-embeddings-batch")
async def extract_embeddings_batch(
    images: List[UploadFile] = File(...),
    include_image: Optional[bool] = Form(False),
):
    include = parse_bool(include_image, default=False)
    results = []

    for item in images:
        results.append(await extract_one(item, include_image=include))

    success_count = sum(1 for r in results if r["status"] == "success")
    failure_count = len(results) - success_count

    return {
        "status": "success",
        "total": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "results": results,
    }

@app.post("/compare-two")
async def compare_two(
    base_file: UploadFile = File(...),
    compare_file: UploadFile = File(...),
    requested_similarity: Optional[float] = Form(None),
    include_images: Optional[bool] = Form(False),
):
    include = parse_bool(include_images, default=False)

    base_result = await extract_one(base_file, include_image=include)
    if base_result["status"] != "success":
        return JSONResponse(
            status_code=200,
            content={
                "status": "failure",
                "message": "Nenhum rosto detectado na imagem base.",
                "base": base_result,
            },
        )

    compare_result = await extract_one(compare_file, include_image=include)
    if compare_result["status"] != "success":
        return JSONResponse(
            status_code=200,
            content={
                "status": "failure",
                "message": "Nenhum rosto detectado na imagem de comparação.",
                "base": base_result,
                "compare": compare_result,
            },
        )

    base_emb = np.array(base_result["embedding"], dtype=np.float32)
    compare_emb = np.array(compare_result["embedding"], dtype=np.float32)

    raw_cos = cosine_similarity(base_emb, compare_emb)
    similarity = score_from_cosine(raw_cos)

    meets_requested_similarity = None
    if requested_similarity is not None:
        meets_requested_similarity = similarity >= float(requested_similarity)

    return {
        "status": "success",
        "requested_similarity": requested_similarity,
        "meets_requested_similarity": meets_requested_similarity,
        "similarity": similarity,
        "raw_cosine": round(raw_cos, 6),
        "base": {
            "name": base_result["name"],
            "bbox": base_result["bbox"],
            "det_score": base_result["det_score"],
            "size": base_result["size"],
            "image": base_result["image"],
        },
        "compare": {
            "name": compare_result["name"],
            "bbox": compare_result["bbox"],
            "det_score": compare_result["det_score"],
            "size": compare_result["size"],
            "image": compare_result["image"],
        },
    }

@app.post("/compare.php", include_in_schema=False)
async def compare_legacy(request: Request):
    form = await request.form()

    base_file = form.get("base_file")
    compare_files = form.getlist("compare_files[]")

    if not compare_files:
        compare_files = form.getlist("compare_files")

    if not compare_files:
        single_compare = form.get("compare_file")
        if single_compare and hasattr(single_compare, "read"):
            compare_files = [single_compare]

    include_images = parse_bool(form.get("include_images"), default=True)

    requested_similarity = None
    raw_req = form.get("requested_similarity")
    if raw_req is not None:
        try:
            requested_similarity = float(str(raw_req).replace(",", "."))
        except Exception:
            requested_similarity = None

    if not base_file or not hasattr(base_file, "read"):
        return JSONResponse(
            status_code=400,
            content={
                "status": "failure",
                "message": "Campo base_file não enviado.",
            },
        )

    compare_files = [f for f in compare_files if hasattr(f, "read")]
    if not compare_files:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failure",
                "message": "Envie pelo menos um arquivo em compare_files[] ou compare_file.",
            },
        )

    base_result = await extract_one(base_file, include_image=include_images)
    if base_result["status"] != "success":
        return JSONResponse(
            status_code=200,
            content={
                "status": "failure",
                "message": "Nenhum rosto detectado na imagem base.",
                "target_image": base_result.get("image"),
                "results": [],
            },
        )

    base_emb = np.array(base_result["embedding"], dtype=np.float32)
    results = []

    for item in compare_files:
        item_result = await extract_one(item, include_image=include_images)

        if item_result["status"] != "success":
            results.append(
                {
                    "name": item_result["name"],
                    "filename": item_result["name"],
                    "similarity": 0,
                    "raw_cosine": 0.0,
                    "size": item_result["size"],
                    "source": "api",
                    "status": "failure",
                    "reason": item_result["reason"],
                    "image": item_result.get("image"),
                }
            )
            continue

        item_emb = np.array(item_result["embedding"], dtype=np.float32)
        raw_cos = cosine_similarity(base_emb, item_emb)
        similarity = score_from_cosine(raw_cos)

        meets_requested_similarity = None
        if requested_similarity is not None:
            meets_requested_similarity = similarity >= requested_similarity

        results.append(
            {
                "name": item_result["name"],
                "filename": item_result["name"],
                "similarity": similarity,
                "raw_cosine": round(raw_cos, 6),
                "size": item_result["size"],
                "source": "api",
                "status": "success",
                "meets_requested_similarity": meets_requested_similarity,
                "image": item_result.get("image"),
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "status": "success",
        "requested_similarity": requested_similarity,
        "results": results,
        "target_image": base_result.get("image"),
    }
