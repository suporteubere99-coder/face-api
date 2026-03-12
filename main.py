from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import shutil
import os

app = FastAPI()

def save_file(upload_file, path):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

@app.post("/compare")
async def compare_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)):

    img1_path = "img1.jpg"
    img2_path = "img2.jpg"

    save_file(img1, img1_path)
    save_file(img2, img2_path)

    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            distance_metric="cosine",
            enforce_detection=True,
            align=True,
            normalization="ArcFace"
        )

        distance = result["distance"]
        similarity = (1 - distance) * 100

        return {
            "similaridade": round(similarity, 2),
            "match": result["verified"],
            "distance": distance
        }

    except Exception as e:
        return {"erro": str(e)}

    finally:
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
