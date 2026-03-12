from fastapi import FastAPI, File, UploadFile
from deepface import DeepFace
import shutil
import os
import cv2
import numpy as np
from typing import Dict

app = FastAPI()

# Função para salvar o arquivo de imagem
def save_file(upload_file, path):
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

# Função para redimensionar a imagem para 224x224 para melhorar a performance
def preprocess_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Redimensionar imagem para 224x224
    return img

@app.post("/compare")
async def compare_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)) -> Dict:
    # Caminhos temporários para salvar as imagens
    img1_path = "img1.jpg"
    img2_path = "img2.jpg"

    # Salvar as imagens no servidor
    save_file(img1, img1_path)
    save_file(img2, img2_path)

    try:
        # Pré-processar as imagens
        img1_processed = preprocess_image(img1_path)
        img2_processed = preprocess_image(img2_path)

        # Salvar as imagens processadas temporariamente
        cv2.imwrite("img1_processed.jpg", img1_processed)
        cv2.imwrite("img2_processed.jpg", img2_processed)

        # Verificar a similaridade entre as faces usando DeepFace com o modelo ArcFace
        result = DeepFace.verify(
            img1_path="img1_processed.jpg",
            img2_path="img2_processed.jpg",
            model_name="ArcFace",  # Modelo ArcFace para alta precisão
            detector_backend="retinaface",  # Detecção de faces com RetinaFace
            distance_metric="cosine",  # Métrica de distância para comparar faces
            enforce_detection=True,  # Garantir que as faces sejam detectadas
            align=True,  # Alinhar as faces antes de comparar
            normalization="ArcFace"  # Normalização ArcFace para maior precisão
        )

        # Calcular a similaridade (quanto mais próximo de 100, mais similares)
        distance = result["distance"]
        similarity = (1 - distance) * 100  # Calcular a porcentagem de similaridade

        # Retornar o resultado
        return {
            "similaridade": round(similarity, 2),  # Similaridade em porcentagem
            "match": result["verified"],  # Verificação de correspondência
            "distance": distance  # Distância entre as faces
        }

    except Exception as e:
        return {"erro": str(e)}  # Caso haja algum erro no processo

    finally:
        # Remover os arquivos temporários
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
        if os.path.exists("img1_processed.jpg"):
            os.remove("img1_processed.jpg")
        if os.path.exists("img2_processed.jpg"):
            os.remove("img2_processed.jpg")
