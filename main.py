from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import math

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str
    ruler_length_um: float

@app.post("/process-image")
def process_image(data: ImageRequest):
    try:
        # Extrair imagem do base64
        header, encoded = data.image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Erro ao decodificar a imagem")

        # Conversões e pré-processamento
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, brain_bin = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        brain_bin = cv2.morphologyEx(brain_bin, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(brain_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detecção da régua
        roi_ruler = gray[-150:, :]
        _, ruler_bin = cv2.threshold(roi_ruler, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        lines = cv2.HoughLinesP(ruler_bin, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

        scale = None
        ruler_coords_px = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}
        if lines is not None:
            horizontal_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))
                if angle < 5:
                    horizontal_lines.append(line)

            if len(horizontal_lines) >= 2:
                x_coords = [x for line in horizontal_lines for x in [line[0][0], line[0][2]]]
                x_min, x_max = min(x_coords), max(x_coords)
                total_dist_px = x_max - x_min
                scale = data.ruler_length_um / total_dist_px  # um/pixel
                ruler_coords_px = {
                    "x1": int(x_min),
                    "y1": img.shape[0] - 150,
                    "x2": int(x_max),
                    "y2": img.shape[0] - 150
                }

        # Detecção de organoides
        organoids = []
        for cnt in contours:
            area_px = cv2.contourArea(cnt)
            if area_px < 150:
                continue
            area_um2 = (area_px * scale * scale) if scale else 0
            if scale and area_um2 > 50000:  # ~5 mm² em um²
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            diameter_um = math.sqrt(4 * area_um2 / math.pi) if scale else 0
            radius_um = diameter_um / 2
            volume_um3 = (4/3) * math.pi * (radius_um ** 3) if scale else 0

            organoids.append({
                "x_px": x,
                "y_px": y,
                "width_px": w,
                "height_px": h,
                "diameter_um": round(diameter_um, 2),
                "area_um2": round(area_um2, 2),
                "volume_um3": round(volume_um3, 2)
            })

            # Desenhar contornos
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Codificar imagem com marcações
        _, buffer = cv2.imencode('.png', img)
        processed_base64 = base64.b64encode(buffer).decode("utf-8")
        processed_image_base64 = f"data:image/png;base64,{processed_base64}"

        return JSONResponse(content={
            "scale_factor": scale if scale else 0,
            "ruler_coords_px": ruler_coords_px,
            "organoids": organoids,
            "processed_image_base64": processed_image_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "API de medição ativa"}
