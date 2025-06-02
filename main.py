from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import io

app = FastAPI()

# Permitir frontend acessar
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Substitua por seu domínio se quiser restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- MODELOS --------
class DetectionParams(BaseModel):
    threshold: int = 180
    min_area: int = 150
    max_area: float = 2.0
    blur_size: int = 5

class ImageRequest(BaseModel):
    image_base64: str
    ruler_length_um: float
    params: Optional[DetectionParams] = DetectionParams()

# -------- ENDPOINT PRINCIPAL --------
@app.post("/process-image")
def process_image(req: ImageRequest):
    # Decode base64 image
    image_data = req.image_base64.split(",")[1]
    img_bytes = base64.b64decode(image_data)
    np_img = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Erro ao decodificar a imagem"}

    # Pré-processamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (req.params.blur_size, req.params.blur_size), 0)

    _, brain_bin = cv2.threshold(gray, req.params.threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    brain_bin = cv2.morphologyEx(brain_bin, cv2.MORPH_CLOSE, kernel)

    # Detecção de contornos
    contours, _ = cv2.findContours(brain_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detecção da régua
    roi_ruler = gray[-150:, :]
    _, ruler_bin = cv2.threshold(roi_ruler, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(ruler_bin, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    scale = None
    ruler_coords = {}
    if lines is not None:
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 5:
                horizontal_lines.append(line)
        if len(horizontal_lines) >= 2:
            x_coords = [x for line in horizontal_lines for x in [line[0][0], line[0][2]]]
            x_min, x_max = min(x_coords), max(x_coords)
            ruler_coords = {
                "x1": int(x_min),
                "y1": int(img.shape[0] - 150),
                "x2": int(x_max),
                "y2": int(img.shape[0])
            }
            total_dist_px = x_max - x_min
            scale = req.ruler_length_um / total_dist_px  # µm/pixel
            cv2.rectangle(img, (x_min, img.shape[0]-150), (x_max, img.shape[0]), (255,0,255), 2)

    organoids = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        area_px = cv2.contourArea(cnt)
        area_um2 = area_px * (scale ** 2) if scale else 0
        if area_px < req.params.min_area:
            continue
        if scale and (area_um2 / 1_000_000 > req.params.max_area):  # convert to mm²
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        diameter_px = max(w, h)
        diameter_um = diameter_px * scale if scale else 0
        radius_um = diameter_um / 2
        volume_um3 = (4/3) * np.pi * (radius_um ** 3)
        cv2.drawContours(img, [cnt], -1, (0,255,0), 2)
        cv2.putText(img, f"{diameter_um:.1f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        organoids.append({
            "x_px": int(x),
            "y_px": int(y),
            "width_px": int(w),
            "height_px": int(h),
            "diameter_um": float(round(diameter_um, 2)),
            "area_um2": float(round(area_um2, 2)),
            "volume_um3": float(round(volume_um3, 2))
        })

    # Codificar imagem processada como base64
    _, buffer = cv2.imencode('.png', img)
    processed_b64 = base64.b64encode(buffer).decode('utf-8')
    processed_image_base64 = f"data:image/png;base64,{processed_b64}"

    return {
        "scale_factor": float(round(scale, 4)) if scale else None,
        "ruler_coords_px": ruler_coords if scale else None,
        "organoids": organoids,
        "processed_image_base64": processed_image_base64
    }
