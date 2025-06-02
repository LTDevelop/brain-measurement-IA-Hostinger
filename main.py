from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import math
from skimage import measure
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage

app = FastAPI()

class ImageRequest(BaseModel):
    image_base64: str
    ruler_length_um: float  # comprimento da régua conhecida (ex: 1250 µm)

@app.post("/process-image")
def process_image(data: ImageRequest):
    try:
        # Decodificar imagem base64
        header, encoded = data.image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Erro ao decodificar a imagem.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        distance = ndimage.distance_transform_edt(binary)
        local_max_coords = peak_local_max(distance, min_distance=20, footprint=np.ones((5, 5)), labels=binary)
        local_max = np.zeros_like(binary, dtype=bool)
        local_max[tuple(local_max_coords.T)] = True
        markers = measure.label(local_max)
        labels = watershed(-distance, markers, mask=binary)

        scale_factor = data.ruler_length_um / 490  # conversão de pixel para µm
        organoids = []

        for region in measure.regionprops(labels):
            equivalent_diameter = region.equivalent_diameter
            equivalent_diameter_um = equivalent_diameter * scale_factor
            if 100 <= equivalent_diameter_um <= 3000:
                radius_um = equivalent_diameter_um / 2
                volume_um3 = (4 / 3) * math.pi * (radius_um ** 3)
                area_um2 = region.area * (scale_factor ** 2)

                organoids.append({
                    "x_px": int(region.centroid[1]),
                    "y_px": int(region.centroid[0]),
                    "width_px": int(region.bbox[3] - region.bbox[1]),
                    "height_px": int(region.bbox[2] - region.bbox[0]),
                    "diameter_um": round(equivalent_diameter_um, 2),
                    "area_um2": round(area_um2, 2),
                    "volume_um3": round(volume_um3, 2)
                })

        # Desenhar contornos na imagem
        for region in measure.regionprops(labels):
            equivalent_diameter = region.equivalent_diameter * scale_factor
            if 100 <= equivalent_diameter <= 3000:
                y, x = region.centroid
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Codificar imagem de volta para base64
        _, buffer = cv2.imencode('.png', image)
        processed_base64 = base64.b64encode(buffer).decode("utf-8")
        processed_image_base64 = f"data:image/png;base64,{processed_base64}"

        return JSONResponse(content={
            "scale_factor": scale_factor,
            "ruler_coords_px": { "x1": 10, "y1": 550, "x2": 500, "y2": 550 },
            "organoids": organoids,
            "processed_image_base64": processed_image_base64
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "API online e pronta para medir organoides"}
