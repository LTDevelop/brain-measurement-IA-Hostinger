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
    ruler_length_um: float

@app.post("/process-image")
def process_image(data: ImageRequest):
    try:
        # Extrair base64 limpo
        header, encoded = data.image_base64.split(",", 1)
        image_data = base64.b64decode(encoded)
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Erro ao decodificar a imagem")

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

        # Simulando detecção de régua fixa de 490 px
        ruler_length_px = 490
        scale_factor = data.ruler_length_um / ruler_length_px

        organoids = []
        for region in measure.regionprops(labels):
            equivalent_diameter_um = region.equivalent_diameter * scale_factor
            if 100 <= equivalent_diameter_um <= 3000:
                area_um2 = region.area * (scale_factor ** 2)
                radius_um = equivalent_diameter_um / 2
                volume_um3 = (4/3) * math.pi * (radius_um ** 3)

                minr, minc, maxr, maxc = region.bbox
                organoids.append({
                    "x_px": minc,
                    "y_px": minr,
                    "width_px": maxc - minc,
                    "height_px": maxr - minr,
                    "diameter_um": round(equivalent_diameter_um, 2),
                    "area_um2": round(area_um2, 2),
                    "volume_um3": round(volume_um3, 2)
                })

                # Marcar visualmente
                cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)

        # Codificar imagem final
        _, buffer = cv2.imencode('.png', image)
        processed_base64 = base64.b64encode(buffer).decode("utf-8")
        processed_image_base64 = f"data:image/png;base64,{processed_base64}"

        response = {
            "scale_factor": scale_factor,
            "ruler_coords_px": {
                "x1": 50,
                "y1": 550,
                "x2": 540,
                "y2": 550
            },
            "organoids": organoids,
            "processed_image_base64": processed_image_base64
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "API online e compatível com Horizons"}
