from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/analyze")
def analyze_organoid():
    output = {
        "scale_factor": 0.5,
        "ruler_coords_px": {
            "x1": 10, "y1": 550, "x2": 500, "y2": 550
        },
        "organoids": [
            {
                "x_px": 50, "y_px": 60,
                "width_px": 40, "height_px": 45,
                "diameter_um": 150.25,
                "area_um2": 17720.12,
                "volume_um3": 1772012.34
            }
        ],
        "processed_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
    }
    return JSONResponse(content=output)
