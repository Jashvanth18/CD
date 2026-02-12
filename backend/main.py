from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

# Import the inference logic
# This assumes inference.py is in the same directory
from inference import inspect_image

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://aivision-machine.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image from upload
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "ERROR", "message": "Failed to decode image"}
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run inference
        vis, stage, label, dist = inspect_image(img_rgb)
        
        # Convert result image (vis) back to BGR for encoding
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        
        # Encode to Base64
        _, buffer = cv2.imencode('.jpg', vis_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "status": label,
            "class_name": stage,
            "similarity": float(dist) if dist is not None else 0.0,
            "step1_conf": 1.0, 
            "image": img_str
        }
    except Exception as e:
        print(f"CRITICAL ERROR in /predict: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "ERROR", "message": str(e), "similarity": 0}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
