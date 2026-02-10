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
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Backend is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image from upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB (inference expects RGB if passed as array, 
    # but let's check inference.py usage. 
    # inference.py: inspect_image(obs_img)
    # Inside inspect_image: obs_t = tf(Image.fromarray(obs_img))...
    # Usually cv2 loads BGR. PIL Image.fromarray expects RGB.
    # So we should convert BGR to RGB here.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run inference
    vis, stage, label, dist = inspect_image(img_rgb)
    
    # Convert result image (vis) back to BGR for encoding (cv2 usually encodes BGR)
    # Wait, 'vis' comes from inference.py lines:
    # vis = cv2.resize(obs_img, (224,224)) ... cv2.rectangle ...
    # obs_img passed was RGB. So vis is RGB.
    # cv2.imencode expects BGR? Yes, usually.
    # So convert RGB -> BGR before encoding
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    
    # Encode to Base64
    _, buffer = cv2.imencode('.jpg', vis_bgr)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "status": label,
        "class_name": stage,
        "similarity": float(dist),
        "step1_conf": 1.0, # Dummy value for now since inference returns a joint score
        "image": img_str
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
