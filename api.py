import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from werkzeug.utils import secure_filename
import numpy as np
import cv2

# Import your existing code
from inference import DeepfakeDetector
from config import Config

# --- CONFIGURATION ---
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS_IMG = {'png', 'jpg', 'jpeg'}
ALLOWED_EXTENSIONS_VID = {'mp4', 'avi', 'mov'}

# Create the FastAPI app
app = FastAPI(title="Deepfake Detection API")

# Add CORS middleware
# This allows your frontend (index.html) to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],
)

# --- LOAD YOUR MODEL ON STARTUP ---
config = Config()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(config.MODEL_SAVE_PATH, 'best_efficientnet.h5')

detector = None

@app.on_event("startup")
def load_model():
    """Load the ML model when the server starts."""
    global detector
    if not os.path.exists(MODEL_PATH):
        print("="*80)
        print(f"WARNING: Model file not found at {MODEL_PATH}")
        print("The API will run, but /detect will fail.")
        print("Please train a model first using: python main.py train")
        print("="*80)
    else:
        print(f"Loading model from {MODEL_PATH}...")
        detector = DeepfakeDetector(MODEL_PATH, img_size=config.IMG_SIZE)
        print("Model loaded successfully.")

# --- API ENDPOINT ---
@app.post("/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    The main detection endpoint. Receives a file and returns a prediction.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model is not loaded or failed to load.")

    # 1. Check if the file is valid
    file_type = get_file_type(file.filename)
    if not file_type:
        raise HTTPException(status_code=400, detail="File type not allowed")

    # 2. Save the file temporarily
    filename = secure_filename(file.filename)
    temp_path = os.path.join(UPLOAD_FOLDER, filename)
    
    result = {}
    try:
        # Write the uploaded file to the temp path
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # 4. Call your inference logic
        print(f"Analyzing file: {temp_path} (Type: {file_type})")
        if file_type == 'image':
            result = detector.predict_image(temp_path)
        else:
            result = detector.predict_video(temp_path, num_frames=config.FRAMES_PER_VIDEO)
        
        print(f"Prediction result: {result}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 5. Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 6. Return the JSON result to the frontend
    if not result.get('success'):
        raise HTTPException(status_code=400, detail=result.get('error', 'Prediction failed'))

    return result

def get_file_type(filename):
    """Helper to check file extension."""
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS_IMG:
        return 'image'
    if ext in ALLOWED_EXTENSIONS_VID:
        return 'video'
    return None

# --- FRONTEND SERVING ---
# This serves your index.html, style.css, and app.js
# It MUST be the last route added.
app.mount("/", StaticFiles(directory=".", html=True), name="static")


# --- RUN THE SERVER ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    print("Access the API docs at http://127.0.0.1:8000/docs")
    print("Access the frontend at http://127.0.0.1:8000")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
