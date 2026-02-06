import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
import time
import logging

# Setup Logging (M5)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cat vs Dog Classifier")

# Global variables
model = None

def load_model():
    global model
    try:
        model = tf.keras.models.load_model("model.h5")
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # For CI/CD smoke tests, we don't crash if model is missing, just log it
        print("Warning: Model file not found. Prediction will fail.")

@app.on_event("startup")
async def startup_event():
    load_model()

def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "MLOps Cat vs Dog API is running!"}

@app.get("/health")
def health_check():
    # Simple health check
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    start_time = time.time()
    
    # Read Image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_image)[0][0]
    label = "Dog" if prediction > 0.5 else "Cat"
    confidence = float(prediction) if label == "Dog" else 1.0 - float(prediction)
    
    # Logging (M5)
    latency = time.time() - start_time
    logger.info(f"Prediction: {label}, Confidence: {confidence:.2f}, Latency: {latency:.4f}s")
    
    return {
        "label": label,
        "confidence": confidence,
        "latency_seconds": latency
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
