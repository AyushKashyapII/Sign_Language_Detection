from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import string
import mediapipe as mp
import cv2
import base64
from typing import List

app = FastAPI(title="Sign Language Classifier API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Create labels dictionary
labels_dict = {i: letter for i, letter in enumerate(list(string.ascii_uppercase) + ['Space'])}

class PredictionRequest(BaseModel):
    image_data: str  # Base64 encoded image
    confidence_threshold: float = 0.3

class PredictionResponse(BaseModel):
    predicted_character: str
    confidence: float
    success: bool
    message: str

def extract_landmarks_from_image(image_data: str) -> List[float]:
    """Extract hand landmarks from base64 encoded image"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(img_rgb)
        
        if not results.multi_hand_landmarks:
            raise ValueError("No hand landmarks detected")
        
        # Use only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        
        data_aux = []
        x_ = []
        y_ = []
        
        # Extract x and y coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            x_.append(x)
            y_.append(y)
        
        # Normalize coordinates
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x - min(x_))
            data_aux.append(y - min(y_))
        
        return data_aux
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Sign Language Classifier API is running!"}

@app.get("/health")
async def health_check():
    if model is None:
        return {"status": "unhealthy", "message": "Model not loaded"}
    return {"status": "healthy", "message": "Model loaded and ready"}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sign(request: PredictionRequest):
    """
    Predict sign language character from image
    """
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Extract landmarks from image
        landmarks = extract_landmarks_from_image(request.image_data)
        
        # Make prediction
        prediction = model.predict([np.asarray(landmarks)])
        predicted_class = int(prediction[0])
        
        # Get predicted character
        predicted_character = labels_dict.get(predicted_class, "Unknown")
        
        # Get prediction probabilities for confidence
        probabilities = model.predict_proba([np.asarray(landmarks)])
        confidence = float(np.max(probabilities))
        
        # Check confidence threshold
        if confidence < request.confidence_threshold:
            return PredictionResponse(
                predicted_character="Unknown",
                confidence=confidence,
                success=False,
                message=f"Low confidence ({confidence:.2f}) below threshold ({request.confidence_threshold})"
            )
        
        return PredictionResponse(
            predicted_character=predicted_character,
            confidence=confidence,
            success=True,
            message=f"Successfully predicted {predicted_character} with confidence {confidence:.2f}"
        )
        
    except ValueError as e:
        return PredictionResponse(
            predicted_character="",
            confidence=0.0,
            success=False,
            message=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/labels")
async def get_labels():
    """Get all available labels"""
    return {"labels": labels_dict}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
