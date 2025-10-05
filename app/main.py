from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
import numpy as np
import base64
import psycopg2
from contextlib import contextmanager
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

@contextmanager
def get_db_connection():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

class SelfieRequest(BaseModel):
    email: str
    imageData: str

@app.post("/validate-selfie")
async def validate_selfie(request: SelfieRequest):
    """
    Validates selfie, stores image bytes, and adds all group IDs to un_matched_groups
    """
    try:
        # Decode base64 image
        image_data = request.imageData
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return JSONResponse(
                status_code=400,
                content={"valid": False, "error": "Invalid image format"}
            )
        
        # Basic validations using Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        if len(faces) == 0:
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": "No face detected. Please ensure your face is clearly visible."
                }
            )
        elif len(faces) > 1:
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": f"Multiple faces detected ({len(faces)}). Only one person should be in the frame."
                }
            )
        
        # Quality checks
        x, y, w, h = faces[0]
        face_area = w * h
        img_area = img.shape[0] * img.shape[1]
        face_ratio = face_area / img_area
        
        if face_ratio < 0.15:
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": "Face is too small. Please move closer to the camera."
                }
            )
        
        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logger.info(f"Blur metric (Laplacian variance): {laplacian_var}")
        
        if laplacian_var < 10:
            return JSONResponse(
                status_code=400,
                content={
                    "valid": False,
                    "error": "Image is too blurry. Please ensure good lighting and focus."
                }
            )
        
        # Store in database and update groups
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if user exists and get their current groups
                    cur.execute("SELECT id, groups FROM users WHERE email = %s", (request.email,))
                    user = cur.fetchone()
                    
                    if not user:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "valid": False,
                                "error": "User not found with this email"
                            }
                        )
                    
                    user_id = user[0]
                    user_groups = user[1] if user[1] else []
                    
                    logger.info(f"User {request.email} has {len(user_groups)} existing groups")
                    
                    # Update user with image bytes and set un_matched_groups to their existing groups
                    cur.execute(
                        """
                        UPDATE users 
                        SET face_image_bytes = %s,
                            un_matched_groups = %s,
                            face_updated_at = NOW()
                        WHERE email = %s
                        """,
                        (img_bytes, user_groups, request.email)
                    )
                    conn.commit()
                    
                    logger.info(f"Face image stored for user: {request.email}")
                    logger.info(f"Set un_matched_groups to user's existing {len(user_groups)} groups")
                    
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            return JSONResponse(
                status_code=500,
                content={
                    "valid": False,
                    "error": "Failed to store face image in database"
                }
            )
        
        return JSONResponse(
            status_code=200,
            content={
                "valid": True,
                "message": "Selfie validated and stored successfully",
                "face_detected": True,
                "groups_added": len(user_groups),
                "face_coordinates": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h)
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Error in validate_selfie: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"valid": False, "error": f"Server error: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Selfie Validation API", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)