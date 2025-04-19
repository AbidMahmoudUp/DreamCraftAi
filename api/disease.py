from fastapi import APIRouter, HTTPException, File, UploadFile
from config.settings import Settings
from services.disease_service import DiseaseService
from utils.logging import setup_logging  # Fixed typo from 'setup_logger' to 'setup_logging'

router = APIRouter()
settings = Settings()
disease_service = DiseaseService(settings)
logger = setup_logging()

@router.post("/disease/predict", response_model=dict)
async def predict_disease(file: UploadFile = File(...)):
    try:
        content = await file.read()
        predicted_class = disease_service.process_uploaded_image(content, file.filename)
        return {"predicted_class": predicted_class}
    except Exception as e:
        logger.error(f"Error in disease prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))