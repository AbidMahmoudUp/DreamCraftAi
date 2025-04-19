from fastapi import FastAPI
from api import chat, weather, disease  
from config.settings import Settings

# Initialize the FastAPI app
app = FastAPI(
    title="Agri-Weather Solution",
    description="A scalable solution combining agriculture LLM expertise, weather data analysis, and plant disease detection",
    version="1.0.0"
)

# Include API routers from the api/ directory
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(weather.router, prefix="/api/v1", tags=["Weather"])
app.include_router(disease.router, prefix="/api/v1", tags=["Disease"])  

# Entry point for running the application
if __name__ == "__main__":
    settings = Settings()
    import uvicorn
    uvicorn.run(
        "main:app",  # Pass the app as a string
        host=settings.host,
        port=settings.port,
        reload=True  # Enable auto-reload for development
    )