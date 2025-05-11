from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    ffmpeg_path: str = r"C:\ffmpeg\bin"  
    chroma_persist_dir: str = "chroma_storage"
    chroma_collection_name: str = "conversation_memory"
    llm_model: str = "llama3.2"
    weather_api_base_url: str = "http://localhost:8000"  # Internal base URL
    mongodb_db_name: str = "chat_db"
    mongodb_uri: str = "mongodb://localhost:27017/"
    nestjs_api_base_url: str = "http://192.168.43.232:3000"
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"