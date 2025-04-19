import os
import sys
import shutil
import chromadb

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from services.chroma_service import ChromaService
from services.llm_service import LLMService
from services.weather_service import WeatherService
from config.settings import Settings
from models.chat import ChatRequest
import asyncio

# Rest of your script remains unchanged
settings = Settings()

# Step 1: Clear existing Chroma data
persist_dir = settings.chroma_persist_dir
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)
    print(f"Deleted Chroma persistence directory: {persist_dir}")

# Step 2: Initialize ChromaService and load datasets
chroma_service = ChromaService(settings)
chroma_service.load_all_datasets()
print("Finished loading datasets.")

# Step 3: Test data retrieval
test_query = "What are the symptoms of blight in plants?"
context = chroma_service.retrieve_context(test_query, max_results=5)
print(f"Retrieved context for '{test_query}':\n{context}")

# Step 4: Test LLMService
weather_service = WeatherService()
llm_service = LLMService(model_name="llama3.1", chroma_service=chroma_service)
request = ChatRequest(
    question=test_query,
    country=None,
    detected_disease=None
)
response = asyncio.run(llm_service.process_query(request, weather_service))
print(f"Response: {response.response}")
print(f"Sources: {response.sources}")
print(f"Language: {response.language}")