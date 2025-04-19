import os
import sys
import json
from typing import List, Dict

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from services.chroma_service import ChromaService
from config.settings import Settings

def process_olive_diseases(data: List[Dict]) -> List[Dict]:
    """Process and enhance olive disease data with additional context."""
    enhanced_data = []
    for item in data:
        if "disease_info" in item:
            # Add more contextual information
            item["disease_info"]["context"] = {
                "common_cases": [
                    "El mardh hetha yji fel wa9t mta3 {season}",
                    "Akthar 7aja tshofha fel {symptom}",
                    "Lazem ta3mel el wa9aya {timing}"
                ],
                "farmer_tips": [
                    "5alli ba3dhom el wra9 el maridha bech ta3ref kifeh tet3amel m3ahom",
                    "Rosh el dawé bekri fel sbe7 walla fel 3shiya",
                    "A3mel belek mel rtoba wel 7rara"
                ]
            }
            enhanced_data.append(item)
    return enhanced_data

def process_conversations(data: List[Dict]) -> List[Dict]:
    """Process and enhance agricultural conversations with more natural patterns."""
    enhanced_data = []
    for item in data:
        if "conversation" in item and "info" in item:
            # Add more natural conversation patterns
            item["conversation"]["patterns"] = {
                "greetings": [
                    "Ya3tik el sa77a ya fella7",
                    "Sbah el 5ir 3al ghalla",
                    "Marhbe bik ya si el fella7"
                ],
                "understanding": [
                    "Ay na3ref 3al mochkol hetha...",
                    "Fehemt 3lik ya si el fella7...",
                    "El 7keya wadh7a..."
                ],
                "advice": [
                    "A3mel belek...",
                    "5alli n9ollek 7aja mhemma...",
                    "El 7all mta3 el mochkol hetha..."
                ]
            }
            enhanced_data.append(item)
    return enhanced_data

def main():
    settings = Settings()
    chroma_service = ChromaService(settings)
    
    # Process olive disease data
    olive_path = "data/Tunisian/olive_diseases_conversations.json"
    with open(olive_path, "r", encoding="utf-8") as f:
        olive_data = json.load(f)
    enhanced_olive_data = process_olive_diseases(olive_data)
    with open(olive_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_olive_data, f, ensure_ascii=False, indent=2)
    
    # Process agricultural conversations
    conv_path = "data/Tunisian/agriculture_Arabic_Tunisan_Conversations.json"
    with open(conv_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    enhanced_conv_data = process_conversations(conv_data)
    with open(conv_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_conv_data, f, ensure_ascii=False, indent=2)
    
    # Load all enhanced data into Chroma
    chroma_service.load_all_datasets()
    print("Successfully loaded and enhanced Tunisian agricultural data")

if __name__ == "__main__":
    main() 