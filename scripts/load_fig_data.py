import os
import sys
import json
from typing import List, Dict

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from services.chroma_service import ChromaService
from config.settings import Settings

def process_fig_content(content: Dict) -> str:
    """Process fig content into a well-structured document."""
    doc = f"Topic: {content['subtitle']}\n\n"
    doc += f"Description: {content['description']}\n\n"
    
    for field, value in content.items():
        if field not in ['subtitle', 'description']:
            if field == 'dialogue_patterns':
                doc += "Common Dialogues:\n"
                for q, r in zip(value['questions'], value['responses']):
                    doc += f"Q: {q}\nR: {r}\n\n"
            elif field == 'situational_context':
                doc += "Situational Context:\n"
                for context_type, examples in value.items():
                    doc += f"{context_type.replace('_', ' ').title()}:\n"
                    for example in examples:
                        doc += f"- {example}\n"
                doc += "\n"
            elif field == 'emotional_expressions':
                doc += "Common Expressions:\n"
                for expr in value:
                    doc += f"- {expr}\n"
                doc += "\n"
            elif isinstance(value, list):
                doc += f"{field.capitalize()}:\n"
                for item in value:
                    doc += f"- {item}\n"
                doc += "\n"
            elif isinstance(value, dict):
                doc += f"{field.capitalize()}:\n"
                for subkey, subitems in value.items():
                    doc += f"{subkey.capitalize()}:\n"
                    for item in subitems:
                        doc += f"- {item}\n"
                doc += "\n"
    
    return doc

def enhance_metadata(topic: str, content: Dict) -> Dict:
    """Create enhanced metadata for better retrieval."""
    metadata = {
        "type": "fig_knowledge",
        "language": "tounsi",
        "topic": topic,
        "subtopic": content['subtitle'],
        "category": "fig_care" if "5edma" in topic else "fig_disease",
        "dialect_markers": ",".join(extract_dialect_markers(content)),
        "agricultural_terms": ",".join(extract_agricultural_terms(content)),
        "has_dialogue": str("dialogue_patterns" in content),
        "has_context": str("situational_context" in content),
        "has_expressions": str("emotional_expressions" in content)
    }
    return metadata

def extract_dialect_markers(content: Dict) -> List[str]:
    markers = set()
    text = json.dumps(content, ensure_ascii=False)
    common_markers = [
        "el", "fil", "bel", "wel", "mta3", "mte3",
        "3and", "3end", "famma", "tawa", "barcha",
        "ya3ni", "5ater", "bech", "mech", "elli"
    ]
    for marker in common_markers:
        if marker in text.lower():
            markers.add(marker)
    return list(markers)

def extract_agricultural_terms(content: Dict) -> List[str]:
    terms = set()
    text = json.dumps(content, ensure_ascii=False)
    agri_terms = [
        "karmous", "figue", "ghalla", "trab", "chajra",
        "mardh", "amrath", "war9a", "jdher",
        "ta9lim", "tesmi7", "roch", "dawé", "mbidet"
    ]
    for term in agri_terms:
        if term in text.lower():
            terms.add(term)
    return list(terms)

def process_fig_file(filename: str, chroma_service: ChromaService):
    path = f"data/Tunisian/{filename}.json"
    with open(path, "r", encoding="utf-8") as f:
        fig_data = json.load(f)

    print(f"Processing {filename}.json ...")
    for topic in fig_data["fig_topics"]:
        topic_name = topic["topic"]
        for content_item in topic["content"]:
            doc = process_fig_content(content_item)
            metadata = enhance_metadata(topic_name, content_item)
            try:
                embedding = chroma_service.embedding_model.embed_documents([doc])[0]
                chroma_service.collection.add(
                    documents=[doc],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"fig_{topic_name}_{chroma_service.document_id}"]
                )
                chroma_service.document_id += 1
                print(f"Added document for {content_item['subtitle']}")
            except Exception as e:
                print(f"Error processing {content_item['subtitle']}: {str(e)}")

def main():
    settings = Settings()
    chroma_service = ChromaService(settings)
    
    fig_files = ["fig_extended_knowledge", "fig_knowledge_base"]
    for file in fig_files:
        process_fig_file(file, chroma_service)
    
    print("✅ All fig (karmous) knowledge loaded into Chroma.")

if __name__ == "__main__":
    main()
