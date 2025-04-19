import os
import json
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import Settings
from utils.logging import setup_logging

logger = setup_logging()

class ChromaService:
    def __init__(self, settings: Settings):
        self.persist_dir = settings.chroma_persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name=settings.chroma_collection_name)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.data_dir = "data"
        self.document_id = 0

    def load_plant_dataset(self, file_path: str, language: str):
        """Load plant disease dataset JSON files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loading plant dataset from {file_path} in {language}")
            
            for item in data:
                if not isinstance(item, dict) or "disease_info" not in item:
                    continue
                
                disease_info = item["disease_info"]
                
                # Create structured document
                doc = f"Disease: {disease_info.get('name', 'Unknown')}\n"
                if 'scientific_name' in disease_info:
                    doc += f"Scientific Name: {disease_info['scientific_name']}\n"
                if 'category' in disease_info:
                    doc += f"Category: {disease_info['category']}\n"
                if 'affected_plants' in disease_info:
                    plants = ', '.join(disease_info['affected_plants']) if isinstance(disease_info['affected_plants'], list) else disease_info['affected_plants']
                    doc += f"Affected Plants: {plants}\n"
                if 'symptoms' in disease_info:
                    symptoms = ', '.join(disease_info['symptoms']) if isinstance(disease_info['symptoms'], list) else disease_info['symptoms']
                    doc += f"Symptoms: {symptoms}\n"
                if 'treatment_steps' in disease_info:
                    treatment = ', '.join(disease_info['treatment_steps']) if isinstance(disease_info['treatment_steps'], list) else disease_info['treatment_steps']
                    doc += f"Treatment: {treatment}\n"
                if 'prevention' in disease_info:
                    prevention = ', '.join(disease_info['prevention']) if isinstance(disease_info['prevention'], list) else disease_info['prevention']
                    doc += f"Prevention: {prevention}\n"
                
                # Add example conversations - up to 3 for better context
                if 'conversation' in item:
                    doc += f"\nExample Conversation:\n"
                    doc += f"Human: {item['conversation']['human']}\n"
                    doc += f"Assistant: {item['conversation']['assistant']}\n"
                
                # Generate embedding
                embedding = self.embedding_model.embed_documents([doc])[0]
                
                # Store in Chroma
                metadata = {
                    "type": "plant_disease",
                    "language": language,
                    "source_file": os.path.basename(file_path)
                }
                if 'name' in disease_info:
                    metadata["disease_name"] = disease_info["name"]
                
                self.collection.add(
                    documents=[doc],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"disease_{self.document_id}"]
                )
                self.document_id += 1
                
            logger.info(f"Successfully loaded {len(data)} plant disease records from {file_path}")
        except Exception as e:
            logger.error(f"Error loading plant dataset from {file_path}: {str(e)}")

    def load_conversation_dataset(self, file_path: str, language: str):
        """Load agricultural machinery conversation JSON files."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            logger.info(f"Loading conversation dataset from {file_path} in {language}")
            
            for item in data:
                if not isinstance(item, dict) or "conversation" not in item or "info" not in item:
                    continue
                
                info = item["info"]
                conversation = item["conversation"]
                
                # Create structured document with enhanced context
                doc = f"Topic: {info.get('topic', 'Unknown')}\n"
                if 'category' in info:
                    doc += f"Category: {info['category']}\n"
                if 'details' in info:
                    details = '\n'.join([f"- {detail}" for detail in info['details']]) if isinstance(info['details'], list) else info['details']
                    doc += f"Details:\n{details}\n"
                
                # Add conversation patterns if available
                if 'patterns' in conversation:
                    doc += "\nCommon Patterns:\n"
                    for pattern_type, patterns in conversation['patterns'].items():
                        doc += f"{pattern_type.capitalize()}:\n"
                        for pattern in patterns:
                            doc += f"- {pattern}\n"
                
                doc += f"\nConversation:\n"
                doc += f"Human: {conversation['human']}\n"
                doc += f"Assistant: {conversation['assistant']}\n"
                
                # Generate embedding
                embedding = self.embedding_model.embed_documents([doc])[0]
                
                # Store in Chroma with enhanced metadata
                metadata = {
                    "type": "agricultural_conversation",
                    "language": language,
                    "topic": info.get("topic", "Unknown"),
                    "category": info.get("category", "Unknown"),
                    "source_file": os.path.basename(file_path)
                }
                
                # Add additional metadata for Tunisian content
                if language.lower() == "tunisian":
                    metadata.update({
                        "dialect_markers": self._extract_dialect_markers(doc),
                        "agricultural_terms": self._extract_agricultural_terms(doc)
                    })
                
                self.collection.add(
                    documents=[doc],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[f"conv_{self.document_id}"]
                )
                self.document_id += 1
                
            logger.info(f"Successfully loaded {len(data)} conversations from {file_path}")
        except Exception as e:
            logger.error(f"Error loading conversation dataset from {file_path}: {str(e)}")

    def _extract_dialect_markers(self, text: str) -> list:
        """Extract Tunisian dialect markers from text."""
        markers = []
        common_markers = [
            "el", "fil", "bel", "wel", "mta3", "mte3",
            "3and", "3end", "famma", "tawa", "barsha",
            "ya3ni", "5ater", "bech", "mech", "elli"
        ]
        for marker in common_markers:
            if marker in text.lower():
                markers.append(marker)
        return list(set(markers))

    def _extract_agricultural_terms(self, text: str) -> list:
        """Extract agricultural terms from text."""
        terms = []
        agri_terms = [
            "zitoun", "zitouna", "ghalla", "bidha", "trab",
            "mardh", "amrath", "3afej", "nabta", "chajra",
            "war9a", "jdher", "sa9", "thmar", "rosh",
            "dawé", "mbidet", "ri", "ta9tir", "s9aya"
        ]
        for term in agri_terms:
            if term in text.lower():
                terms.append(term)
        return list(set(terms))

    def load_olive_knowledge(self, file_path: str):
        """Load olive knowledge base into Chroma DB."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for topic in data['olive_topics']:
                topic_name = topic['title']
                
                for content_item in topic['content']:
                    # Create comprehensive content string
                    content_text = f"{topic_name} - {content_item['subtitle']}\n\n"
                    content_text += f"Description: {content_item['description']}\n\n"
                    
                    # Add all available fields
                    for field, value in content_item.items():
                        if field not in ['subtitle', 'description']:
                            if isinstance(value, list):
                                content_text += f"{field.capitalize()}:\n"
                                content_text += "\n".join(f"- {item}" for item in value)
                                content_text += "\n\n"
                            else:
                                content_text += f"{field.capitalize()}: {value}\n\n"
                    
                    # Generate embedding
                    embedding = self.embedding_model.embed_documents([content_text])[0]
                    
                    # Add to Chroma
                    self.collection.add(
                        documents=[content_text],
                        embeddings=[embedding],
                        metadatas=[{
                            "type": "olive_knowledge",
                            "topic": topic['topic'],
                            "subtopic": content_item['subtitle'],
                            "language": "tounsi",
                            "source_file": file_path
                        }],
                        ids=[f"olive_{topic['topic']}_{self.document_id}"]
                    )
                    self.document_id += 1
                    
            logger.info(f"Successfully loaded olive knowledge from {file_path}")
        except Exception as e:
            logger.error(f"Error loading olive knowledge from {file_path}: {str(e)}")

    def load_all_datasets(self):
        """Load all datasets from organized directory structure."""
        language_dirs = {
            "English": "English",
            "French": "French",
            "Arabic": "Arabic",
            "Tunisian": "Tunisian"
        }
        
        for language, dir_name in language_dirs.items():
            dir_path = os.path.join(self.data_dir, dir_name)
            
            if not os.path.exists(dir_path):
                logger.warning(f"Directory not found: {dir_path}")
                continue
                
            logger.info(f"Processing {language} data from {dir_path}")
            
            for file in os.listdir(dir_path):
                if not file.endswith('.json'):
                    continue
                    
                file_path = os.path.join(dir_path, file)
                
                if "olive_knowledge_base" in file:
                    self.load_olive_knowledge(file_path)
                elif "Plant_Dataset" in file or "disease" in file.lower():
                    self.load_plant_dataset(file_path, language)
                elif "Conversations" in file:
                    self.load_conversation_dataset(file_path, language)
                else:
                    logger.warning(f"Unknown file type: {file_path}")

    def retrieve_context(self, query: str, max_results: int = 5, where: dict = None) -> str:
        """Retrieve relevant context based on the query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Perform the search with where clause if provided
            search_params = {
                "query_embeddings": [query_embedding],
                "n_results": max_results
            }
            
            # Handle where clause properly
            if where:
                # Ensure where clause is correctly formatted for chromadb
                # Simplify complex where clauses that might cause dimensionality issues
                if "$and" in where:
                    # Extract simple conditions from the $and clause
                    conditions = where["$and"]
                    simplified_where = {}
                    
                    for condition in conditions:
                        if isinstance(condition, dict) and len(condition) == 1:
                            key = list(condition.keys())[0]
                            simplified_where[key] = condition[key]
                    
                    # Use simplified where if we got any conditions
                    if simplified_where:
                        search_params["where"] = simplified_where
                    else:
                        # Fall back to just language if available
                        language_condition = next((c for c in conditions if "language" in c), None)
                        if language_condition:
                            search_params["where"] = {"language": language_condition["language"]}
                else:
                    search_params["where"] = where
                    
            # Add error handling for empty results
            results = self.collection.query(**search_params)
            
            if not results["documents"] or not results["documents"][0]:
                return ""
                
            # Format results into a single context string
            context = ""
            for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
                source = metadata.get("source_file", "Unknown")
                context += f"---\n{doc}\nsource_file:{source}\n---\n"
                
            return context.strip()
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            # Try a more basic query if the complex one fails
            try:
                # Simplified query without complex where clauses
                basic_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=max_results
                )
                
                if not basic_results["documents"] or not basic_results["documents"][0]:
                    return ""
                    
                # Format results into a single context string
                context = ""
                for doc, metadata in zip(basic_results["documents"][0], basic_results["metadatas"][0]):
                    source = metadata.get("source_file", "Unknown")
                    context += f"---\n{doc}\nsource_file:{source}\n---\n"
                    
                return context.strip()
                
            except Exception as nested_e:
                logger.error(f"Even basic retrieval failed: {str(nested_e)}")
                return ""
    def store_conversation(self, question: str, response: str):
        """Store a conversation for future retrieval."""
        try:
            from services.language_detection import detect_language
            language = detect_language(question)
            
            combined_text = f"Question: {question}\nResponse: {response}"
            embedding = self.embedding_model.embed_documents([combined_text])[0]
            
            self.collection.add(
                documents=[combined_text],
                embeddings=[embedding],
                metadatas=[{
                    "type": "conversation_history",
                    "language": language,
                    "source_file": "conversation"
                }],
                ids=[f"history_{self.document_id}"]
            )
            self.document_id += 1
            logger.info(f"Stored conversation in {language}")
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")