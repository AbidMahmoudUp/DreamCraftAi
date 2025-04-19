from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from services.chroma_service import ChromaService
from services.weather_service import WeatherService
from models.chat import ChatRequest, ChatResponse
from utils.logging import setup_logging
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime

logger = setup_logging()

def detect_language(text: str) -> str:
    """Detect query language (english, french, arabic, tounsi)."""
    from services.language_detection import detect_language
    return detect_language(text)

class LLMService:
    def __init__(self, model_name: str, chroma_service: ChromaService):
        self.model_name = model_name
        self.chroma_service = chroma_service
        
        # Initialize HuggingFaceEmbeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize LangChain Chroma vector store
        self.vector_store = Chroma(
            collection_name=self.chroma_service.collection.name,
            embedding_function=self.embedding_model,
            persist_directory=self.chroma_service.persist_dir
        )
        
        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Initialize LLM with improved parameters for fluency
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.8,  # Increased from 0.7 for more creative responses
            top_p=0.92,       # Slightly increased for more varied word choice
            repetition_penalty=1.1,  # Reduced from 1.2 to allow more natural flow
            max_tokens=400     # Increased from 300 for more complete responses
        )
        
        # Initialize chat history
        self.memory = ChatMessageHistory()
        self.last_topic = None  # Store last detected topic
        
        # Define topic keywords based on dataset
        self.topic_keywords = {
            "mildiou": ["mildiou", "war9a safra", "fongus", "3inab", "tomitim"],
            "anwa3 el bidhour": ["bidhour", "anwa3 el bidhour", "zira3a", "9amh", "tomitim", "khodra"],
            "anzimet el ri": ["anzimet el ri", "ta9tir", "s9aya", "ma", "system ri"],
            "fle7a_general": [
                # Tunisian terms
                "fle7a", "fla7a", "nsi7a fil fle7a", "zra3a", 
                "ardh", "ghalla", "mazrou3at",
                # Arabic terms
                "زراعة", "فلاحة", "نصيحة في الفلاحة",
                # English terms
                "farming", "agriculture", "cultivation",
                # French terms
                "agriculture", "cultivation", "conseil agricole"
            ],
            "amrath_zitoun": [
                # Tunisian terms
                "amrath zitoun", "amrath il zitoun", "mardh zitoun", 
                "3afej zitoun", "cycloconium", "peacock", "vertichillium", 
                "amrath zitouna", "mardh zitouna", "3ayn el tawous",
                "dboul", "ta3fen", "wra9 zitoun",
                # Arabic terms
                "امراض الزيتون", "مرض الزيتون", "عين الطاووس",
                "الذبول", "تعفن", "سيكلوكونيوم", "فيرتيسيليوم",
                # French terms
                "maladies olivier", "oeil de paon", "verticilliose",
                # English terms
                "olive diseases", "peacock eye", "verticillium"
            ]
        }
        
        # Define language-specific fallbacks
        self.topic_fallbacks = {
            "fle7a_general": {
                "tounsi": """5alli n9ollek 3al el fle7a w nsa7ni el mohemmin:

• El Ardh wel 5edma:
  - Lazem t7adher el ardh mli7 9bal el zra3a
  - El trab lazem ykoun mfakek w msam7ed
  - E5dem el ardh fel wa9t el monaseb

• El Zra3a:
  - Kol mazrou3 3ando wa9to el monaseb
  - Lazem t5ayyer el bidher el behi
  - Ma tensa ch el msafa bin el zra3a

• El S9aya:
  - Es9i fel wa9t el monaseb
  - Ma tfayedh ch bel ma
  - Raqeb el nabta kol nhar

• El 7maya:
  - Dima raqeb el nabta mel 7achrat
  - Ken tal9a mochkla, ta7arrek bekri
  - Est3mel el mbidet ken lazem bark

A3mel belek: el fle7a t7eb e5las w sabr. El ghalla el behia ma tjich ken ba3d el t3ab.""",
                "arabic": """فيما يلي أهم النصائح الزراعية الأساسية:

• تحضير التربة:
  - تهيئة الأرض جيداً قبل الزراعة
  - ضمان تفكيك التربة وتسميدها
  - اختيار الوقت المناسب للحرث

• الزراعة:
  - مراعاة المواسم المناسبة لكل محصول
  - اختيار البذور الجيدة
  - الالتزام بالمسافات المناسبة بين النباتات

• الري:
  - الري في الأوقات المناسبة
  - تجنب الإفراط في الري
  - مراقبة النباتات يومياً

• الوقاية:
  - المراقبة المستمرة للآفات
  - التدخل المبكر عند ظهور أي مشكلة
  - استخدام المبيدات عند الضرورة فقط""",
                "english": """Here are the key agricultural guidelines:

• Soil Preparation:
  - Prepare the soil well before planting
  - Ensure proper soil loosening and fertilization
  - Choose the right time for cultivation

• Planting:
  - Respect the appropriate seasons for each crop
  - Select quality seeds
  - Maintain proper spacing between plants

• Irrigation:
  - Water at appropriate times
  - Avoid over-irrigation
  - Monitor plants daily

• Protection:
  - Regular pest monitoring
  - Early intervention when problems arise
  - Use pesticides only when necessary""",
                "french": """Voici les conseils agricoles essentiels:

• Préparation du Sol:
  - Bien préparer le sol avant la plantation
  - Assurer l'ameublissement et la fertilisation
  - Choisir le bon moment pour le travail du sol

• Plantation:
  - Respecter les saisons pour chaque culture
  - Sélectionner des semences de qualité
  - Maintenir un espacement approprié

• Irrigation:
  - Arroser aux moments appropriés
  - Éviter la sur-irrigation
  - Surveiller les plantes quotidiennement

• Protection:
  - Surveillance régulière des parasites
  - Intervention précoce en cas de problème
  - Utiliser des pesticides uniquement si nécessaire"""
            },
            "amrath_zitoun": {
                "tounsi": "El zitoun 3ando barsha amrath ma3roufa. El cycloconium (3ayn el tawous) y5alli ta9a3 kahla 3al war9a, vertichillium y5alli el war9 yeb9a asfar w ytieh, w fama el 3afej elli yji m3a el rtoba. Lazem el wa9aya w el 3ilej bekri bel mbidet el monasba.",
                "arabic": """هناك العديد من الأمراض المعروفة التي تصيب شجرة الزيتون. من أهمها:
- عين الطاووس (سيكلوكونيوم): يسبب بقع سوداء على الأوراق
- الذبول الفيرتيسيليومي: يسبب اصفرار وسقوط الأوراق
- التعفن: يظهر مع الرطوبة العالية
يجب المعالجة الوقائية واستخدام المبيدات المناسبة في الوقت المناسب.""",
                "french": """L'olivier est sujet à plusieurs maladies importantes:
- L'œil de paon (Cycloconium): cause des taches noires sur les feuilles
- La verticilliose: provoque le jaunissement et la chute des feuilles
- La pourriture: apparaît avec l'humidité
Un traitement préventif et l'utilisation de pesticides appropriés sont nécessaires.""",
                "english": """Olive trees are susceptible to several well-known diseases:
- Peacock eye (Cycloconium): causes black spots on leaves
- Verticillium wilt: causes yellowing and leaf drop
- Rot: occurs with high humidity
Preventive treatment and appropriate pesticide use are essential."""
            }
        }
        
        # Define comprehensive prompt template with enhanced scenarios
        self.prompt_template = """
You are an experienced agricultural expert helping farmers. You MUST respond ONLY in the language of the user's question ({language}).

IMPORTANT: Select and follow ONLY the guidelines for the detected language ({language}):

[FOR TUNISIAN DIALECT - ONLY USE IF language="tounsi"]
• Response Structure:
  - Start with a warm greeting: "Marhbe bik ya fella7!" or "3aslema!"
  - Use authentic Tunisian dialect (derja) as spoken by farmers
  - End with encouragement: "Rabbi m3ak!" or "Tawfiq inchallah!"

• Common Patterns:
  - Greetings: "3aslema", "Marhbe", "Sbah el 5ir", "Bnet3achew"
  - Starters: "Chouf", "Isma3", "A3mel belek", "5alli n9ollek"
  - Connectors: "Ya3ni", "5ater", "3la 5ater", "Amma", "W zeda"
  - Time: "Tawa", "Lyoum", "Ghodwa", "Ba3d", "9bal"
  - Conditions: "Ken", "Wa9telli", "Ma7abech", "Lazem"

• Agricultural Terms:
  - Plants: "Zitoun", "Ghalla", "Nabta", "Chajra"
  - Actions: "A7reth", "Azra3", "Es9i", "9allem"
  - Problems: "Mardh", "3afej", "7achra", "Dboul"
  - Solutions: "Dawé", "Mbidet", "Roch", "3ilej"

• Scenario-Specific Responses:
  1. For Disease Questions:
     - "El mardh mta3... y3ayyet 3lih..."
     - "El 3alamet mta3ou..."
     - "Bech t3aljou lazem..."
  
  2. For Cultivation Advice:
     - "El wa9t el mriguel bech..."
     - "Lazem el ardh tkoun..."
     - "Ma tensech tzid..."
  
  3. For Problem Solving:
     - "Ken tal9a mochkla fi..."
     - "Awel 7aja lazem..."
     - "Rodbelk mel..."

[FOR STANDARD ARABIC - ONLY USE IF language="arabic"]
• Response Structure:
  - Start formally: "السلام عليكم" or "مرحباً بك"
  - Use Modern Standard Arabic (فصحى)
  - End professionally: "نتمنى لك التوفيق" or "مع خالص التحيات"

• Professional Patterns:
  - Introductions: "بداية،", "في البداية،", "دعني أوضح"
  - Transitions: "من ناحية أخرى", "علاوة على ذلك", "بالإضافة إلى"
  - Emphasis: "من المهم", "يجب التأكيد", "من الضروري"
  - Conclusions: "في النهاية", "ختاماً", "وأخيراً"

• Agricultural Terminology:
  - Technical: "المكافحة المتكاملة", "التسميد العضوي", "الري بالتنقيط"
  - Scientific: "الأمراض الفطرية", "المبيدات الحيوية", "التربة القلوية"
  - Practical: "تقليم الأشجار", "مكافحة الآفات", "تحضير التربة"

• Scenario-Specific Responses:
  1. For Disease Diagnosis:
     - "تظهر الأعراض في شكل..."
     - "يمكن تشخيص المرض من خلال..."
     - "تتطلب المعالجة استخدام..."
  
  2. For Agricultural Guidance:
     - "يفضل اتباع الخطوات التالية..."
     - "من الضروري مراعاة..."
     - "ينصح بتطبيق..."

[FOR FRENCH - ONLY USE IF language="french"]
• Response Structure:
  - Start professionally: "Bonjour" or "Cher agriculteur"
  - Use formal French with agricultural expertise
  - End courteously: "Cordialement" or "Bien à vous"

• Professional Patterns:
  - Introductions: "Tout d'abord", "En premier lieu", "Pour commencer"
  - Transitions: "Par ailleurs", "En ce qui concerne", "De plus"
  - Emphasis: "Il est essentiel", "Il faut noter", "Il est crucial"
  - Conclusions: "En conclusion", "Pour résumer", "Enfin"

• Agricultural Terminology:
  - Technical: "La lutte intégrée", "La fertilisation", "L'irrigation"
  - Scientific: "Les maladies cryptogamiques", "Les agents pathogènes"
  - Practical: "La taille des arbres", "Le désherbage", "Le labour"

• Scenario-Specific Responses:
  1. For Disease Management:
     - "Les symptômes se manifestent par..."
     - "Le traitement nécessite..."
     - "Pour prévenir la maladie..."
  
  2. For Cultivation Advice:
     - "La période optimale pour..."
     - "Il est recommandé de..."
     - "Veillez à respecter..."

[FOR ENGLISH - ONLY USE IF language="english"]
• Response Structure:
  - Start professionally: "Hello" or "Dear farmer"
  - Use clear, technical yet accessible language
  - End helpfully: "Best regards" or "Wishing you success"

• Professional Patterns:
  - Introductions: "First", "To begin with", "Initially"
  - Transitions: "Furthermore", "Moreover", "Additionally"
  - Emphasis: "It's crucial", "Please note", "It's essential"
  - Conclusions: "In conclusion", "Finally", "To summarize"

• Agricultural Terminology:
  - Technical: "Integrated pest management", "Soil amendment", "Drip irrigation"
  - Scientific: "Fungal diseases", "Pathogenic agents", "Nutrient deficiency"
  - Practical: "Pruning", "Crop rotation", "Soil preparation"

• Scenario-Specific Responses:
  1. For Disease Management:
     - "The symptoms appear as..."
     - "Treatment requires..."
     - "To prevent the disease..."
  
  2. For Cultivation Advice:
     - "The optimal time for..."
     - "It's recommended to..."
     - "Make sure to..."

Context: {context}
Question: {question}

Remember: 
1. Respond ONLY in {language}
2. Use ONLY the patterns and terminology specified for {language} above
3. Match the formality level of the user's question
4. Provide practical, actionable advice
5. Include specific examples when possible
6. End with an encouraging note in the appropriate language
"""
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        logger.info(f"LLMService initialized with model {model_name}")

    def format_docs(self, docs):
        try:
            logger.info(f"Raw retriever output: {[doc.__dict__ for doc in docs]}")
            
            # Load a lightweight model for similarity checking
            similarity_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            # Extract document texts
            doc_texts = [doc.page_content for doc in docs]
            
            # Compute embeddings for similarity
            embeddings = similarity_model.encode(doc_texts, convert_to_tensor=True)
            
            # Deduplicate based on cosine similarity
            unique_docs = []
            seen_indices = set()
            
            for i, doc in enumerate(doc_texts):
                if i in seen_indices:
                    continue
                unique_docs.append(doc)
                seen_indices.add(i)
                
                # Compare with other documents
                for j in range(i + 1, len(doc_texts)):
                    if j in seen_indices:
                        continue
                    similarity = util.cos_sim(embeddings[i], embeddings[j]).item()
                    if similarity > 0.9:  # Threshold for considering documents as duplicates
                        seen_indices.add(j)
            
            # Format unique documents
            formatted = "\n\n".join(unique_docs)
            logger.info(f"Formatted context (after deduplication): {formatted}")
            return formatted
        except Exception as e:
            logger.error(f"Error formatting docs: {str(e)}")
            return ""

    def detect_topic(self, query: str) -> str:
        """Detect the topic from the query based on keyword matching with priority."""
        query_lower = query.lower()
        best_match = None
        max_matches = 0
        
        for topic, keywords in self.topic_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            # Prioritize longer keyword matches
            weighted_matches = sum(len(keyword) for keyword in keywords if keyword in query_lower)
            if weighted_matches > max_matches:
                max_matches = weighted_matches
                best_match = topic
        
        if best_match:
            logger.info(f"Detected topic: {best_match}")
            return best_match
            
        logger.info(f"No topic detected for query: {query}")
        return ""

    def check_repetitive_patterns(self, text: str) -> bool:
        """Check if the response has repetitive patterns that suggest the LLM is stuck."""
        # Look for the same sentence repeated multiple times
        sentences = re.split(r'[.!?]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if sentences[i] == sentences[j] and len(sentences[i]) > 10:
                    return True
                    
        # Check for repetitive structural patterns
        if text.count("### ") > 3 or text.count("## ") > 2:
            sections = re.split(r'###|##', text)
            if len(sections) > 3 and len(set(sections)) < len(sections) * 0.7:
                return True
                
        return False

    def postprocess_response(self, response_text: str) -> str:
        """Clean up common issues in responses to improve fluency."""
        # Remove duplicate greetings
        greeting_patterns = ["Salem, ena nsajjel 3la l zira3a w l nabat!", 
                             "Ahlan!", "Sam7ni", "Marhbe"]
        
        # Count occurrences of each greeting
        for pattern in greeting_patterns:
            if response_text.count(pattern) > 1:
                # Keep only the first occurrence
                first_pos = response_text.find(pattern)
                response_text = response_text[:first_pos + len(pattern)] + \
                               response_text[first_pos + len(pattern):].replace(pattern, "")
        
        # Clean up excessive markdown formatting that disrupts flow
        if "### " in response_text and "## Detected Disease" not in response_text:
            response_text = response_text.replace("### ", "**").replace("\n- ", "\n• ")
        
        # Remove duplicate information in the response
        lines = response_text.split("\n")
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        
        # Join unique lines back together
        cleaned_response = "\n".join(unique_lines)
        
        return cleaned_response

    def save_good_response(self, question: str, response: str, language: str, topic: str):
        """Save good examples of responses for future model improvement."""
        try:
            with open("good_responses.jsonl", "a") as f:
                import json
                example = {
                    "question": question,
                    "response": response,
                    "language": language,
                    "topic": topic,
                    "timestamp": str(datetime.now())
                }
                f.write(json.dumps(example) + "\n")
        except Exception as e:
            logger.error(f"Error saving good response example: {str(e)}")

    async def process_query(self, request: ChatRequest, weather_service: WeatherService) -> ChatResponse:
        try:
            # Get weather data if relevant
            weather_data = ""
            if request.country and ("weather" in request.question.lower() or "grow" in request.question.lower()):
                weather_data = await weather_service.fetch_weather_summary(request.country)
            
            # Extract question string
            question_str = str(request.question).strip()
            logger.info(f"Processing question: {question_str}")
            
            # Get language
            language = detect_language(question_str)
            logger.info(f"Detected language: {language}")
            
            # Detect topic
            topic = self.detect_topic(question_str)
            
            # Handle last topic request
            last_topic = self.last_topic or ""
            if "last topic" in question_str.lower() or "previous topic" in question_str.lower():
                if last_topic:
                    topic = last_topic
                    logger.info(f"Using last topic: {last_topic}")
                else:
                    topic = ""
                    logger.info("No last topic available")
            
            # Get context from retriever with topic filter
            try:
                # Create proper where clause for Chroma
                where_clause = {
                    "$and": [
                        {"language": language},
                        {"topic": topic} if topic else {"type": {"$ne": "conversation_history"}}
                    ]
                }
                
                context = self.chroma_service.retrieve_context(
                    question_str, 
                    max_results=5,
                    where=where_clause
                )
                sources = []
                if context:
                    # Extract sources from context
                    docs = context.split("---")
                    for doc in docs:
                        if "source_file" in doc:
                            source = doc.split("source_file:")[-1].split("\n")[0].strip()
                            if source and source != "Unknown":
                                sources.append(source)
                
                # Query for document types with fixed where clause
                doc_types = [meta.get("type", "unknown") for meta in self.chroma_service.collection.query(
                    query_embeddings=[self.embedding_model.embed_query(question_str)],
                    n_results=5,
                    where=where_clause
                )["metadatas"][0]] if context else []
                logger.info(f"Retrieved document types: {doc_types}")
            except Exception as e:
                logger.error(f"Retriever error: {str(e)}")
                context = ""
                sources = []
            
            # Get conversation history
            history = "\n".join([str(msg.content) for msg in self.memory.messages]) if self.memory.messages else ""
            
            # Use fallback response if no context and topic is known
            if not context and topic in self.topic_fallbacks:
                context = self.topic_fallbacks[topic][language]
                logger.info(f"Using fallback context for topic: {topic}")
            
            # Prepare input for prompt
            prompt_input = {
                "question": question_str,
                "detected_disease": str(request.detected_disease or "No disease detected"),
                "topic": topic or "Unknown",
                "last_topic": last_topic,
                "conversation_history": history,
                "context": context,
                "weather_data": str(weather_data),
                "language": language
            }
            
            # Log prompt input
            logger.info(f"Prompt input: {prompt_input}")
            
            # Generate prompt
            try:
                prompt = self.prompt.format_messages(**prompt_input)
                logger.info(f"Formatted prompt: {prompt}")
            except Exception as e:
                logger.error(f"Prompt formatting error: {str(e)}")
                # Fallback prompt
                prompt = f"Answer this question in {language} about {topic or 'agriculture'}: {question_str}"
            
            # Invoke LLM
            try:
                response_text = self.llm.invoke(prompt)
                response_text = str(response_text).strip()
                
                # Check for repetitive patterns
                if self.check_repetitive_patterns(response_text):
                    logger.warning("Detected repetitive patterns in response, regenerating...")
                    # Try regenerating with slightly different parameters
                    temp_llm = OllamaLLM(
                        model=self.model_name,
                        temperature=0.85,
                        top_p=0.95,
                        repetition_penalty=1.2,
                        max_tokens=400
                    )
                    response_text = temp_llm.invoke(prompt)
                    response_text = str(response_text).strip()
                
                # Post-process the response to improve fluency
                response_text = self.postprocess_response(response_text)
                
            except Exception as e:
                logger.error(f"LLM invocation error: {str(e)}")
                # Fallback response in the detected language
                if language == "tounsi":
                    response_text = "Sam7ni, ma najjamtsh njaweb mzyan. 3awd essay wala 3andek su2al ekher?"
                else:
                    response_text = "Sorry, there was an error processing your request. Please try again."
            
            # Log response
            logger.info(f"LLM response: {response_text}")
            
            # Store conversation in memory
            self.memory.add_user_message(question_str)
            self.memory.add_ai_message(response_text)
            self.last_topic = topic if topic else last_topic  # Update last topic
            
            # Save good responses for future improvements
            self.save_good_response(question_str, response_text, language, topic)
            
            # Return ChatResponse with sources and language
            return ChatResponse(
                response=response_text,
                sources=sources,
                language=language
            )
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise