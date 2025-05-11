from pymongo import MongoClient
from datetime import datetime
from config.settings import Settings

class ConversationService:
    def __init__(self, settings: Settings):
        self.client = MongoClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db_name]
        self.conversations = self.db["conversations"]

    def save_message(self, user_id: str, question: str, response: str):
        message = {
            "user_id": user_id,
            "question": question,
            "response": response,
            "timestamp": datetime.utcnow()
        }
        self.conversations.insert_one(message)

    def get_user_conversations(self, user_id: str):
        return list(self.conversations.find({"user_id": user_id}).sort("timestamp", -1))