import requests
from pydantic import ValidationError
from models.user import User
from config.settings import Settings

class UserService:
    def __init__(self, settings: Settings):
        self.base_url = settings.nestjs_api_base_url.rstrip("/")

    def get_user_by_id(self, user_id: str) -> User:
        """
        Fetch a user by ID from the NestJS backend.
        
        Args:
            user_id: The ID of the user to fetch.
        
        Returns:
            User: The user object parsed from the response.
        
        Raises:
            ValueError: If the request fails or the response is invalid.
            ValidationError: If the response doesn't match the User model.
        """
        url = f"{self.base_url}/get-account/{user_id}"
        
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raises an HTTPError for 4xx/5xx responses
            
            user_data = response.json()
            return User(**user_data)
        
        except requests.exceptions.HTTPError as http_err:
            raise ValueError(f"Failed to fetch user {user_id}: {response.status_code} {response.text}")
        except requests.exceptions.RequestException as req_err:
            raise ValueError(f"Error connecting to NestJS backend: {req_err}")
        except ValidationError as val_err:
            raise ValueError(f"Invalid user data received: {val_err}")