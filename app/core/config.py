from pydantic import BaseModel
import os

class Settings(BaseModel):
    app_name: str = "Mezan Credit Scoring API"
    version: str = "0.1.0"
    model_path: str = os.getenv("MODEL_PATH", "outputs/models/credit_engine.pkl")

settings = Settings()