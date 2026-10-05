import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")

SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASS: str = os.getenv("SMTP_PASS", "")

DB_PATH: str = os.getenv("DB_PATH", "chatbot.db")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./chroma_db")
POLICY_PATH: str = os.getenv("POLICY_PATH", "data/policy.md")
ESCALATIONS_LOG: str = os.getenv("ESCALATIONS_LOG", "escalations.jsonl")

ALLOWED_ORIGINS: list[str] = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# Use OpenAI embeddings when key is present; otherwise HuggingFace (free, local)
USE_OPENAI_EMBEDDINGS: bool = bool(OPENAI_API_KEY)
