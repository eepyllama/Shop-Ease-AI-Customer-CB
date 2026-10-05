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

# ── Resolution / escalation guardrails (all overridable via env) ─────────────
MAX_AUTO_APPROVE_ABS: float = float(os.getenv("MAX_AUTO_APPROVE_ABS", "500.0"))
POLICY_CONFIDENCE_MIN: float = float(os.getenv("POLICY_CONFIDENCE_MIN", "0.60"))
MAX_RESOLUTION_ATTEMPTS: int = int(os.getenv("MAX_RESOLUTION_ATTEMPTS", "3"))
MAX_REJECTION_ATTEMPTS: int = int(os.getenv("MAX_REJECTION_ATTEMPTS", "2"))
FULL_REFUND_MIN_SEVERITY: str = os.getenv("FULL_REFUND_MIN_SEVERITY", "HIGH")
ALLERGEN_ALWAYS_ESCALATE: bool = os.getenv("ALLERGEN_ALWAYS_ESCALATE", "true").lower() == "true"
REPEAT_RESPONSE_THRESHOLD: float = float(os.getenv("REPEAT_RESPONSE_THRESHOLD", "0.92"))
FRUSTRATION_THRESHOLD: int = int(os.getenv("FRUSTRATION_THRESHOLD", "70"))

# In dev, re-raise LLM errors instead of falling back silently.
# Set DEV_FAIL_LOUD=true in .env to surface API errors immediately.
DEV_FAIL_LOUD: bool = os.getenv("DEV_FAIL_LOUD", "false").lower() == "true"

# ── Photo-evidence verification ───────────────────────────────────────────────
# Minimum risk_score (inclusive) that triggers evidence request for non-HIGH severity
EVIDENCE_RISK_THRESHOLD: int = int(os.getenv("EVIDENCE_RISK_THRESHOLD", "40"))
# Groq vision model used for image analysis (must be a vision-capable model)
VISION_MODEL_ID: str = os.getenv("VISION_MODEL_ID", "meta-llama/llama-4-scout-17b-16e-instruct")
