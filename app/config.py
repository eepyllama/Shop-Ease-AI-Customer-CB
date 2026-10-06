import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3-8b-8192")

# ── OpenRouter fallback (used when Groq is unavailable / key expired) ─────────
# Set OPENROUTER_API_KEY in .env to enable.  Groq is always tried first.
# Get a key at: https://openrouter.ai/keys
OPENROUTER_API_KEY: str  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Any OpenRouter model works; gpt-4o-mini gives a good speed/cost trade-off.
OPENROUTER_MODEL: str    = os.getenv("OPENROUTER_MODEL",    "openai/gpt-4o-mini")

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

# ── Late-delivery severity thresholds (minutes over promised ETA) ─────────────
# Delays shorter than NONE_MIN are not considered a delay at all.
LATE_DELIVERY_NONE_MIN: int    = int(os.getenv("LATE_DELIVERY_NONE_MIN",   "10"))
LATE_DELIVERY_LOW_MIN: int     = int(os.getenv("LATE_DELIVERY_LOW_MIN",    "30"))
LATE_DELIVERY_MEDIUM_MIN: int  = int(os.getenv("LATE_DELIVERY_MEDIUM_MIN", "60"))

# ── ViT Indian food classifier (optional second opinion for wrong_order) ──────
# Model: DrishtiSharma/finetuned-ViT-Indian-Food-Classification-v3
# Labels below MUST match the model's actual output labels exactly.
# Verify against the model card before deploying; add/remove entries freely.
VIT_FOOD_MODEL_ID: str = os.getenv(
    "VIT_FOOD_MODEL_ID",
    "DrishtiSharma/finetuned-ViT-Indian-Food-Classification-v3",
)

# Maps menu dish name → expected ViT label (only dishes known to the classifier).
VIT_LABEL_MAP: dict[str, str] = {
    "Butter Chicken":       "butter_chicken",
    "Chicken Tikka Masala": "chicken_tikka_masala",
    "Paneer Butter Masala": "paneer_butter_masala",
    "Palak Paneer":         "palak_paneer",
    "Paneer Tikka":         "paneer_tikka",
    "Dal Tadka":            "dal_tadka",
    "Chole Bhature":        "chole_bhature",
    "Pav Bhaji":            "pav_bhaji",
    "Masala Dosa":          "masala_dosa",
    "Idli Sambar":          "idli_sambar",
    "Chicken Biryani":      "chicken_biryani",
    "Vegetable Biryani":    "veg_biryani",
    "Gulab Jamun":          "gulab_jamun",
    "Samosa":               "samosa",
}
