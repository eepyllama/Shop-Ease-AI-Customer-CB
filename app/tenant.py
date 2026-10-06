"""
Tenant management — load and cache per-business configuration.

Each business that deploys the ShopEase platform is a tenant.  Tenant configs
are stored as JSON files under data/tenants/ and seeded into the DB by
setup_db.py.  The food delivery demo runs as tenant_id='demo_foodco'.

Public interface
----------------
  get_tenant_config(tenant_id)  -> dict | None
  list_tenants()                -> list[dict]
  DEFAULT_TENANT_ID             = 'demo_foodco'
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.tools import _conn

logger = logging.getLogger(__name__)

DEFAULT_TENANT_ID = "demo_foodco"
_TENANT_DIR = Path(__file__).parent.parent / "data" / "tenants"


# ── Cache (cleared only on hot-reload in dev) ─────────────────────────────────

@lru_cache(maxsize=32)
def _cached_config(tenant_id: str) -> Optional[dict]:
    """Load config from DB; fall back to JSON file on cache miss."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT config_json FROM tenant_configs WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
        if row:
            return json.loads(row["config_json"])
    except Exception as exc:
        logger.warning("tenant: DB read failed for %s: %s", tenant_id, exc)

    # Fallback: read from JSON file (useful before setup_db.py has been run)
    path = _TENANT_DIR / f"{tenant_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("tenant: cannot load %s: %s", path, exc)

    return None


def get_tenant_config(tenant_id: str) -> Optional[dict]:
    """Return the full config dict for a tenant, or None if unknown."""
    cfg = _cached_config(tenant_id)
    if cfg is None:
        logger.warning("tenant: unknown tenant_id '%s', falling back to default", tenant_id)
    return cfg


def get_tenant_widget_config(tenant_id: str) -> dict:
    """
    Return just the widget-safe config for the given tenant.
    Falls back to demo_foodco if the tenant is unknown.
    Never exposes server-side secrets (escalation email, etc.).
    """
    cfg = get_tenant_config(tenant_id) or get_tenant_config(DEFAULT_TENANT_ID) or {}
    widget = dict(cfg.get("widget", {}))
    contact = dict(cfg.get("contact", {}))
    return {
        "tenant_id":    cfg.get("tenant_id", DEFAULT_TENANT_ID),
        "name":         cfg.get("name", "Support"),
        "industry":     cfg.get("industry", "general"),
        "widget":       widget,
        "contact":      contact,
    }


def list_tenants() -> list[dict]:
    """Return a summary of all registered tenants."""
    try:
        with _conn() as conn:
            rows = conn.execute(
                "SELECT tenant_id, name, industry, status FROM tenants ORDER BY tenant_id"
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        pass

    # Fallback: scan data/tenants/ directory
    tenants = []
    for path in sorted(_TENANT_DIR.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            tenants.append({
                "tenant_id": d.get("tenant_id", path.stem),
                "name":      d.get("name", path.stem),
                "industry":  d.get("industry", ""),
                "status":    d.get("status", "active"),
            })
        except Exception:
            pass
    return tenants


def clear_cache() -> None:
    """Clear the LRU config cache (useful after an in-process config update)."""
    _cached_config.cache_clear()
