"""List the Groq models your API key currently has access to.

Groq changes which models are on the free vs. Enterprise tier from time to
time (this is exactly what broke llama-3.1-8b-instant). Run this whenever
you get a "model not found / no access" error to see what's actually usable,
then update GROQ_MODEL in your .env accordingly.

Usage:
    python scripts/list_groq_models.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from groq import Groq
from src.config import settings

if not settings.GROQ_API_KEY:
    print("GROQ_API_KEY is not set in your .env file.")
    sys.exit(1)

client = Groq(api_key=settings.GROQ_API_KEY)
models = client.models.list()

print(f"{'MODEL ID':45} {'CONTEXT':>10} {'ACTIVE':>8}")
print("-" * 65)
for m in sorted(models.data, key=lambda x: x.id):
    context = getattr(m, "context_window", "?")
    active = getattr(m, "active", "?")
    print(f"{m.id:45} {str(context):>10} {str(active):>8}")

print(f"\nCurrently configured GROQ_MODEL = {settings.GROQ_MODEL}")
