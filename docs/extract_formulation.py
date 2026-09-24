"""
extract_formulation.py

Reads main.md from docs/ and extracts the Methods section
into formulation_only.md. Overwrites on every run.

Usage: python3 docs/extract_formulation.py
"""

import os
import re

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_FILE = os.path.join(DOCS_DIR, "main.md")
OUTPUT_FILE = os.path.join(DOCS_DIR, "formulation_only.md")

with open(MAIN_FILE, "r") as f:
    content = f.read()

# Find Methods and Results section headers regardless of heading level
start = re.search(r'^#{1,3}\s+Methods', content, re.MULTILINE)
end = re.search(r'^#{1,3}\s+Results', content, re.MULTILINE)

if not start:
    raise ValueError("Could not find Methods section — check heading name in exported markdown")
if not end:
    raise ValueError("Could not find Results section — check heading name in exported markdown")

formulation = content[start.start():end.start()]

with open(OUTPUT_FILE, "w") as f:
    f.write(formulation)

print(f"Extracted {len(formulation):,} characters")
print(f"Written to: {OUTPUT_FILE}")
print(f"\nPreview (first 300 chars):\n{formulation[:300]}")
