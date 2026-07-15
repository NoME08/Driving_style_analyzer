"""Lightweight session catalog — scan & manage trip data files."""

import os
import json
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
CATALOG = os.path.join(DATA_DIR, "_catalog.json")


def scan_data_files():
    """Scan data/ for CSV trip files (exclude annotations, catalog, udds)."""
    if not os.path.isdir(DATA_DIR):
        return []
    files = []
    for f in sorted(os.listdir(DATA_DIR), reverse=True):
        if not f.endswith(".csv"):
            continue
        if f == "udds.csv":
            continue
        full = os.path.join(DATA_DIR, f)
        files.append({"name": f, "path": full, "size_kb": round(os.path.getsize(full) / 1024, 1)})
    return files


def load_catalog():
    """Load catalog from disk (filenames → metadata)."""
    if os.path.exists(CATALOG):
        with open(CATALOG, "r") as fh:
            return json.load(fh)
    return {}


def save_catalog(cat):
    """Persist catalog to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CATALOG, "w") as fh:
        json.dump(cat, fh, indent=2, ensure_ascii=False)


def update_entry(filename, summary, features):
    """Create or update a catalog entry for a trip file."""
    cat = load_catalog()
    cat[filename] = {
        "summary": summary,
        "features": features,
        "analyzed_at": datetime.now().isoformat(),
    }
    save_catalog(cat)


def delete_file(filename):
    """Delete a trip CSV file and its catalog entry."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
    cat = load_catalog()
    cat.pop(filename, None)
    save_catalog(cat)
