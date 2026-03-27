"""
build_metadata_index.py
-----------------------
After running scrape_website_pages.py and scrape_scholarworks.py,
run this script to produce a single metadata_index.json file
summarizing every document in rawdata/.

This index is what your RAG pipeline will load to know what
documents exist and where they are stored.

Usage:
    python build_metadata_index.py
"""

import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)

DIRS = {
    "website_pages":   os.path.join(BASE_DIR, "website_pages"),
    "institute_report": os.path.join(BASE_DIR, "institute_report"),
    "scholary_papers":  os.path.join(BASE_DIR, "scholary_papers"),
}

records = []

for source_type, folder in DIRS.items():
    if not os.path.isdir(folder):
        print(f"Skipping (folder not found): {folder}")
        continue

    json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
    for fname in sorted(json_files):
        fpath = os.path.join(folder, fname)
        try:
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Could not read {fpath}: {e}")
            continue

        # Build a standardized record
        record = {
            "id":          fname.replace(".json", ""),
            "title":       data.get("title", ""),
            "source_type": data.get("source_type", source_type),
            "url":         data.get("url", ""),
            "pub_date":    data.get("pub_date", ""),
            "authors":     data.get("authors", []),
            "abstract":    data.get("abstract", ""),
            "has_pdf":     "pdf_file" in data,
            "pdf_file":    data.get("pdf_file", ""),
            "text_length": len(data.get("text", "")),
            "folder":      source_type,
            "json_file":   fname,
        }
        records.append(record)

# Save index
out_path = os.path.join(BASE_DIR, "metadata_index.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "built_at": datetime.utcnow().isoformat() + "Z",
        "total_documents": len(records),
        "breakdown": {
            k: sum(1 for r in records if r["folder"] == k) for k in DIRS
        },
        "documents": records,
    }, f, ensure_ascii=False, indent=2)

print(f"\nMetadata index saved → {out_path}")
print(f"Total documents indexed: {len(records)}")
for k in DIRS:
    count = sum(1 for r in records if r["folder"] == k)
    print(f"  {k}: {count} documents")
