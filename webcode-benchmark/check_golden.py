"""Check which golden_reference files are missing or too short (<1000 chars)."""

import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data/contents/code_contents.jsonl"
GOLDEN_DIR = Path(__file__).parent / "golden_output/golden_reference"

queries = []
with open(DATA_FILE) as f:
    for line in f:
        if line.strip():
            queries.append(json.loads(line))

missing = []
too_short = []
timed_out = []

for q in queries:
    id_ = q["id"]
    url = q["url"]
    path = GOLDEN_DIR / f"{id_}.md"

    if not path.exists():
        missing.append((id_, url))
    else:
        content = path.read_text(errors="replace")
        length = len(content)
        if length < 1000:
            too_short.append((id_, url, length))
        if "connection timed out" in content.lower():
            timed_out.append((id_, url))

print(f"Total queries: {len(queries)}")
print(f"\nMissing ({len(missing)}):")
for id_, url in missing:
    print(f"  {id_}  {url}")

print(f"\nToo short (<1000 chars) ({len(too_short)}):")
for id_, url, length in too_short:
    print(f"  {id_}  {length} chars  {url}")

print(f"\nConnection timed out ({len(timed_out)}):")
for id_, url in timed_out:
    print(f"  {id_}  {url}")
