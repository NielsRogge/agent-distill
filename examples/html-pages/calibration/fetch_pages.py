"""Rebuild `pages/<id>.html` (the labelled pages) from the published teacher-baseline records.

python examples/html-pages/calibration/fetch_pages.py
"""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent


def main() -> None:
    pages = json.loads((HERE / "pairs.json").read_text())["pages"]
    out = HERE / "pages"
    out.mkdir(exist_ok=True)
    records: dict[tuple[str, str], list[dict]] = {}
    for page_id, source in pages.items():
        key = (source["repo"], source["file"])
        if key not in records:
            path = hf_hub_download(source["repo"], source["file"], repo_type="dataset")
            records[key] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        record = records[key][source["index"]]
        assert str(record["row_id"]) == source["row_id"], (page_id, record["row_id"])
        (out / f"{page_id}.html").write_text(record["answer"], encoding="utf-8")
    print(f"wrote {len(pages)} pages to {out}")


if __name__ == "__main__":
    main()
