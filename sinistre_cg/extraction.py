"""Extraction du texte d'un PDF ligne par ligne, avec position et typographie.

Usage :
    python -m sinistre_cg.extraction data/raw/cg.pdf data/processed/cg_lines.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pymupdf

# Bit "gras" dans le champ `flags` d'un span PyMuPDF.
BOLD_FLAG = 1 << 4


def _span_to_dict(span: dict) -> dict:
    return {
        "text": span["text"],
        "font": span["font"],
        "size": round(span["size"], 2),
        "bold": bool(span["flags"] & BOLD_FLAG) or "bold" in span["font"].lower(),
        "x0": round(span["bbox"][0], 2),
    }


def _dominant(spans: list[dict], key: str):
    """Valeur de `key` portant le plus de caractères (non blancs) dans la ligne."""
    weights = Counter()
    for span in spans:
        weights[span[key]] += len(span["text"].strip())
    return weights.most_common(1)[0][0]


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """Dimensions de chaque page (numérotées à partir de 1)."""
    with pymupdf.open(pdf_path) as doc:
        return [
            {"page": page.number + 1, "width": round(page.rect.width, 2), "height": round(page.rect.height, 2)}
            for page in doc
        ]


def extract_lines(pdf_path: str | Path) -> list[dict]:
    """Retourne une entrée par ligne de texte non vide, triée par position sur la page."""
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            data = page.get_text("dict", sort=True)
            for block_no, block in enumerate(data["blocks"]):
                if block["type"] != 0:  # 0 = texte, 1 = image
                    continue
                for line in block["lines"]:
                    spans = [_span_to_dict(s) for s in line["spans"] if s["text"]]
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    x0, y0, x1, y1 = (round(v, 2) for v in line["bbox"])
                    lines.append({
                        "page": page.number + 1,
                        "block": block_no,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "text": text,
                        "font": _dominant(spans, "font"),
                        "size": _dominant(spans, "size"),
                        "bold": _dominant(spans, "bold"),
                        "spans": spans,
                    })
    return lines


def save_json(lines: list[dict], pages: list[dict], out_path: str | Path, source: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source": Path(source).name, "pages": pages, "lines": lines}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = extract_lines(args.pdf)
    save_json(lines, extract_pages(args.pdf), args.output, args.pdf)
    pages = len({line["page"] for line in lines})
    print(f"{len(lines)} lignes extraites sur {pages} pages -> {args.output}")


if __name__ == "__main__":
    main()
