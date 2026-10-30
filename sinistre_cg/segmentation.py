"""Découpage des conditions générales en articles.

Un article commence à une ligne « Article N » isolée, placée dans la marge gauche de sa
colonne, avec son titre à droite. Les chapitres sont repérés par une ligne « CHAPITRE N »
accompagnée d'un titre en gros caractères. Le texte qui précède le premier article
forme un morceau « préambule ».

Usage :
    python -m sinistre_cg.segmentation data/processed/cg_lines.json data/processed/articles
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sinistre_cg.layout import layout_document

ARTICLE_RE = re.compile(r"^article\s+(\d+\s*[a-z]?)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^chapitre\s+(\d+)$", re.IGNORECASE)

# Écart vertical maximal (pt) entre deux éléments considérés sur la même ligne.
SAME_ROW = 3.0
# Un titre de chapitre est au moins ce multiple de la taille du corps de texte.
CHAPTER_TITLE_SCALE = 1.5
# Interligne maximal, en multiple de la taille de police, entre deux lignes d'un même titre.
TITLE_LINE_GAP = 1.6
# Une étiquette en marge ne commence pas plus à droite que le bord de sa colonne (+ tolérance, pt).
MARGIN_TOLERANCE = 2.0


def _body_size(lines: list[dict]) -> float:
    return Counter(line["size"] for line in lines).most_common(1)[0][0]


def _column_left_edges(lines: list[dict]) -> dict:
    """Abscisse la plus fréquente de début de ligne, par (page, colonne)."""
    starts = {}
    for line in lines:
        starts.setdefault((line["page"], line["column"]), Counter())[round(line["x0"])] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in starts.items()}


def _row_neighbour(lines: list[dict], i: int) -> int | None:
    """Indice de la ligne située juste à droite de lines[i], sur la même rangée,
    dans la même colonne ou en pleine largeur."""
    ref = lines[i]
    candidates = [
        j for j, line in enumerate(lines)
        if j != i and line["page"] == ref["page"] and line["column"] in (ref["column"], None)
        and abs(line["y0"] - ref["y0"]) <= SAME_ROW and line["x0"] >= ref["x1"]
    ]
    return min(candidates, key=lambda j: lines[j]["x0"], default=None)


def _title_block(lines: list[dict], start: int) -> list[int]:
    """Indices des lignes qui forment un titre avec lines[start] : même police, même
    taille, même colonne, et verticalement contiguës (au-dessus comme en dessous)."""
    ref = lines[start]
    same_style = sorted(
        (j for j, line in enumerate(lines)
         if line["page"] == ref["page"] and line["column"] == ref["column"]
         and line["font"] == ref["font"] and line["size"] == ref["size"]),
        key=lambda j: lines[j]["y0"],
    )
    pos = same_style.index(start)
    max_gap = ref["size"] * TITLE_LINE_GAP
    lo = hi = pos
    while lo > 0 and lines[same_style[lo]]["y0"] - lines[same_style[lo - 1]]["y0"] <= max_gap:
        lo -= 1
    while hi < len(same_style) - 1 and lines[same_style[hi + 1]]["y0"] - lines[same_style[hi]]["y0"] <= max_gap:
        hi += 1
    return same_style[lo:hi + 1]


def _join(lines: list[dict], indices: list[int]) -> str:
    return re.sub(r"\s+", " ", " ".join(lines[j]["text"] for j in indices)).strip()


def find_chapters(lines: list[dict]) -> list[dict]:
    """Repère les en-têtes de chapitre : « CHAPITRE N » + titre en grands caractères."""
    min_size = _body_size(lines) * CHAPTER_TITLE_SCALE
    chapters = []
    for i, line in enumerate(lines):
        match = CHAPTER_RE.match(line["text"])
        if not match:
            continue
        # Le titre est la plus grande ligne de la même rangée, à gauche ou à droite.
        row = [
            j for j, other in enumerate(lines)
            if j != i and other["page"] == line["page"] and abs(other["y0"] - line["y0"]) <= 2 * SAME_ROW
            and other["size"] >= min_size
        ]
        if not row:
            continue
        title_idx = _title_block(lines, max(row, key=lambda j: lines[j]["size"]))
        chapters.append({
            "index": i,
            "numero": int(match.group(1)),
            "titre": _join(lines, title_idx),
            "lines": {i, *title_idx},
        })
    return chapters


def find_articles(lines: list[dict]) -> list[dict]:
    """Repère les en-têtes d'article : ligne « Article N » seule, dans la marge."""
    left_edges = _column_left_edges(lines)
    articles = []
    for i, line in enumerate(lines):
        match = ARTICLE_RE.match(line["text"])
        if not match:
            continue
        body_left = left_edges[(line["page"], line["column"])]
        if line["x0"] > body_left + MARGIN_TOLERANCE:
            continue
        neighbour = _row_neighbour(lines, i)
        title_idx = _title_block(lines, neighbour) if neighbour is not None else []
        articles.append({
            "index": i,
            "numero": re.sub(r"\s+", "", match.group(1)).upper(),
            "titre": _join(lines, title_idx) or None,
            "lines": {i, *title_idx},
        })
    return articles


def _public_line(line: dict) -> dict:
    keys = ("page", "column", "x0", "y0", "x1", "y1", "text", "font", "size", "bold", "spans")
    return {k: line[k] for k in keys}


def segment(lines: list[dict]) -> list[dict]:
    """Découpe des lignes (déjà dans l'ordre de lecture) en morceaux."""
    chapters = find_chapters(lines)
    articles = find_articles(lines)
    heading_lines = set().union(*(c["lines"] for c in chapters), *(a["lines"] for a in articles))

    starts = [{"index": 0, "numero": None, "titre": "Préambule", "lines": set()}]
    if articles and articles[0]["index"] == 0:
        starts = []
    starts += articles

    pieces = []
    for k, start in enumerate(starts):
        end = starts[k + 1]["index"] if k + 1 < len(starts) else len(lines)
        chapter = next((c for c in reversed(chapters) if c["index"] <= start["index"]), None)
        body = [_public_line(lines[j]) for j in range(start["index"], end) if j not in heading_lines]
        pages = sorted({lines[j]["page"] for j in range(start["index"], end)})
        pieces.append({
            "id": f"article_{start['numero']}" if start["numero"] else "preambule",
            "numero": start["numero"],
            "titre": start["titre"],
            "chapitre": {"numero": chapter["numero"], "titre": chapter["titre"]} if chapter else None,
            "page_debut": pages[0],
            "page_fin": pages[-1],
            "pages": pages,
            "texte": "\n".join(line["text"] for line in body),
            "lignes": body,
        })
    return pieces


def _file_name(piece: dict) -> str:
    if piece["numero"] is None:
        return "article_000_preambule.json"
    match = re.match(r"(\d+)(.*)", piece["numero"])
    return f"article_{int(match.group(1)):03d}{match.group(2)}.json"


def save_pieces(pieces: list[dict], out_dir: Path, source: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for piece in pieces:
        payload = {"source": source, **piece}
        (out_dir / _file_name(piece)).write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lines_json", type=Path, help="sortie de sinistre_cg.extraction")
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    data = json.loads(args.lines_json.read_text(encoding="utf-8"))
    lines = layout_document(data["lines"], data["pages"])
    pieces = segment(lines)
    save_pieces(pieces, args.out_dir, data["source"])
    print(f"{len(pieces)} morceaux -> {args.out_dir}")


if __name__ == "__main__":
    main()
