"""Mise en page : retrait des en-têtes/pieds de page et ordre de lecture par colonnes."""

from collections import Counter, defaultdict

# Part de la hauteur de page, en haut et en bas, où se trouvent en-têtes et pieds de page.
FURNITURE_BAND = 0.06
# Un texte répété au même endroit sur au moins ce nombre de pages est un en-tête courant.
FURNITURE_MIN_PAGES = 3
# Nombre minimal de lignes alignées dans la moitié droite pour conclure à une 2e colonne.
MIN_COLUMN_LINES = 8
# Tolérance (pt) autour de la gouttière entre colonnes.
GUTTER_TOLERANCE = 6


def _in_band(line: dict, height: float) -> bool:
    return line["y1"] < height * FURNITURE_BAND or line["y0"] > height * (1 - FURNITURE_BAND)


def remove_page_furniture(lines: list[dict], pages: list[dict]) -> list[dict]:
    """Retire numéros de page et en-têtes courants situés dans les marges haute et basse."""
    heights = {p["page"]: p["height"] for p in pages}
    pages_by_text = defaultdict(set)
    for line in lines:
        if _in_band(line, heights[line["page"]]):
            pages_by_text[line["text"]].add(line["page"])

    def is_furniture(line: dict) -> bool:
        if not _in_band(line, heights[line["page"]]):
            return False
        return line["text"].isdigit() or len(pages_by_text[line["text"]]) >= FURNITURE_MIN_PAGES

    return [line for line in lines if not is_furniture(line)]


def find_gutter(page_lines: list[dict], width: float) -> float | None:
    """Abscisse de début de la colonne de droite, ou None si la page n'a qu'une colonne."""
    starts = Counter(
        round(line["x0"]) for line in page_lines if 0.45 * width < line["x0"] < 0.65 * width
    )
    if not starts:
        return None
    x, count = starts.most_common(1)[0]
    if count < MIN_COLUMN_LINES:
        return None
    # La colonne commence à la plus petite abscisse proche du pic (titres en marge inclus).
    return min(s for s in starts if x - 3 * GUTTER_TOLERANCE <= s <= x)


def assign_columns(page_lines: list[dict], gutter: float | None) -> None:
    """Ajoute `column` à chaque ligne : 0 gauche, 1 droite, None pleine largeur."""
    for line in page_lines:
        if gutter is None:
            line["column"] = 0
        elif line["x0"] >= gutter - GUTTER_TOLERANCE:
            line["column"] = 1
        elif line["x1"] <= gutter + GUTTER_TOLERANCE:
            line["column"] = 0
        else:
            line["column"] = None


def reading_order(page_lines: list[dict]) -> list[dict]:
    """Ordonne les lignes d'une page : bandes pleine largeur, et dans chaque bande,
    colonne de gauche puis colonne de droite."""
    ordered, band = [], []

    def flush():
        band.sort(key=lambda l: (l["column"], l["y0"], l["x0"]))
        ordered.extend(band)
        band.clear()

    for line in sorted(page_lines, key=lambda l: (l["y0"], l["x0"])):
        if line["column"] is None:
            flush()
            ordered.append(line)
        else:
            band.append(line)
    flush()
    return ordered


def layout_document(lines: list[dict], pages: list[dict]) -> list[dict]:
    """Lignes nettoyées, annotées de leur colonne et triées dans l'ordre de lecture."""
    widths = {p["page"]: p["width"] for p in pages}
    by_page = defaultdict(list)
    for line in remove_page_furniture(lines, pages):
        by_page[line["page"]].append(dict(line))

    result = []
    for page_no in sorted(by_page):
        page_lines = by_page[page_no]
        assign_columns(page_lines, find_gutter(page_lines, widths[page_no]))
        result.extend(reading_order(page_lines))
    return result
