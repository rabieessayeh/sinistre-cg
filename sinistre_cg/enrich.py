"""Enrichissement des articles découpés : nettoyage, sous-parties typées, clauses,
lexique, renvois et termes définis.

Usage :
    python -m sinistre_cg.enrich data/processed/articles data/processed
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

from sinistre_cg.segmentation import _file_name

# --- 0. Nettoyage -------------------------------------------------------------

# Glyphes de puce mal décodés -> caractère lisible.
CHAR_REPLACEMENTS = {"\u0003": "•"}


def _clean_text(text: str) -> str:
    for bad, good in CHAR_REPLACEMENTS.items():
        text = text.replace(bad, good)
    return text


def clean_article(article: dict) -> dict:
    """Remplace les glyphes mal décodés dans `texte`, `titre` et `lignes` (spans compris)."""
    article = dict(article)
    article["texte"] = _clean_text(article["texte"])
    if article.get("titre"):
        article["titre"] = _clean_text(article["titre"])
    lines = []
    for line in article["lignes"]:
        line = dict(line, text=_clean_text(line["text"]))
        if "spans" in line:
            line["spans"] = [dict(s, text=_clean_text(s["text"])) for s in line["spans"]]
        lines.append(line)
    article["lignes"] = lines
    return article


# --- 1. Sous-parties ----------------------------------------------------------

# Début de ligne -> type de sous-partie. Testés dans l'ordre (le plus spécifique d'abord).
SUBPART_MARKERS = [
    ("exclusion", ("Nous ne garantissons pas", "Sont exclus", "Ne sont pas", "Nous n'intervenons pas")),
    ("garantie", ("Nous garantissons", "Sont garantis", "Lorsque vous avez souscrit",
                  "Nous intervenons", "Nous prenons en charge")),
    ("prevention", ("Mesures de prévention", "IMPORTANT")),
    ("condition", ("Inoccupation",)),
]
DEFAULT_SUBPART = "general"
MAX_SUBPART_CHARS = 2500
SENTENCE_END = re.compile(r"[.!?;:]\s*[»”)]?\s*$")


def _normalize_quotes(text: str) -> str:
    return text.replace("’", "'").replace("`", "'")


def line_type(text: str) -> str | None:
    """Type annoncé par le début de la ligne, ou None si la ligne n'en annonce aucun."""
    start = _normalize_quotes(text.lstrip())
    for kind, prefixes in SUBPART_MARKERS:
        if start.startswith(prefixes):
            return kind
    return None


def _make_subpart(kind: str, lines: list[dict]) -> dict:
    return {
        "type": kind,
        "texte": "\n".join(line["text"] for line in lines),
        "pages": sorted({line["page"] for line in lines}),
    }


def _split_long(lines: list[dict], max_chars: int) -> list[list[dict]]:
    """Coupe une suite de lignes en morceaux de moins de `max_chars` caractères,
    de préférence après une ligne qui termine une phrase."""
    chunks, current, size, last_cut = [], [], 0, None
    for line in lines:
        current.append(line)
        size += len(line["text"]) + 1
        if SENTENCE_END.search(line["text"]):
            last_cut = len(current)
        if size > max_chars and len(current) > 1:
            cut = last_cut if last_cut else len(current) - 1
            chunks.append(current[:cut])
            current = current[cut:]
            size = sum(len(l["text"]) + 1 for l in current)
            last_cut = None
            for k, l in enumerate(current, 1):
                if SENTENCE_END.search(l["text"]):
                    last_cut = k
    if current:
        chunks.append(current)
    return chunks


def split_subparts(lines: list[dict], max_chars: int = MAX_SUBPART_CHARS) -> list[dict]:
    """Découpe des lignes en sous-parties typées. Une nouvelle sous-partie commence à
    chaque ligne dont le début annonce un type différent du type courant."""
    groups, current_kind, current = [], DEFAULT_SUBPART, []
    for line in lines:
        kind = line_type(line["text"])
        if kind is not None and kind != current_kind and current:
            groups.append((current_kind, current))
            current = []
        if kind is not None:
            current_kind = kind
        current.append(line)
    if current:
        groups.append((current_kind, current))

    return [
        _make_subpart(kind, chunk)
        for kind, group in groups
        for chunk in _split_long(group, max_chars)
    ]


# --- 2. Clauses ---------------------------------------------------------------

CLAUSE_CODE_RE = re.compile(r"^[A-Z0-9]{1,4}$")
SAME_ROW = 4.0
# Le titre commence au moins à cette distance (pt) à droite du code.
CODE_TITLE_MIN_OFFSET = 8.0


def _clause_title_line(code_line: dict, line: dict) -> bool:
    """`line` est-elle le titre de la clause dont `code_line` porte le code ?"""
    if not line["bold"] or line is code_line:
        return False
    if line["page"] != code_line["page"] or line["column"] != code_line["column"]:
        return False
    if abs(line["y0"] - code_line["y0"]) > SAME_ROW:
        return False
    if line["x0"] < code_line["x0"] + CODE_TITLE_MIN_OFFSET:
        return False
    text = line["text"].strip()
    # Certains titres répètent le code (« 6 - Exclusion du mobilier »).
    if text.startswith(code_line["text"].strip()):
        text = text[len(code_line["text"].strip()):].lstrip()
    return text.startswith("-")


def _clean_title(text: str, code: str) -> str:
    text = text.strip()
    if text.startswith(code):
        text = text[len(code):]
    text = re.sub(r"\s+", " ", text.lstrip(" -–")).strip()
    return re.sub(r"\s*:$", "", text)


def find_clauses(article: dict, max_chars: int = MAX_SUBPART_CHARS) -> list[dict]:
    """Repère les clauses d'un article : code court en gras dans la marge, suivi sur la
    même rangée d'un titre en gras commençant par « - »."""
    lines = article["lignes"]
    heads = []  # (début, indices du titre, indice du code)
    used = set()
    for i, line in enumerate(lines):
        if not line["bold"] or not CLAUSE_CODE_RE.match(line["text"].strip()) or i in used:
            continue
        title = next((j for j, other in enumerate(lines) if _clause_title_line(line, other)), None)
        if title is None:
            continue
        title_idx = [title]
        # Suite du titre : lignes en gras commençant par une minuscule (le code, lu
        # parfois juste après le titre, est sauté).
        k = title + 1
        while k < len(lines):
            if k != i:
                if not (lines[k]["bold"] and lines[k]["text"].lstrip()[:1].islower()):
                    break
                title_idx.append(k)
            k += 1
        used.update({i, *title_idx})
        heads.append((min(i, title), title_idx, i))

    heads.sort()
    clauses = []
    for n, (start, title_idx, code_idx) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        code = lines[code_idx]["text"].strip()
        header = {code_idx, *title_idx}
        body = [lines[j] for j in range(start, end) if j not in header]
        span = [lines[j] for j in range(start, end)]
        clauses.append({
            "code": code,
            "titre": _clean_title(" ".join(lines[j]["text"] for j in title_idx), code),
            "article": article["numero"],
            "texte": "\n".join(line["text"] for line in body),
            "pages": sorted({line["page"] for line in span}),
            "sous_parties": split_subparts(body, max_chars),
        })
    return clauses


# --- 3. Lexique ---------------------------------------------------------------

LEXICON_FONT_PREFIX = "ArialRounded"


def _term_split(line: dict) -> tuple[str, str] | None:
    """(terme, début de définition) si la ligne ouvre une entrée du lexique."""
    spans = line.get("spans") or []
    if not spans or not spans[0]["font"].startswith(LEXICON_FONT_PREFIX):
        return None
    # Le terme s'étend sur les spans en police du lexique ou en gras, jusqu'au « : ».
    head = ""
    for span in spans:
        if not (span["font"].startswith(LEXICON_FONT_PREFIX) or span["bold"]):
            break
        head += span["text"]
    if ":" in head:
        term = head.split(":", 1)[0]
        rest = line["text"].split(":", 1)[1]
    else:
        term = head
        rest = line["text"][len(head):]
    return re.sub(r"\s+", " ", term).strip(), rest.strip()


def parse_lexicon(article: dict) -> dict:
    """{terme: {definition, page}} : chaque terme court jusqu'au terme suivant."""
    definitions, current = {}, None
    for line in article["lignes"]:
        split = _term_split(line)
        if split:
            term, rest = split
            current = {"definition": [rest] if rest else [], "page": line["page"]}
            definitions[term] = current
        elif current is not None:
            current["definition"].append(line["text"])
    return {
        term: {"definition": "\n".join(entry["definition"]).strip(), "page": entry["page"]}
        for term, entry in definitions.items()
    }


# --- 4. Renvois et termes définis ---------------------------------------------

_NUM = r"\d+[A-Z]?(?:-\d+(?:-?[A-Z])?)*"
_SEP = r"\s*(?:,|\bet\b|\bou\b|\bà\b)\s*"
ARTICLE_REF_RE = re.compile(rf"\b(?:art\.|articles?)\s*({_NUM}(?:{_SEP}{_NUM})*)", re.IGNORECASE)
LEGAL_SOURCE_RE = re.compile(r"^\W{0,3}(?:du|de la|des|de l')?\s*(?:Code|code|loi|Loi|décret|Décret)\b")
_CODE = r"(?:\d{1,3}(?:\s?[A-Z]\b)?|[A-Z]{1,4}\d{0,3})"
CLAUSE_REF_RE = re.compile(rf"\b[Cc]lauses?\s+({_CODE}(?:\s*,?\s*(?:\bou\b|\bet\b)?\s*{_CODE})*)")


def _article_base(ref: str) -> str:
    """« 2-1B » -> « 2 », « 42A » -> « 42A »."""
    return re.match(r"\d+[A-Z]?", ref).group(0)


def find_article_refs(text: str, known: set[str], self_numero: str | None = None) -> list[str]:
    """Numéros d'articles du contrat cités dans `text` (hors codes juridiques et hors
    l'article lui-même). Les plages « 44 à 47 » sont développées."""
    found = []
    for match in ARTICLE_REF_RE.finditer(text):
        if LEGAL_SOURCE_RE.match(text[match.end():match.end() + 20]):
            continue
        group = match.group(1)
        tokens = re.split(rf"({_SEP})", group)
        nums = tokens[0::2]
        seps = [s.strip() for s in tokens[1::2]]
        refs = [_article_base(nums[0])]
        for sep, num in zip(seps, nums[1:]):
            base = _article_base(num)
            if sep == "à" and refs[-1].isdigit() and base.isdigit() and int(refs[-1]) < int(base):
                refs.extend(str(n) for n in range(int(refs[-1]) + 1, int(base)))
            refs.append(base)
        found.extend(refs)
    return _unique(r for r in found if r in known and r != self_numero)


def find_clause_refs(text: str, known: set[str], self_code: str | None = None) -> list[str]:
    """Codes de clauses cités dans `text` (« clause 22 B » -> « 22B »)."""
    found = []
    for match in CLAUSE_REF_RE.finditer(text):
        for code in re.findall(_CODE, match.group(1)):
            found.append(code.replace(" ", ""))
    return _unique(c for c in found if c in known and c != self_code)


def _unique(items) -> list:
    return list(dict.fromkeys(items))


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", _normalize_quotes(text).lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(?:l|d|qu|j|n|s)'", " ", text)  # élisions : l'assuré -> assuré
    text = re.sub(r"[“”\"«»]", " ", text)
    return " ".join(_singular(w) for w in re.findall(r"[a-z0-9']+", text))


def _singular(word: str) -> str:
    return word[:-1] if len(word) > 3 and word[-1] in "sx" else word


class TermIndex:
    """Rattache une forme suivie d'un astérisque au terme exact du lexique."""

    MAX_WORDS = 6
    MIN_PREFIX = 6

    def __init__(self, terms):
        self.by_norm = {}
        for term in terms:
            self.by_norm.setdefault(_norm(term), term)
            # « Objets/Biens précieux » : chaque variante est une entrée.
            for variant in term.split("/"):
                self.by_norm.setdefault(_norm(variant), term)

    def lookup(self, words: list[str]) -> tuple[str, str | None]:
        """(forme retenue, terme du lexique ou None) pour les mots précédant un « * »."""
        for k in range(min(self.MAX_WORDS, len(words)), 0, -1):
            form = " ".join(words[-k:])
            term = self.by_norm.get(_norm(form))
            if term:
                return form, term
        last = words[-1]
        stem = _norm(last)
        candidates = {
            term for key, term in self.by_norm.items()
            if any(self._shares_prefix(stem, w) for w in key.split())
        }
        return last, candidates.pop() if len(candidates) == 1 else None

    def _shares_prefix(self, a: str, b: str) -> bool:
        n = 0
        while n < min(len(a), len(b)) and a[n] == b[n]:
            n += 1
        return n >= self.MIN_PREFIX and n >= 0.7 * min(len(a), len(b))


STARRED_RE = re.compile(r"((?:[\w’'-]+[ ,]+){0,5}[\w’'-]+)\*")


def find_defined_terms(text: str, index: TermIndex) -> list[dict]:
    """Termes suivis d'un astérisque, reliés au lexique quand c'est possible."""
    found = {}
    for match in STARRED_RE.finditer(text.replace("\n", " ")):
        words = [w.strip("-") for w in re.split(r"[ ,]+", match.group(1))]
        words = [w for w in words if w]
        form, term = index.lookup(words)
        found.setdefault((form.lower(), term), {"forme": form, "terme": term})
    return list(found.values())


def add_references(item: dict, known_articles: set[str], known_clauses: set[str],
                   index: TermIndex, self_article: str | None, self_clause: str | None = None) -> dict:
    """Ajoute renvois_articles, renvois_clauses et termes_definis à `item` (et à ses
    sous-parties), à partir de son titre et de son texte."""
    def annotate(target: dict, text: str) -> None:
        target["renvois_articles"] = find_article_refs(text, known_articles, self_article)
        target["renvois_clauses"] = find_clause_refs(text, known_clauses, self_clause)
        target["termes_definis"] = find_defined_terms(text, index)

    annotate(item, "\n".join(filter(None, [item.get("titre"), item["texte"]])))
    for sub in item.get("sous_parties", []):
        annotate(sub, sub["texte"])
    return item


# --- Script -------------------------------------------------------------------

def _load_articles(articles_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(articles_dir.glob("*.json"))]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def enrich_all(articles: list[dict], clauses_article: str, lexicon_article: str) -> tuple:
    articles = [clean_article(a) for a in articles]
    by_numero = {a["numero"]: a for a in articles}

    clauses = find_clauses(by_numero[clauses_article])
    definitions = parse_lexicon(by_numero[lexicon_article])

    known_articles = {a["numero"] for a in articles if a["numero"]}
    known_clauses = {c["code"] for c in clauses}
    index = TermIndex(definitions)

    for article in articles:
        article["sous_parties"] = split_subparts(article["lignes"])
        if article["numero"] == clauses_article:
            article["clauses"] = [c["code"] for c in clauses]
        add_references(article, known_articles, known_clauses, index, article["numero"])
    for clause in clauses:
        add_references(clause, known_articles, known_clauses, index, clause["article"], clause["code"])
    return articles, clauses, definitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("articles_dir", type=Path, help="sortie de sinistre_cg.segmentation")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--clauses-article", default="70")
    parser.add_argument("--lexicon-article", default="69")
    args = parser.parse_args()

    articles, clauses, definitions = enrich_all(
        _load_articles(args.articles_dir), args.clauses_article, args.lexicon_article
    )
    articles_out = args.out_dir / "articles_enrichis"
    articles_out.mkdir(parents=True, exist_ok=True)
    for old in articles_out.glob("*.json"):
        old.unlink()
    for article in articles:
        _write(articles_out / _file_name(article), article)
    _write(args.out_dir / "clauses.json", clauses)
    _write(args.out_dir / "definitions.json", definitions)
    print(f"{len(articles)} articles, {len(clauses)} clauses, {len(definitions)} définitions -> {args.out_dir}")


if __name__ == "__main__":
    main()
