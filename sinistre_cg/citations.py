"""Vérification des citations d'un verdict dans le texte du contrat (sans LLM).

Chaque extrait cité est comparé au passage qu'il cite, après normalisation (apostrophes,
tirets, espaces, césures, puces, casse ; les accents sont gardés). Les « … » ou « ... » placés
au début ou à la fin de l'extrait sont retirés. Statuts :
    exacte          l'extrait normalisé est une sous-chaîne du passage normalisé
    tronquee        l'extrait contient « … » ou « ... » au milieu et chaque morceau d'au moins
                    MIN_PIECE_WORDS mots est retrouvé dans le passage, dans l'ordre
    approchee       fuzz.partial_ratio >= FUZZY_THRESHOLD
    mauvais_passage introuvable dans le passage cité, mais trouvé (exact ou approché) dans un
                    autre passage envoyé : la citation est rattachée à ce passage
                    (« passage_origine » garde l'identifiant cité)
    introuvable     aucun des cas ci-dessus
    passage_inconnu l'identifiant cité ne figure pas parmi les passages envoyés
Une citation valide reçoit le texte réel du contrat et, si possible, la page et les
coordonnées des lignes correspondantes (pour surligner le passage dans le PDF).

Une étape sans citation valide est « non étayée » ; un verdict ferme (couvert ou
non_couvert) dont aucune étape n'est étayée devient « incertain ».
"""

import bisect
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

from sinistre_cg.retrieval import CLAUSES_ARTICLE, DATA_DIR, article_key, load_contract

FUZZY_THRESHOLD = 92
MIN_PIECE_WORDS = 4
VALID = ("exacte", "approchee", "tronquee", "mauvais_passage")
STATUSES = ("exacte", "approchee", "tronquee", "mauvais_passage", "introuvable", "passage_inconnu")
FIRM_VERDICTS = ("couvert", "non_couvert")
NOT_VERIFIED = "Aucune citation n'a pu être vérifiée dans le contrat."
CITATION_FIELDS = ("statut", "score", "correspondance", "texte_contrat", "page", "lignes")

# --- Normalisation ------------------------------------------------------------

CHAR_MAP = {
    **dict.fromkeys("’‘‛ʼ′`´", "'"),
    **dict.fromkeys("“”„‟«»″", '"'),
    **dict.fromkeys("‐‑‒–—―−", "-"),
}
_DASHES = "-‐‑‒–—―−­"
# Césure en fin de ligne (lettre-tiret-saut de ligne-lettre), puces et tiret conditionnel.
_DROP = re.compile(
    rf"(?<=[^\W\d_])[{_DASHES}][ \t ]*\n\s*(?=[^\W\d_])"
    r"|[•\u0003­-]"
)
_ELLIPSIS = re.compile(r"\s*[\[(]?\s*(?:…|\.{3,})\s*[\])]?\s*")
_EDGE_ELLIPSIS = re.compile(rf"^(?:{_ELLIPSIS.pattern})+|(?:{_ELLIPSIS.pattern})+$")


def normalize_map(text: str) -> tuple[str, list[int]]:
    """Texte normalisé et, pour chacun de ses caractères, sa position dans `text`."""
    dropped = bytearray(len(text))
    for m in _DROP.finditer(text):
        dropped[m.start():m.end()] = b"\1" * (m.end() - m.start())
    out, index = [], []
    for i, ch in enumerate(text):
        if dropped[i]:
            continue
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                index.append(i)
            continue
        for c in CHAR_MAP.get(ch, ch).lower():
            out.append(c)
            index.append(i)
    if out and out[-1] == " ":
        out.pop()
        index.pop()
    return "".join(out), index


def normalize(text: str) -> str:
    return normalize_map(text)[0]


# --- Recherche d'un extrait ---------------------------------------------------

def _snap(text: str, start: int, end: int) -> tuple[int, int]:
    """Recale un intervalle approché sur la limite de mot la plus proche, à chaque bout."""
    if 0 < start < len(text) and text[start - 1] != " ":
        back = text.rfind(" ", 0, start) + 1
        forward = text.find(" ", start)
        forward = len(text) if forward < 0 else forward + 1
        start = back if start - back <= forward - start else forward
    if 0 < end < len(text) and text[end] != " ":
        back = text.rfind(" ", 0, end)
        forward = text.find(" ", end)
        forward = len(text) if forward < 0 else forward
        end = forward if forward - end <= end - back else back
    return start, end


def _find(needle: str, haystack: str, start: int = 0) -> tuple[int, int, float] | None:
    """(début, fin, score) de `needle` dans `haystack[start:]` : exact, sinon approché
    (bornes recalées sur des limites de mots)."""
    pos = haystack.find(needle, start)
    if pos >= 0:
        return pos, pos + len(needle), 100.0
    if not needle or start >= len(haystack):
        return None
    al = fuzz.partial_ratio_alignment(needle, haystack[start:])
    if al is None or al.score < FUZZY_THRESHOLD:
        return None
    begin, end = _snap(haystack, start + al.dest_start, start + al.dest_end)
    if begin >= end:
        begin, end = start + al.dest_start, start + al.dest_end
    return begin, end, al.score


def match_extract(extract: str, text: str) -> dict:
    """Statut de `extract` dans `text`, avec les intervalles trouvés (positions dans `text`)."""
    extract = _EDGE_ELLIPSIS.sub("", extract)
    norm_text, index = normalize_map(text)
    norm_extract = normalize(extract)
    if not norm_extract:
        return {"statut": "introuvable"}

    def raw(spans):
        return [(index[s], index[e - 1] + 1) for s, e in spans]

    pos = norm_text.find(norm_extract)
    if pos >= 0:
        return {"statut": "exacte", "spans": raw([(pos, pos + len(norm_extract))])}
    if _ELLIPSIS.search(extract):
        pieces = [normalize(p) for p in _ELLIPSIS.split(extract)]
        pieces = [p for p in pieces if len(p.split()) >= MIN_PIECE_WORDS]
        spans, scores, start = [], [], 0
        for piece in pieces:
            found = _find(piece, norm_text, start)
            if found is None:
                return {"statut": "introuvable"}
            spans.append(found[:2])
            scores.append(found[2])
            start = found[1]
        if not pieces:
            return {"statut": "introuvable"}
        return {"statut": "tronquee", "score": round(min(scores), 1), "spans": raw(spans)}
    found = _find(norm_extract, norm_text)
    if found is None:
        return {"statut": "introuvable"}
    return {"statut": "approchee", "score": round(found[2], 1), "spans": raw([found[:2]])}


# --- Textes et lignes du contrat ----------------------------------------------

def _box(line: dict) -> dict:
    return {k: line[k] for k in ("page", "x0", "y0", "x1", "y1")}


class ContractIndex:
    """Texte et lignes (avec coordonnées) des sections et des définitions du contrat."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.sections, self.definitions = load_contract(data_dir)
        self._article_lines = {}
        for path in (data_dir / "articles_enrichis").glob("*.json"):
            a = json.loads(path.read_text(encoding="utf-8"))
            if a["numero"] is not None:
                self._article_lines[article_key(a["numero"])] = a["lignes"]
        lines = json.loads((data_dir / "cg_lines.json").read_text(encoding="utf-8"))["lines"]
        self._page_lines = {}
        for line in lines:
            self._page_lines.setdefault(line["page"], []).append(line)

    def passage_text(self, passage: dict) -> str | None:
        if passage["section"] == "lexique":
            d = self.definitions.get(passage["libelle"])
            return d["definition"] if d else None
        s = self.sections.get(passage["section"])
        return s.text if s else None

    def passage_lines(self, passage: dict) -> list[dict]:
        """Lignes où chercher le passage : celles de l'article ; pour une clause, celles de
        l'article des clauses sur ses pages ; pour une définition, celles de sa page et de la suivante."""
        section, pages = passage["section"], set(passage["pages"])
        if section == "lexique":
            pages |= {p + 1 for p in pages}
            return [line for p in sorted(pages) for line in self._page_lines.get(p, [])]
        if section in self._article_lines:
            return self._article_lines[section]
        return [line for line in self._article_lines.get(article_key(CLAUSES_ARTICLE), [])
                if line["page"] in pages]

    def locate(self, passage: dict, segment: str) -> list[dict]:
        """Coordonnées des lignes qui contiennent `segment` (texte brut du contrat)."""
        lines = self.passage_lines(passage)
        if not lines:
            return []
        starts, offset = [], 0
        for line in lines:
            starts.append(offset)
            offset += len(line["text"]) + 1
        norm_lines, index = normalize_map("\n".join(line["text"] for line in lines))
        found = _find(normalize(segment), norm_lines)
        if found is None:
            return []
        first = bisect.bisect_right(starts, index[found[0]]) - 1
        last = bisect.bisect_right(starts, index[found[1] - 1]) - 1
        return [_box(line) for line in lines[first:last + 1]]


@lru_cache(maxsize=1)
def default_index() -> ContractIndex:
    return ContractIndex()


# --- Vérification d'un verdict ------------------------------------------------

def _text(passage: dict, index: ContractIndex | None) -> str | None:
    return passage.get("texte") or (index.passage_text(passage) if index else None)


def _elsewhere(extract: str, cited: str, passages: dict[str, dict],
               index: ContractIndex | None) -> tuple[dict, dict] | None:
    """Meilleure correspondance exacte ou approchée de `extract` dans un passage autre que
    `cited` (exacte d'abord, sinon le meilleur score) ; None s'il n'y en a pas. Les extraits
    tronqués ne sont pas cherchés ailleurs : leurs morceaux, souvent des formules types,
    se retrouvent trop facilement dans d'autres clauses."""
    best = None
    for pid, passage in passages.items():
        text = _text(passage, index)
        if pid == cited or not text:
            continue
        match = match_extract(extract, text)
        if match["statut"] not in ("exacte", "approchee"):
            continue
        rank = (match["statut"] == "exacte", match.get("score", 100.0))
        if best is None or rank > best[0]:
            best = (rank, passage, match)
    return best[1:] if best else None


def verify_citation(citation: dict, passages: dict[str, dict], index: ContractIndex | None = None) -> dict:
    """Champs de vérification d'une citation {"passage", "extrait"}. Pour un mauvais passage,
    renvoie aussi le passage corrigé (« passage », « section », « libelle », « pages »)."""
    passage = passages.get(citation.get("passage"))
    if passage is None:
        return {"statut": "passage_inconnu"}
    text = _text(passage, index)
    if not text:
        return {"statut": "passage_inconnu"}
    extract = citation.get("extrait") or ""
    match = match_extract(extract, text)
    if match["statut"] == "introuvable":
        found = _elsewhere(extract, passage["id"], passages, index)
        if found is not None:
            passage, other = found
            text = _text(passage, index)
            match = {"statut": "mauvais_passage", "correspondance": other["statut"], "spans": other["spans"],
                     "passage_origine": citation["passage"], "passage": passage["id"],
                     "section": passage["section"], "libelle": passage["libelle"], "pages": passage["pages"]}
            if "score" in other:
                match["score"] = other["score"]
    spans = match.pop("spans", None)
    if not spans:
        return match
    segments = [text[s:e] for s, e in spans]
    match["texte_contrat"] = " […] ".join(segments)
    if index is not None:
        boxes = [box for seg in segments for box in index.locate(passage, seg)]
        if boxes:
            match["page"] = boxes[0]["page"]
            match["lignes"] = boxes
    return match


def verify_result(result: dict, passages: list[dict], index: ContractIndex | None = None) -> dict:
    """Vérifie les citations de `result` (modifié et renvoyé). Rejouable : repart de
    « verdict_llm » s'il existe, pour refaire la vérification sur un résultat enregistré."""
    by_id = {p["id"]: p for p in passages}
    counts = Counter()
    for step in result["raisonnement"]:
        for citation in step["citations"]:
            for key in CITATION_FIELDS:
                citation.pop(key, None)
            if "passage_origine" in citation:  # vérification rejouée : on repart du passage cité
                citation["passage"] = citation.pop("passage_origine")
                cited = by_id.get(citation["passage"])
                citation.update({"section": cited["section"], "libelle": cited["libelle"],
                                 "pages": cited["pages"]} if cited else
                                {"section": None, "libelle": None, "pages": []})
            citation.update(verify_citation(citation, by_id, index))
            counts[citation["statut"]] += 1
        step["non_etayee"] = not any(c["statut"] in VALID for c in step["citations"])

    verdict_llm = result.get("verdict_llm", result["verdict"])
    supported = any(not step["non_etayee"] for step in result["raisonnement"])
    downgraded = verdict_llm in FIRM_VERDICTS and not supported
    conditions = [c for c in result["conditions_a_verifier"] if c != NOT_VERIFIED]
    result.update({
        "verdict": "incertain" if downgraded else verdict_llm,
        "verdict_llm": verdict_llm,
        "verdict_retrograde": downgraded,
        "conditions_a_verifier": conditions + [NOT_VERIFIED] if downgraded else conditions,
        "citations_verifiees": {
            "valides": sum(counts[s] for s in VALID),
            "total": sum(counts.values()),
            "par_statut": dict(counts),
        },
    })
    return result
