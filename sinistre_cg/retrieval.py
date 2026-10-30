"""Recherche des passages du contrat à fournir au LLM pour une déclaration de sinistre.

Index BM25 sur les sous-parties, regroupement par section (article ou clause), puis
expansion : renvois cités, exclusions communes, étendue territoriale, définitions du lexique.
Si un LLM est configuré, une analyse préalable de la déclaration ajoute des sections
et des requêtes dans le vocabulaire du contrat.

Usage :
    python retrieval.py "description du sinistre" [--llm] [--map-size N]
"""

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import snowballstemmer
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TOP_SECTIONS = 6
MAX_DEFINITIONS = 15
MAX_CHARS = 45_000
TITLE_WEIGHT = 3
# Chapitres administratifs (déclaration du risque, vie du contrat, cotisations) : pas des garanties.
EXCLUDED_CHAPTERS = {12, 13, 14}
# Article des exclusions communes à tous les risques, toujours fourni.
COMMON_EXCLUSIONS_ARTICLE = "41"
# Sections toujours fournies, repérées par un fragment de leur titre.
ALWAYS_INCLUDED_TITLES = ("étendue territoriale",)
# Article qui regroupe les clauses : il est remplacé par une section par clause.
CLAUSES_ARTICLE = "70"
# Termes du lexique trop génériques pour être ajoutés automatiquement.
GENERIC_TERMS = {
    "Vous", "Assuré", "Sinistre", "Société", "Sociétaire", "Indemnité", "Franchise", "Cotisation",
    "Événement", "Échéance", "Résiliation", "Préavis", "Tiers", "Accident", "Indice",
    "Année d'assurance", "Année révolue",
}
# Sous-parties plus courtes ignorées dans le score d'une section (biais de BM25), sauf si
# la section n'en a qu'une.
MIN_SUBPART_CHARS = 200

# Carte du contrat (résumés par section, voir build_map.py) et analyse par LLM.
MAP_PATH = DATA_DIR / "contract_map.json"
MAP_FALLBACK_CHARS = 160
# Carte envoyée à l'analyse : les MAP_SIZE sections les mieux classées par BM25 pour la
# déclaration, plus toujours celles des chapitres MAP_ALWAYS_CHAPTERS (responsabilité civile, options).
MAP_SIZE = 30
MAP_ALWAYS_CHAPTERS = {3, 6}
LLM_MIN_SECTIONS, LLM_MAX_SECTIONS = 3, 10
LLM_MAX_QUERIES = 6
FACT_FIELDS = ("evenement", "biens_endommages", "lieu", "circonstances", "personnes",
               "informations_manquantes")
ANALYSIS_SYSTEM = f"""Tu analyses une déclaration de sinistre pour retrouver les sections utiles des \
conditions générales d'un contrat d'assurance habitation. Tu reçois un extrait de la carte du contrat \
(une ligne par section : clé | titre | résumé) puis la déclaration.

Consignes :
- pense aux garanties qui pourraient couvrir le sinistre ET aux exclusions, conditions, options et \
clauses qui pourraient s'appliquer ;
- n'invente rien : les faits viennent uniquement de la déclaration ; ce qui n'y figure pas va dans \
"informations_manquantes" ;
- ne choisis que des clés présentes dans la carte, recopiées exactement ;
- les requêtes sont courtes et utilisent le vocabulaire du contrat.

Réponds uniquement avec ce JSON :
{{
  "faits": {{{", ".join(f'"{f}": "..."' for f in FACT_FIELDS)}}},
  "sections": ["{LLM_MIN_SECTIONS} à {LLM_MAX_SECTIONS} clés de la carte"],
  "requetes": ["4 à {LLM_MAX_QUERIES} requêtes courtes"]
}}"""

FRENCH_STOPWORDS = set("""
a afin ai aie aient aies ait alors as au aucun aucune aupres auquel aura aurai auraient aurais
aurait auras aurez auriez aurions aurons auront aussi autre autres aux auxquelles auxquels avaient
avais avait avant avec avez aviez avions avoir avons ayant ayez ayons c ca ce ceci cela celle celles
celui ces cet cette ceux chacun chacune chaque ci comme comment d dans de des desquelles desquels
deja donc dont du duquel elle elles en encore entre es est et etaient etais etait etant ete etes
etiez etions etre eu eue eues eumes eurent eus eusse eussent eusses eussiez eussions eut eutes eux
fait faites fut fumes furent fus fusse fussent fusses fussiez fussions fut futes ici il ils j je
jusqu jusque l la laquelle le lequel les lesquelles lesquels leur leurs lors lorsque lui m ma mais
me meme memes mes moi mon n ne ni nos notre nous on ont ou par parce pas peu peut plus pour pourquoi
qu quand que quel quelle quelles quels qui quoi s sa sans se sera serai seraient serais serait
seras serez seriez serions serons seront ses si soi soient sois soit sommes son sont soyez soyons
suis sur t ta te tes toi ton tous tout toute toutes tu un une vos votre vous y
apres assez beaucoup bien chez contre depuis derriere dessous dessus devant deux dont hors jamais
loin moins non oui parmi pendant pres puis quelque quelques rien selon sous souvent tant tard tot
tres trop vers voici voila
""".split())

_stemmer = snowballstemmer.stemmer("french")


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def _quote_key(text: str) -> str:
    return text.replace("’", "'").lower()


def tokenize(text: str) -> list[str]:
    """Minuscules, mots vides retirés, racinisation, accents retirés."""
    words = re.findall(r"\w+", text.lower())
    words = [w for w in words if _strip_accents(w) not in FRENCH_STOPWORDS and not w.isdigit()]
    return [_strip_accents(s) for s in _stemmer.stemWords(words)]


# --- Sections -----------------------------------------------------------------

@dataclass
class Section:
    key: str            # « article 7 », « clause DOG »
    label: str
    pages: list[int]
    text: str
    title: str
    subparts: list[dict]
    article_refs: list[str]
    clause_refs: list[str]
    starred_terms: list[str]
    searchable: bool
    chapter: int | None = None
    tokens: list[str] = field(default_factory=list)


def article_key(numero: str) -> str:
    return f"article {numero}"


def clause_key(code: str) -> str:
    return f"clause {code}"


def _starred(item: dict) -> list[str]:
    return list(dict.fromkeys(t["terme"] for t in item["termes_definis"] if t["terme"]))


def load_sections(data_dir: Path = DATA_DIR) -> dict[str, Section]:
    sections = {}
    for path in sorted((data_dir / "articles_enrichis").glob("*.json")):
        a = json.loads(path.read_text(encoding="utf-8"))
        if a["numero"] is None or a["numero"] == CLAUSES_ARTICLE:
            continue
        chapter = (a["chapitre"] or {}).get("numero")
        title = a["titre"] or ""
        sections[article_key(a["numero"])] = Section(
            key=article_key(a["numero"]),
            label=f"Article {a['numero']} — {title}",
            pages=a["pages"], text=a["texte"], title=title, subparts=a["sous_parties"],
            article_refs=a["renvois_articles"], clause_refs=a["renvois_clauses"],
            starred_terms=_starred(a),
            searchable=chapter is not None and chapter not in EXCLUDED_CHAPTERS,
            chapter=chapter,
        )
    for c in json.loads((data_dir / "clauses.json").read_text(encoding="utf-8")):
        sections[clause_key(c["code"])] = Section(
            key=clause_key(c["code"]),
            label=f"Clause {c['code']} (art. {c['article']}) — {c['titre']}",
            pages=c["pages"], text=c["texte"], title=c["titre"], subparts=c["sous_parties"],
            article_refs=c["renvois_articles"], clause_refs=c["renvois_clauses"],
            starred_terms=_starred(c), searchable=True,
        )
    for s in sections.values():
        s.tokens = tokenize(s.text)
    return sections


def load_definitions(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    return json.loads((data_dir / "definitions.json").read_text(encoding="utf-8"))


def load_contract_map(sections: dict[str, Section], path: Path = MAP_PATH) -> dict[str, str]:
    """Résumé d'une ligne par section : carte du LLM si elle existe, sinon titre et début
    de la première sous-partie « garantie » (ou de la première sous-partie)."""
    stored = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    summaries = {}
    for key, s in sections.items():
        if key in stored and stored[key].get("resume"):
            summaries[key] = stored[key]["resume"]
            continue
        sub = next((x for x in s.subparts if x["type"] == "garantie"), s.subparts[0] if s.subparts else None)
        excerpt = re.sub(r"\s+", " ", sub["texte"])[:MAP_FALLBACK_CHARS] if sub else ""
        summaries[key] = excerpt
    return summaries


def _as_text(value) -> str:
    if isinstance(value, list):
        return " ; ".join(str(v) for v in value if v)
    return str(value or "").strip()


# --- Index --------------------------------------------------------------------

class Retriever:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.sections = load_sections(data_dir)
        self.definitions = load_definitions(data_dir)
        self._doc_sections = []
        self._doc_counts = []  # la sous-partie compte-t-elle dans le score de sa section ?
        corpus = []
        for s in self.sections.values():
            if not s.searchable:
                continue
            for sub in s.subparts:
                corpus.append(tokenize(" ".join([s.title] * TITLE_WEIGHT + [sub["texte"]])))
                self._doc_sections.append(s.key)
                self._doc_counts.append(len(s.subparts) == 1 or len(sub["texte"]) >= MIN_SUBPART_CHARS)
        self.bm25 = BM25Okapi(corpus)
        # Termes du lexique, sous forme de suites de racines (parenthèses ignorées).
        self._term_tokens = {
            term: tokenize(re.sub(r"\([^)]*\)", " ", term)) for term in self.definitions
        }
        generic = {_quote_key(t) for t in GENERIC_TERMS}
        self._generic = {t for t in self.definitions if _quote_key(t) in generic}
        self.contract_map = load_contract_map(self.sections)
        wanted = [_strip_accents(t).lower() for t in ALWAYS_INCLUDED_TITLES]
        self._always_by_title = [
            key for key, s in self.sections.items()
            if s.searchable and any(w in _strip_accents(s.title).lower() for w in wanted)
        ]

    # 2. Recherche
    def search(self, queries: str | list[str], k: int = TOP_SECTIONS) -> list[tuple[str, float]]:
        """Meilleur score par section, sur toutes les requêtes."""
        best = {}
        for query in [queries] if isinstance(queries, str) else queries:
            scores = self.bm25.get_scores(tokenize(query))
            for key, counts, score in zip(self._doc_sections, self._doc_counts, scores):
                if counts:
                    best[key] = max(score, best.get(key, float("-inf")))
        ranked = sorted(((key, s) for key, s in best.items() if s > 0), key=lambda kv: -kv[1])
        return ranked[:k]

    # 3. Expansion
    def references(self, keys: list[str]) -> list[str]:
        """Sections citées par `keys` (un saut), dans l'ordre des sections citantes.
        Un renvoi à l'article des clauses n'ajoute que les clauses nommées."""
        found = []
        for key in keys:
            s = self.sections[key]
            cited = [article_key(n) for n in s.article_refs if n != CLAUSES_ARTICLE]
            cited += [clause_key(c) for c in s.clause_refs]
            found += [c for c in cited if c in self.sections and self.sections[c].searchable]
        return [k for k in dict.fromkeys(found) if k not in keys]

    def _terms_in(self, tokens: list[str]) -> list[str]:
        present = []
        for term, seq in self._term_tokens.items():
            n = len(seq)
            if n and any(tokens[i:i + n] == seq for i in range(len(tokens) - n + 1)):
                present.append(term)
        return present

    def definitions_for(self, claim: str, principal: list[str], renvois: list[str],
                        limit: int = MAX_DEFINITIONS) -> list[str]:
        """Termes du lexique par priorité : dans la déclaration > étoilé dans une section
        principale > présent dans une section principale > étoilé dans un renvoi.
        À priorité égale, le terme qui apparaît dans le plus de sections passe devant.
        Les termes génériques ne sont jamais ajoutés."""
        # Chaque palier est une liste de « sources » (déclaration ou section) et des termes qu'elles contiennent.
        tiers = [
            [self._terms_in(tokenize(claim))],
            [self.sections[k].starred_terms for k in principal],
            [self._terms_in(self.sections[k].tokens) for k in principal],
            [self.sections[k].starred_terms for k in renvois],
        ]
        best_tier, count, first_seen = {}, {}, {}
        for rank, sources in enumerate(tiers):
            for terms in sources:
                for term in dict.fromkeys(terms):
                    if term not in self.definitions or term in self._generic:
                        continue
                    best_tier.setdefault(term, rank)
                    first_seen.setdefault(term, len(first_seen))
                    count[term] = count.get(term, 0) + 1
        ordered = sorted(best_tier, key=lambda t: (best_tier[t], -count[t], first_seen[t]))
        return ordered[:limit]

    # Analyse par LLM
    def map_text(self, claim: str, map_size: int = MAP_SIZE) -> str:
        """Extrait de la carte : une ligne « clé | titre | résumé » pour les `map_size` sections
        les mieux classées par BM25 pour `claim` et pour celles des chapitres MAP_ALWAYS_CHAPTERS,
        dans l'ordre du contrat."""
        wanted = {key for key, _ in self.search(claim, k=map_size)}
        wanted |= {key for key, s in self.sections.items() if s.searchable and s.chapter in MAP_ALWAYS_CHAPTERS}
        return "\n".join(
            " ".join(f"{key} | {s.title} | {self.contract_map.get(key, '')}".split())
            for key, s in self.sections.items() if key in wanted
        )

    def analysis_message(self, claim: str, map_size: int = MAP_SIZE) -> str:
        return f"CARTE DU CONTRAT\n{self.map_text(claim, map_size)}\n\nDÉCLARATION\n{claim}"

    def analyze(self, claim: str, llm, map_size: int = MAP_SIZE) -> dict:
        """Faits, sections et requêtes proposés par le LLM (clés inconnues ignorées)."""
        raw = llm.chat_json(ANALYSIS_SYSTEM, self.analysis_message(claim, map_size), kind="analyse")
        facts = raw.get("faits") if isinstance(raw.get("faits"), dict) else {}
        proposed = [str(k).strip() for k in raw.get("sections") or [] if isinstance(k, (str, int))]
        valid = [k for k in dict.fromkeys(proposed) if k in self.sections and self.sections[k].searchable]
        queries = [str(q).strip() for q in raw.get("requetes") or [] if str(q).strip()]
        return {
            "faits": {f: _as_text(facts.get(f)) for f in FACT_FIELDS},
            "sections": valid[:LLM_MAX_SECTIONS],
            "sections_ignorees": [k for k in proposed if k not in valid],
            "requetes": queries[:LLM_MAX_QUERIES],
        }

    # 4. Sortie
    def retrieve(self, claim: str, max_chars: int = MAX_CHARS, llm=None, map_size: int = MAP_SIZE) -> dict:
        """Passages pour `claim`. Avec `llm`, l'analyse préalable ajoute des sections
        principales et des requêtes ; en cas d'échec, on continue avec BM25 seul."""
        analysis, llm_error = None, None
        if llm is not None:
            try:
                analysis = self.analyze(claim, llm, map_size)
            except Exception as exc:  # l'analyse est un bonus : la recherche continue sans elle
                llm_error = f"{type(exc).__name__}: {exc}"

        queries = [claim] + (analysis["requetes"] if analysis else [])
        ranked = self.search(queries)
        scores = dict(self.search(queries, k=len(self.sections)))
        bm25_keys = [key for key, _ in ranked]
        llm_keys = analysis["sections"] if analysis else []
        principal = list(dict.fromkeys(bm25_keys + llm_keys))
        # Texte servant à choisir les définitions : déclaration + faits extraits.
        claim_text = " ".join([claim] + (list(analysis["faits"].values()) if analysis else []))

        always_keys = [article_key(COMMON_EXCLUSIONS_ARTICLE)] + self._always_by_title
        always = [k for k in dict.fromkeys(always_keys) if k in self.sections and k not in principal]
        candidates = [k for k in self.references(principal) if k not in always]

        def size(keys):
            return sum(len(self.sections[k].text) for k in keys)

        # Les sections principales et l'article des exclusions communes ne sont jamais
        # retirés ; on réserve la place des définitions déjà connues avant les renvois.
        budget = max_chars - size(principal + always) - sum(
            len(self.definitions[t]["definition"]) for t in self.definitions_for(claim_text, principal, [])
        )
        renvois = list(always)
        for key in candidates:
            if len(self.sections[key].text) <= budget:
                renvois.append(key)
                budget -= len(self.sections[key].text)

        used = size(principal + renvois)
        terms = []
        for term in self.definitions_for(claim_text, principal, renvois):
            length = len(self.definitions[term]["definition"])
            if used + length <= max_chars:
                terms.append(term)
                used += length

        passages = []
        for n, key in enumerate(principal + renvois, 1):
            s = self.sections[key]
            passage = {
                "id": f"P{n}", "section": key, "libelle": s.label, "pages": s.pages,
                "role": "principal" if key in principal else "renvoi", "texte": s.text,
            }
            if key in principal:
                passage["origine"] = [o for o, keys in (("bm25", bm25_keys), ("llm", llm_keys)) if key in keys]
            if key in scores and key in principal:
                passage["score"] = round(float(scores[key]), 3)
            passages.append(passage)
        for n, term in enumerate(terms, 1):
            d = self.definitions[term]
            passages.append({
                "id": f"D{n}", "section": "lexique", "libelle": term, "pages": [d["page"]],
                "role": "définition", "texte": d["definition"],
            })
        return {"passages": passages, "analyse": analysis, "erreur_llm": llm_error}


def main() -> None:
    parser = argparse.ArgumentParser(description="Passages du contrat pour une déclaration de sinistre.")
    parser.add_argument("claim", help="description du sinistre")
    parser.add_argument("--json", action="store_true", help="sortie JSON complète")
    parser.add_argument("--full", action="store_true", help="affiche le texte complet des passages")
    parser.add_argument("--llm", action="store_true", help="analyse la déclaration par LLM avant la recherche")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE,
                        help=f"sections BM25 dans la carte envoyée à l'analyse (défaut {MAP_SIZE})")
    parser.add_argument("--verbose", action="store_true", help="affiche les tokens consommés")
    args = parser.parse_args()

    llm = None
    if args.llm:
        from sinistre_cg.llm import LLM, LLMError
        try:
            llm = LLM(verbose=args.verbose)
        except LLMError as exc:
            print(f"[LLM indisponible : {exc} — recherche BM25 seule]\n", file=sys.stderr)

    result = Retriever().retrieve(args.claim, llm=llm, map_size=args.map_size)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    if result["erreur_llm"]:
        print(f"[Analyse LLM en échec : {result['erreur_llm']} — recherche BM25 seule]\n")
    if result["analyse"]:
        analysis = result["analyse"]
        print("Faits extraits :")
        for name, value in analysis["faits"].items():
            print(f"  {name:<24} {value or '—'}")
        print(f"Sections choisies par le LLM : {', '.join(analysis['sections']) or '—'}")
        if analysis["sections_ignorees"]:
            print(f"Clés inconnues ignorées : {', '.join(analysis['sections_ignorees'])}")
        print("Requêtes :")
        for q in analysis["requetes"]:
            print(f"  - {q}")
        print()
    passages = result["passages"]
    for p in passages:
        score = f"  score={p['score']}" if "score" in p else ""
        origin = f" ({'+'.join(p['origine'])})" if p.get("origine") else ""
        print(f"{p['id']:>4}  [{p['role']}{origin}] {p['libelle']}  (p. {', '.join(map(str, p['pages']))}; "
              f"{len(p['texte'])} car.){score}")
        if args.full:
            print("      " + p["texte"].replace("\n", "\n      ") + "\n")
    print(f"\n{sum(len(p['texte']) for p in passages)} caractères au total")
    if llm is not None:
        print(f"Tokens consommés : {llm.usage['total_tokens']}")


if __name__ == "__main__":
    main()
