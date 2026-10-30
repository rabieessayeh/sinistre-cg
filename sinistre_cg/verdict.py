"""Verdict de couverture d'un sinistre, justifié par des citations du contrat.

Chaîne : analyse de la déclaration et recherche des passages (retrieval.py), puis un
appel au LLM (kind="verdict") qui raisonne uniquement sur les passages numérotés, puis
vérification des extraits cités dans le texte du contrat (citations.py) : un verdict ferme
sans aucune étape étayée devient « incertain ».

Usage :
    python -m sinistre_cg verdict "description du sinistre" [--options art33 clL03] [--json]
"""

import argparse
import json
import re
import sys
import textwrap

from sinistre_cg.citations import default_index, verify_result
from sinistre_cg.llm import LLM, LLMError, endpoint_config
from sinistre_cg.retrieval import (
    CLAUSES_ARTICLE, FACT_FIELDS, MAX_CHARS, Retriever, article_key, clause_key,
)

VERDICTS = ("couvert", "non_couvert", "depend", "incertain")
CONFIDENCES = ("haute", "moyenne", "faible")
VERDICT_LABELS = {
    "couvert": "COUVERT", "non_couvert": "NON COUVERT", "depend": "ÇA DÉPEND", "incertain": "INCERTAIN",
}
STATUS_MARKS = {"exacte": "✓", "approchee": "≈", "tronquee": "✂", "mauvais_passage": "↪",
                "introuvable": "✗", "passage_inconnu": "✗"}
# Chapitres des garanties en option (les autres articles s'appliquent à tout contrat).
OPTION_CHAPTERS = {6}
NATURE_BASE = "contrat de base (s'applique à tout contrat)"
NATURE_OPTION = "garantie en option (s'applique seulement si elle est souscrite)"
NATURE_CLAUSE = (f"clause de l'article {CLAUSES_ARTICLE} (s'applique seulement si elle figure aux "
                 "Conditions particulières)")
NATURE_DEFINITION = "définition du lexique"
UNKNOWN_OPTIONS = ("Les Conditions particulières ne sont pas connues : on ignore quelles options, "
                   "packs ou clauses l'assuré a souscrits.")
QUOTE_MIN_WORDS, QUOTE_MAX_WORDS = 10, 40

VERDICT_SYSTEM = f"""Tu es un gestionnaire de sinistres expert en assurance habitation. Tu dis si un \
sinistre est couvert par les conditions générales d'un contrat, en justifiant chaque point par le texte \
du contrat.

Tu reçois la déclaration de l'assuré, les faits extraits de la déclaration, les options souscrites \
(si on les connaît) et des passages numérotés du contrat : articles et clauses [P1], [P2]..., \
définitions du lexique [D1], [D2]... Sous le titre de chaque passage, une ligne « Nature » dit s'il \
s'agit du contrat de base, d'une garantie en option, d'une clause ou d'une définition.

Règles :
1. Appuie-toi EXCLUSIVEMENT sur les passages fournis. N'utilise ni tes connaissances générales sur \
l'assurance ni d'autres contrats. Si les passages ne permettent pas de conclure, dis-le.
2. Chaque étape du raisonnement cite au moins un passage, avec un extrait COPIÉ MOT POUR MOT du \
passage (même ponctuation, même orthographe), de {QUOTE_MIN_WORDS} à {QUOTE_MAX_WORDS} mots. Un \
extrait est un seul morceau CONTIGU du passage : n'écris jamais « … » ni « ... », ne reformule pas, \
ne recolle pas deux morceaux. Pour citer deux endroits, fais deux citations. Indique le numéro du \
passage d'où vient réellement l'extrait.
3. Vérifie dans cet ordre :
   a. le bien endommagé est-il un bien assuré ?
   b. l'événement entre-t-il dans une garantie ?
   c. les conditions de cette garantie sont-elles remplies (lieu, circonstances, mesures de \
prévention, inoccupation des locaux) ?
   d. une exclusion propre à la garantie ou une exclusion commune s'applique-t-elle ?
4. Les définitions du lexique (passages D) priment sur le sens courant des mots.
5. Distingue la garantie de base des options, packs et clauses grâce à la ligne « Nature » de chaque \
passage : un passage du contrat de base s'applique à tout contrat ; une garantie en option seulement si elle \
est souscrite ; une clause de l'article {CLAUSES_ARTICLE} seulement si elle figure aux Conditions \
particulières. Pour une option ou une clause dont on ignore la souscription, détermine d'après son \
texte si elle ÉTEND la garantie (bien, événement, lieu ou responsabilité couverts en plus) ou la \
RESTREINT (exclusion, moyens de protection exigés, limitation) :
   - extension : si la garantie de base couvre déjà le sinistre, elle ne change rien ; sinon, si elle \
couvrirait le sinistre, le verdict est "depend" et tu la nommes dans "conditions_a_verifier" ;
   - restriction : elle ne change pas le verdict ; si elle pouvait jouer, nomme-la dans \
"conditions_a_verifier".
   Ne fais dépendre le verdict que d'options, packs ou clauses présents dans les passages : une \
option supposée, absente des passages, ne rend pas le verdict "depend". Si les options souscrites \
sont connues, celles qui ne figurent pas dans la liste ne sont pas souscrites, et une clause \
souscrite s'applique, qu'elle étende ou restreigne la garantie.
6. Raisonne sur les faits déclarés. Un fait exceptionnel que la déclaration ne mentionne pas (race \
de chien dangereuse, faute intentionnelle, activité professionnelle, non-respect des mesures de \
prévention...) ne rend le verdict ni "depend" ni "incertain" : il devient une question dans \
"questions_a_poser".
7. Si le contrat est ambigu ou si la déclaration est trop vague pour identifier l'événement, dis-le : \
un verdict "incertain" justifié vaut mieux qu'un verdict affirmé sans appui.
8. Mentionne les conséquences sur l'indemnisation prévues par les passages (réduction, franchise, \
plafond, aucune indemnité) dans "impact_indemnisation" ; chaîne vide si les passages n'en disent rien.

Verdicts :
- "couvert" : la garantie de base couvre le sinistre d'après les faits déclarés. Les formalités \
normales (déclarer le sinistre dans les délais, porter plainte, fournir des preuves) et les \
franchises ou plafonds ne changent pas le verdict : elles vont dans "conditions_a_verifier" ou \
"impact_indemnisation" ;
- "non_couvert" : aucune garantie de base ne couvre le sinistre, ou une exclusion s'applique d'après \
les faits déclarés ;
- "depend" : SEULEMENT si la réponse change selon une option, un pack ou une clause d'extension dont \
on ignore la souscription (règle 5), OU selon un fait que la déclaration laisse explicitement \
ouvert ; nomme cette option ou ce fait ;
- "incertain" : le contrat est ambigu, ou la déclaration est trop vague pour identifier l'événement.

Réponds uniquement avec ce JSON :
{{
  "verdict": "couvert" | "non_couvert" | "depend" | "incertain",
  "resume": "1 à 3 phrases en langage clair pour un assuré",
  "raisonnement": [
    {{"etape": "affirmation", "citations": [{{"passage": "P3", "extrait": "texte exact"}}]}}
  ],
  "conditions_a_verifier": ["options, clauses ou faits qui changeraient le verdict ; formalités à accomplir"],
  "questions_a_poser": ["questions à poser à l'assuré"],
  "impact_indemnisation": "texte ou chaîne vide",
  "confiance": "haute" | "moyenne" | "faible"
}}"""


# --- Options souscrites -------------------------------------------------------

def option_section(key: str) -> str | None:
    """Clé de section pour une option : « art33 » -> « article 33 », « clL03 » -> « clause L03 »."""
    match = re.fullmatch(r"(art|cl)\s*(\w+)", key.strip(), re.IGNORECASE)
    if not match:
        return None
    prefix, ref = match.group(1).lower(), match.group(2)
    return article_key(ref) if prefix == "art" else clause_key(ref.upper())


def describe_options(options: list[str] | None, retriever: Retriever) -> str:
    if options is None:
        return UNKNOWN_OPTIONS
    if not options:
        return "Aucune option, aucun pack ni aucune clause n'est souscrit."
    lines = []
    for key in options:
        section = retriever.sections.get(option_section(key) or "")
        lines.append(f"- {key} : {section.label}" if section else f"- {key}")
    return "\n".join(lines)


def unknown_options(options: list[str] | None, retriever: Retriever) -> list[str]:
    return [k for k in options or [] if option_section(k) not in retriever.sections]


# --- Message utilisateur ------------------------------------------------------

def format_pages(pages: list[int]) -> str:
    return ", ".join(map(str, pages))


def passage_nature(passage: dict, retriever: Retriever) -> str:
    if passage["section"] == "lexique":
        return NATURE_DEFINITION
    section = retriever.sections.get(passage["section"])
    if section is None or section.key.startswith("clause "):
        return NATURE_CLAUSE
    return NATURE_OPTION if section.chapter in OPTION_CHAPTERS else NATURE_BASE


def format_passages(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['id']}] {p['libelle']} (p. {format_pages(p['pages'])})\nNature : {p['nature']}\n{p['texte']}"
        for p in passages
    )


def format_facts(analysis: dict | None) -> str:
    if not analysis:
        return "Analyse indisponible."
    return "\n".join(f"- {name} : {analysis['faits'].get(name) or '—'}" for name in FACT_FIELDS)


def verdict_message(claim: str, analysis: dict | None, options_text: str, passages: list[dict]) -> str:
    return "\n\n".join([
        f"DÉCLARATION\n{claim}",
        f"FAITS EXTRAITS\n{format_facts(analysis)}",
        f"OPTIONS SOUSCRITES\n{options_text}",
        f"PASSAGES DU CONTRAT\n{format_passages(passages)}",
    ])


# --- Réponse ------------------------------------------------------------------

def _str_list(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def _citation(raw, passages: dict[str, dict]) -> dict | None:
    if not isinstance(raw, dict):
        return None
    pid = str(raw.get("passage") or "").strip().strip("[]")
    citation = {"passage": pid, "extrait": str(raw.get("extrait") or "").strip()}
    p = passages.get(pid)
    citation.update({"section": p["section"], "libelle": p["libelle"], "pages": p["pages"]} if p else
                    {"section": None, "libelle": None, "pages": []})
    return citation


def normalize(raw: dict, passages: list[dict]) -> dict:
    """Réponse du LLM mise au format attendu ; chaque citation reçoit la section et les pages
    du passage cité (section None si l'identifiant est inconnu)."""
    by_id = {p["id"]: p for p in passages}
    verdict = str(raw.get("verdict") or "").strip().lower()
    confidence = str(raw.get("confiance") or "").strip().lower()
    steps = []
    for step in raw.get("raisonnement") or []:
        if not isinstance(step, dict):
            continue
        citations = [c for c in map(lambda r: _citation(r, by_id), step.get("citations") or []) if c]
        steps.append({"etape": str(step.get("etape") or "").strip(), "citations": citations})
    result = {
        "verdict": verdict if verdict in VERDICTS else "incertain",
        "resume": str(raw.get("resume") or "").strip(),
        "raisonnement": steps,
        "conditions_a_verifier": _str_list(raw.get("conditions_a_verifier")),
        "questions_a_poser": _str_list(raw.get("questions_a_poser")),
        "impact_indemnisation": str(raw.get("impact_indemnisation") or "").strip(),
        "confiance": confidence if confidence in CONFIDENCES else "faible",
    }
    if verdict not in VERDICTS:
        result["verdict_brut"] = raw.get("verdict")
    return result


# --- Chaîne complète ----------------------------------------------------------

def decide(claim: str, llm: LLM, retriever: Retriever | None = None, options: list[str] | None = None,
           max_chars: int = MAX_CHARS) -> dict:
    """Analyse, recherche, verdict puis vérification des citations.
    Lève LLMError si l'appel du verdict échoue."""
    retriever = retriever or Retriever()
    retrieval = retriever.retrieve(claim, max_chars=max_chars, llm=llm)
    passages = [{**p, "nature": passage_nature(p, retriever)} for p in retrieval["passages"]]
    message = verdict_message(claim, retrieval["analyse"], describe_options(options, retriever), passages)
    raw = llm.chat_json(VERDICT_SYSTEM, message, kind="verdict")
    return {
        **verify_result(normalize(raw, passages), passages, default_index()),
        "options": options,
        "options_inconnues": unknown_options(options, retriever),
        "analyse": retrieval["analyse"],
        "erreur_llm_analyse": retrieval["erreur_llm"],
        "passages": [{k: p[k] for k in ("id", "section", "libelle", "pages", "role", "nature")}
                     for p in passages],
    }


# --- Affichage ----------------------------------------------------------------

def _wrap(text: str, indent: str) -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def print_result(result: dict) -> None:
    if result["erreur_llm_analyse"]:
        print(f"[Analyse LLM en échec : {result['erreur_llm_analyse']} — recherche BM25 seule]\n")
    if result["options_inconnues"]:
        print(f"[Options introuvables dans le contrat : {', '.join(result['options_inconnues'])}]\n")
    banner = f"  VERDICT : {VERDICT_LABELS[result['verdict']]}  "
    print("═" * len(banner))
    print(banner)
    print("═" * len(banner))
    if result.get("verdict_retrograde"):
        print(f"(verdict du LLM : {VERDICT_LABELS[result['verdict_llm']]}, rétrogradé faute de citation vérifiée)")
    print(f"Confiance : {result['confiance']}\n")
    if result["resume"]:
        print(_wrap(result["resume"], "") + "\n")

    print("Raisonnement :")
    for n, step in enumerate(result["raisonnement"], 1):
        flag = "  [non étayée]" if step.get("non_etayee") else ""
        print(_wrap(f"{n}. {step['etape']}{flag}", "  "))
        for c in step["citations"]:
            where = (f"{c['libelle']} (p. {format_pages(c['pages'])})" if c["section"]
                     else f"passage inconnu {c['passage'] or '?'}")
            mark = STATUS_MARKS.get(c.get("statut"), "?")
            score = f" {c['score']:g}" if "approchee" in (c.get("statut"), c.get("correspondance")) else ""
            origin = f" (cité comme {c['passage_origine']})" if c.get("passage_origine") else ""
            print(f"   {mark}{score} [{c['passage']}] {where}{origin}")
            print(_wrap(f"« {c['extrait']} »", "       "))
            if c.get("statut") in ("approchee", "tronquee") or c.get("correspondance") in ("approchee", "tronquee"):
                print(_wrap(f"contrat : « {' '.join(c['texte_contrat'].split())} »", "       "))
    if not result["raisonnement"]:
        print("  —")

    for title, items in (("Conditions à vérifier", result["conditions_a_verifier"]),
                         ("Questions à poser", result["questions_a_poser"])):
        print(f"\n{title} :")
        print("\n".join(_wrap(f"- {item}", "  ") for item in items) if items else "  —")
    print("\nImpact sur l'indemnisation :")
    print(_wrap(result["impact_indemnisation"], "  ") if result["impact_indemnisation"] else "  —")
    checked = result["citations_verifiees"]
    print(f"\nCitations vérifiées : {checked['valides']}/{checked['total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verdict de couverture d'un sinistre.")
    parser.add_argument("claim", help="description du sinistre")
    parser.add_argument("--options", nargs="*", metavar="CLE",
                        help="options souscrites (ex. art33 clL03) ; sans valeur : aucune option ; "
                             "absent : Conditions particulières inconnues")
    parser.add_argument("--json", action="store_true", help="sortie JSON complète")
    parser.add_argument("--max-chars", type=int, default=MAX_CHARS,
                        help=f"taille maximale des passages en caractères (défaut {MAX_CHARS})")
    parser.add_argument("--verbose", action="store_true", help="affiche taille des requêtes et tokens")
    args = parser.parse_args()

    try:
        llm = LLM(verbose=args.verbose)
        result = decide(args.claim, llm, options=args.options, max_chars=args.max_chars)
    except LLMError as exc:
        sys.exit(f"Erreur LLM : {exc}")
    result["tokens"] = llm.usage
    result["modele_verdict"] = endpoint_config("verdict")["model"]
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))
        return
    print_result(result)
    u = llm.usage
    print(f"\nModèle du verdict : {result['modele_verdict']} — tokens consommés : {u['total_tokens']} "
          f"({u['calls']} appels, {u['cached_calls']} en cache)")


if __name__ == "__main__":
    main()
