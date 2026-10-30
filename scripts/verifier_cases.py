"""Vérifie eval/cases.json : format, références existantes, citations présentes
mot pour mot dans le contrat. « verdict » est un verdict ou une liste non vide de verdicts
acceptables, le principal en premier (c'est lui qui est compté).

Usage :
    python scripts/verifier_cases.py eval/cases.json data/processed
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

VERDICTS = {"couvert", "non_couvert", "depend", "incertain"}
FIELDS = {"id", "claim", "verdict", "articles", "clauses", "citations", "note"}

cases_path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/cases.json")
out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/processed")

cases = json.loads(cases_path.read_text(encoding="utf-8"))
articles = {
    a["numero"]: a for a in (json.loads(p.read_text(encoding="utf-8"))
                             for p in (out / "articles_enrichis").glob("*.json")) if a["numero"]
}
clauses = {c["code"]: c for c in json.loads((out / "clauses.json").read_text(encoding="utf-8"))}
definitions = json.loads((out / "definitions.json").read_text(encoding="utf-8"))


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def source_text(source: str, ref: str) -> str | None:
    if source == "article" and ref in articles:
        return articles[ref]["texte"]
    if source == "clause" and ref in clauses:
        return clauses[ref]["titre"] + "\n" + clauses[ref]["texte"]
    if source == "definition" and ref in definitions:
        return definitions[ref]["definition"]
    return None


def accepted(case: dict) -> list:
    return case["verdict"] if isinstance(case["verdict"], list) else [case["verdict"]]


def primary(case: dict):
    return accepted(case)[0] if accepted(case) else None


errors = []
for case in cases:
    cid = case.get("id")
    if set(case) != FIELDS:
        errors.append(f"{cid}: champs {sorted(set(case) ^ FIELDS)}")
    if not accepted(case):
        errors.append(f"{cid}: liste de verdicts vide")
    errors += [f"{cid}: verdict inconnu {v}" for v in accepted(case) if v not in VERDICTS]
    errors += [f"{cid}: article {a} inexistant" for a in case["articles"] if a not in articles]
    errors += [f"{cid}: clause {c} inexistante" for c in case["clauses"] if c not in clauses]
    if "incertain" not in accepted(case) and not case["citations"]:
        errors.append(f"{cid}: aucune citation")
    for cit in case["citations"]:
        text = source_text(cit["source"], cit["ref"])
        if text is None:
            errors.append(f"{cid}: source introuvable {cit['source']} {cit['ref']}")
        elif norm(cit["texte"]) not in norm(text):
            errors.append(f"{cid}: citation absente de {cit['source']} {cit['ref']} : « {cit['texte'][:60]}… »")
        cited = case["articles"] if cit["source"] == "article" else case["clauses"] if cit["source"] == "clause" else None
        if cited is not None and cit["ref"] not in cited:
            errors.append(f"{cid}: {cit['source']} {cit['ref']} cité mais pas listé")

ids = Counter(c["id"] for c in cases)
errors += [f"id en double : {i}" for i, n in ids.items() if n > 1]

verdicts = Counter(primary(c) for c in cases)
print(f"{len(cases)} cas — verdicts : {dict(verdicts)}")
print(f"{sum(len(c['citations']) for c in cases)} citations vérifiées dans le texte du contrat")
if len(cases) != 20:
    errors.append(f"{len(cases)} cas au lieu de 20")
if verdicts["non_couvert"] < 6:
    errors.append("moins de 6 cas non_couvert")
for e in errors:
    print("ÉCHEC", e)
print("OK" if not errors else f"{len(errors)} erreur(s)")
sys.exit(1 if errors else 0)
