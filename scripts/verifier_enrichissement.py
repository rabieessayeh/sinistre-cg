"""Vérifications des sorties de sinistre_cg.enrich sur les CG Mutuelles de Poitiers.

Usage :
    python scripts/verifier_enrichissement.py data/processed
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/processed")
articles = {
    a["numero"]: a
    for a in (json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "articles_enrichis").glob("*.json")))
}
clauses = {c["code"]: c for c in json.loads((out / "clauses.json").read_text(encoding="utf-8"))}
definitions = json.loads((out / "definitions.json").read_text(encoding="utf-8"))

results = []


def check(label: str, ok: bool, detail="") -> None:
    results.append(ok)
    print(f"[{'OK' if ok else 'ÉCHEC'}] {label}" + (f"  ->  {detail}" if detail else ""))


numbered = [n for n in articles if n is not None]
check("76 articles numérotés", len(numbered) == 76, len(numbered))
check("52 clauses", len(clauses) == 52, len(clauses))
check("environ 83 définitions", 78 <= len(definitions) <= 88, len(definitions))

a7 = articles["7"]
types7 = [s["type"] for s in a7["sous_parties"]]
check("article 7 : au moins 4 sous-parties", len(types7) >= 4, types7)
check(
    "article 7 : une exclusion contient « locaux à usage collectif »",
    any(s["type"] == "exclusion" and "locaux à usage collectif" in s["texte"] for s in a7["sous_parties"]),
)

l03 = clauses.get("L03", {}).get("titre")
check("clause L03 : titre attendu",
      l03 == "Détériorations accidentelles et vol de votre matériel de sport en tous lieux", l03)

eff = definitions.get("Effraction", {}).get("definition", "")
check("définition « Effraction » contient « toutes les issues sont verrouillées »",
      "toutes les issues sont verrouillées" in eff.replace("\n", " "))

r7 = a7["renvois_articles"]
check("article 7 renvoie vers 30, 34, 41, 55", {"30", "34", "41", "55"} <= set(r7), r7)
check("article 7 renvoie vers les clauses 14 et 14A", {"14", "14A"} <= set(a7["renvois_clauses"]),
      a7["renvois_clauses"])
check("article 2 renvoie vers 34", "34" in articles["2"]["renvois_articles"], articles["2"]["renvois_articles"])

leftover = [p.name for p in out.rglob("*.json") if p.parent.name != "articles" and p.name != "cg_lines.json"
            and "\\u0003" in p.read_text(encoding="utf-8")]
check("aucun \\u0003 restant dans les sorties enrichies", not leftover, leftover)

print(f"\n{sum(results)}/{len(results)} vérifications réussies")
sys.exit(0 if all(results) else 1)
