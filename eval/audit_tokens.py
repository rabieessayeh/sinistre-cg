"""Estime la taille des requêtes LLM (analyse et verdict) sans appeler l'API.

Tokens estimés = caractères / 3.5 (français). Groq compte aussi max_completion_tokens
dans la taille d'une requête : une requête est signalée si son prompt dépasse
WARN_TOKENS ou si prompt + max_completion_tokens dépasse la limite par requête.

La requête de verdict est estimée comme déclaration + passages renvoyés par BM25 seul
(sans prompt système, faits ni lignes « Nature ») : c'est une borne basse, car l'analyse
par LLM ajoute jusqu'à LLM_MAX_SECTIONS sections.

Usage :
    python eval/audit_tokens.py [--n 3] [--map-size N] [--cases eval/cases.json]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinistre_cg.llm import CHARS_PER_TOKEN, completion_settings  # noqa: E402
from sinistre_cg.retrieval import ANALYSIS_SYSTEM, MAP_SIZE, Retriever  # noqa: E402

WARN_TOKENS = 7000
REQUEST_LIMIT = 8000


def tokens(text: str) -> int:
    return round(len(text) / CHARS_PER_TOKEN)


def passages_text(passages: list[dict]) -> str:
    return "\n\n".join(
        f"[{p['id']}] {p['libelle']} (p. {', '.join(map(str, p['pages']))})\n{p['texte']}" for p in passages
    )


def report(name: str, parts: dict[str, str], kind: str) -> bool:
    sizes = {part: tokens(text) for part, text in parts.items()}
    total = sum(sizes.values())
    max_out = completion_settings(kind)["max_completion_tokens"]
    flags = []
    if total > WARN_TOKENS:
        flags.append(f"> {WARN_TOKENS}")
    if total + max_out > REQUEST_LIMIT:
        flags.append(f"prompt + sortie > {REQUEST_LIMIT}")
    detail = "  ".join(f"{part} {n:>6}" for part, n in sizes.items())
    flag = f"  <-- {', '.join(flags)}" if flags else ""
    print(f"  {name:<8} {detail}  | total {total:>6}  max_completion {max_out:>5}  "
          f"= {total + max_out:>6}{flag}")
    return bool(flags)


def main() -> None:
    parser = argparse.ArgumentParser(description="Taille estimée des requêtes LLM, sans appel à l'API.")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.json")
    parser.add_argument("--n", type=int, default=3, help="nombre de cas audités")
    parser.add_argument("--map-size", type=int, default=MAP_SIZE, help="sections BM25 dans la carte")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))[:args.n]
    retriever = Retriever()
    flagged = 0
    for case in cases:
        claim = case["claim"]
        print(f"{case['id']}")
        flagged += report("analyse", {
            "système": ANALYSIS_SYSTEM,
            "carte": retriever.map_text(claim, args.map_size),
            "déclaration": claim,
        }, "analyse")
        passages = retriever.retrieve(claim)["passages"]
        flagged += report("verdict", {
            "système": "",
            "carte": "",
            "déclaration": claim,
            "passages": passages_text(passages),
        }, "verdict")
    print(f"\n{flagged} requête(s) signalée(s) sur {2 * len(cases)}")


if __name__ == "__main__":
    main()
