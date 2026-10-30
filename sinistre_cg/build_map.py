"""Carte du contrat : un résumé d'une ligne par section (article ou clause), écrit par le LLM.

La carte est sauvegardée après chaque section ; une relance reprend là où elle s'est arrêtée.

Usage :
    python -m sinistre_cg map [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

from sinistre_cg.llm import LLM, LLMError
from sinistre_cg.retrieval import DATA_DIR, load_sections

MAP_PATH = DATA_DIR / "contract_map.json"
MAX_SECTION_CHARS = 6000
MAX_WORDS = 35

SYSTEM = (
    "Tu résumes une section des conditions générales d'un contrat d'assurance habitation. "
    f"Réponds par UNE seule ligne de {MAX_WORDS} mots maximum : ce que la section couvre ou exclut "
    "et sa condition principale, en style télégraphique, avec le vocabulaire du contrat. "
    "N'invente rien, n'ajoute ni titre, ni guillemets, ni puce."
)


def clean_summary(text: str) -> str:
    """Première ligne non vide, ramenée à MAX_WORDS mots."""
    line = next((l.strip(" -•*\"'") for l in text.splitlines() if l.strip()), "")
    words = line.split()
    return " ".join(words[:MAX_WORDS])


def load_map(path: Path = MAP_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_map(contract_map: dict, path: Path = MAP_PATH) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(contract_map, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def build_map(llm: LLM, path: Path = MAP_PATH) -> dict:
    contract_map = load_map(path)
    sections = load_sections()
    todo = [s for key, s in sections.items() if key not in contract_map]
    print(f"{len(sections) - len(todo)} sections déjà résumées, {len(todo)} à faire")
    for n, s in enumerate(todo, 1):
        user = f"{s.label}\n\n{s.text[:MAX_SECTION_CHARS]}"
        summary = clean_summary(llm.chat(SYSTEM, user))
        contract_map[s.key] = {"titre": s.title, "resume": summary}
        save_map(contract_map, path)
        print(f"[{n}/{len(todo)}] {s.key} : {summary}")
    return contract_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Construit data/processed/contract_map.json.")
    parser.add_argument("--verbose", action="store_true", help="affiche les tokens de chaque appel")
    args = parser.parse_args()
    try:
        llm = LLM(verbose=args.verbose)
        build_map(llm)
    except LLMError as exc:
        sys.exit(f"Erreur LLM : {exc} (la carte partielle est conservée, relancer pour reprendre)")
    print(f"Tokens consommés : {llm.usage['total_tokens']} -> {MAP_PATH}")


if __name__ == "__main__":
    main()
