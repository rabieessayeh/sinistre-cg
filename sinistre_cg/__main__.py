"""Point d'entrée unique : python -m sinistre_cg <commande> [arguments].

Commandes :
    parse     PDF des conditions générales -> données (extraction, segmentation, enrichissement)
    map       carte du contrat : un résumé par section (LLM)
    llm       test de connexion au LLM
    retrieve  passages du contrat pour une déclaration
    verdict   verdict de couverture d'une déclaration
    eval      évaluation de la recherche ou du verdict sur eval/cases.json

`python -m sinistre_cg <commande> -h` affiche les options de chaque commande.
"""

import argparse
import importlib
import sys
from pathlib import Path

from sinistre_cg.retrieval import DATA_DIR

# commande -> (module dont on appelle main(), description)
COMMANDS = {
    "parse": (None, "PDF des conditions générales -> données"),
    "map": ("sinistre_cg.build_map", "carte du contrat (LLM)"),
    "llm": ("sinistre_cg.llm", "test de connexion au LLM"),
    "retrieve": ("sinistre_cg.retrieval", "passages du contrat pour une déclaration"),
    "verdict": ("sinistre_cg.verdict", "verdict de couverture d'une déclaration"),
    "eval": ("sinistre_cg.evaluation", "évaluation sur eval/cases.json"),
}
PROG = "python -m sinistre_cg"


def _run(module: str, command: str, argv: list) -> None:
    """Appelle main() de `module` comme si la ligne de commande était `command argv`."""
    sys.argv = [f"{PROG} {command}"] + [str(a) for a in argv]
    # Sous `python -m`, argparse ignore sys.argv[0] pour le nom du programme : sans __spec__,
    # l'aide affiche « python -m sinistre_cg <commande> ».
    sys.modules["__main__"].__spec__ = None
    importlib.import_module(module).main()


def parse(argv: list[str]) -> None:
    """Enchaîne extraction, segmentation et enrichissement avec les chemins habituels."""
    parser = argparse.ArgumentParser(prog=f"{PROG} parse", description=COMMANDS["parse"][1])
    parser.add_argument("pdf", type=Path, help="conditions générales (PDF)")
    parser.add_argument("--out", type=Path, default=DATA_DIR, help=f"dossier de sortie (défaut {DATA_DIR})")
    args = parser.parse_args(argv)
    lines, articles = args.out / "cg_lines.json", args.out / "articles"
    _run("sinistre_cg.extraction", "parse", [args.pdf, lines])
    _run("sinistre_cg.segmentation", "parse", [lines, articles])
    _run("sinistre_cg.enrich", "parse", [articles, args.out])


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        width = max(map(len, COMMANDS))
        usage = "\n".join(f"  {name:<{width}}  {desc}" for name, (_, desc) in COMMANDS.items())
        print(f"Usage : {PROG} <commande> [arguments]\n\nCommandes :\n{usage}")
        sys.exit(0 if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") else 2)
    command, argv = sys.argv[1], sys.argv[2:]
    if command == "parse":
        parse(argv)
    else:
        _run(COMMANDS[command][0], command, argv)


if __name__ == "__main__":
    main()
