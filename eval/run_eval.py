"""Point d'entrée : python eval/run_eval.py [options], équivalent à python -m sinistre_cg eval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sinistre_cg.evaluation import main  # noqa: E402

if __name__ == "__main__":
    main()
