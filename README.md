# sinistre-cg

Dit si un sinistre est couvert par les conditions générales d'un contrat d'assurance
habitation, en citant les articles, et sait répondre « ça dépend » ou « je ne sais pas ».

Chaîne : analyse de la déclaration par un LLM, recherche des passages du contrat (BM25 +
renvois + définitions), verdict du LLM appuyé uniquement sur ces passages, puis vérification
des citations mot pour mot dans le texte du contrat.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Fichier `.env` (API compatible OpenAI) :

```
LLM_BASE_URL=...
LLM_API_KEY=...
LLM_MODEL=...
# Optionnel : autre fournisseur ou modèle pour le verdict
LLM_VERDICT_BASE_URL=...
LLM_VERDICT_API_KEY=...
LLM_VERDICT_MODEL=...
```

Les réponses du LLM sont mises en cache dans `.cache/llm/` ; `LLM_NO_CACHE=1` désactive le cache.

## Interface

```bash
streamlit run app.py
```

## Commandes

```bash
# Préparation du contrat (PDF -> données)
python -m sinistre_cg.extraction data/raw/<cg>.pdf data/processed/cg_lines.json
python -m sinistre_cg.segmentation data/processed/cg_lines.json data/processed/articles
python -m sinistre_cg.enrich data/processed/articles data/processed
python build_map.py                       # carte du contrat (LLM)

# Utilisation
python llm.py                             # test de connexion au LLM
python retrieval.py "description" [--llm]
python verdict.py "description du sinistre" [--options art33 clL03] [--json]

# Évaluation
python eval/run_eval.py [--llm]           # rappel de la recherche
python eval/run_eval.py --mode full [--only id1,id2] [--pause 60]
python eval/run_eval.py --reverify        # revérifie les citations, sans LLM
pytest
```
