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

Configuration du LLM (API compatible OpenAI) : copier `.env.example` en `.env` et renseigner
au moins `LLM_BASE_URL`, `LLM_API_KEY` et `LLM_MODEL`. Les variables `LLM_ANALYSE_*` et
`LLM_VERDICT_*` permettent un autre fournisseur ou modèle par étape.

Les réponses du LLM sont mises en cache dans `.cache/llm/` ; `LLM_NO_CACHE=1` désactive le cache.

## Interface

```bash
streamlit run app.py
```

## Commandes

Point d'entrée unique : `python -m sinistre_cg <commande>` (`-h` pour les options).

```bash
# Préparation du contrat (PDF -> data/processed/)
python -m sinistre_cg parse data/raw/<cg>.pdf      # extraction, segmentation, enrichissement
python -m sinistre_cg map                          # carte du contrat (LLM)

# Utilisation
python -m sinistre_cg llm                          # test de connexion au LLM
python -m sinistre_cg retrieve "description" [--llm]
python -m sinistre_cg verdict "description du sinistre" [--options art33 clL03] [--json]

# Évaluation (python eval/run_eval.py accepte les mêmes options)
python -m sinistre_cg eval [--llm]                 # rappel de la recherche
python -m sinistre_cg eval --mode full [--only id1,id2] [--pause 60]
python -m sinistre_cg eval --reverify              # revérifie les citations, sans LLM
pytest
```

Les étapes de préparation restent disponibles séparément :
`python -m sinistre_cg.extraction`, `.segmentation`, `.enrich`.
