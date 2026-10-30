"""Interface Streamlit : verdict de couverture d'un sinistre, avec citations vérifiées
et surlignées dans le PDF du contrat.

Usage :
    streamlit run app.py
"""

import html
import json
from pathlib import Path

import pymupdf
import streamlit as st

from sinistre_cg.citations import VALID, default_index
from sinistre_cg.llm import LLM, LLMError, endpoint_config
from sinistre_cg.retrieval import CLAUSES_ARTICLE, DATA_DIR, FACT_FIELDS, Retriever
from sinistre_cg.verdict import OPTION_CHAPTERS, STATUS_MARKS, decide, format_pages

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "eval" / "cases.json"
GITHUB_URL = "https://github.com/rabieessayeh/sinistre-cg"
# Exemples proposés : identifiants de eval/cases.json.
EXAMPLES = {
    "Vélo volé dans le local commun": "velo_local_commun",
    "Bougie fondue sur la table du jardin": "bougie_table_jardin",
    "Fuite du lave-linge": "degat_eaux_lave_linge",
    "Morsure d'un chien de 2e catégorie": "rc_chien_dangereux",
    "Vol sans effraction": "cambriolage_sans_effraction",
}
VERDICT_STYLES = {
    "couvert": ("COUVERT", "#1e7d32"),
    "non_couvert": ("NON COUVERT", "#c62828"),
    "depend": ("ÇA DÉPEND", "#e67e00"),
    "incertain": ("INCERTAIN", "#6b7280"),
}
STATUS_LABELS = {
    "exacte": "citation exacte",
    "approchee": "citation approchée",
    "tronquee": "citation tronquée",
    "mauvais_passage": "citation retrouvée dans un autre passage",
    "introuvable": "citation introuvable dans le contrat",
    "passage_inconnu": "passage inconnu",
}
STEPS = {
    "analyse": "Analyse de la déclaration",
    "recherche": "Recherche des passages du contrat",
    "verdict": "Rédaction du verdict",
    "verification": "Vérification des citations",
}
HIGHLIGHT = (1, 0.92, 0.2)
RENDER_ZOOM = 2


# --- Données (chargées une fois) ----------------------------------------------

@st.cache_resource(show_spinner="Chargement du contrat…")
def load_retriever() -> Retriever:
    default_index()  # index des citations, utilisé par decide()
    return Retriever()


@st.cache_resource
def contract_pdf() -> Path:
    first = next((DATA_DIR / "articles_enrichis").glob("*.json"))
    return DATA_DIR.parent / "raw" / json.loads(first.read_text(encoding="utf-8"))["source"]


@st.cache_data
def example_claims() -> dict[str, str]:
    cases = {c["id"]: c["claim"] for c in json.loads(CASES_PATH.read_text(encoding="utf-8"))}
    return {label: cases[cid] for label, cid in EXAMPLES.items() if cid in cases}


def subscribable(retriever: Retriever) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """(clé d'option, libellé) des garanties en option et des clauses, au format de --options."""
    options = [(f"art{key.split()[1]}", s.label) for key, s in retriever.sections.items()
               if key.startswith("article ") and s.chapter in OPTION_CHAPTERS]
    clauses = [(f"cl{key.split()[1]}", s.label) for key, s in retriever.sections.items()
               if key.startswith("clause ")]
    return options, clauses


@st.cache_data(show_spinner=False)
def render_page(pdf: str, page: int, boxes: tuple[tuple[float, float, float, float], ...]) -> bytes:
    """Page du PDF en PNG, avec les rectangles `boxes` surlignés en jaune."""
    with pymupdf.open(pdf) as doc:
        p = doc[page - 1]
        for x0, y0, x1, y1 in boxes:
            p.draw_rect(pymupdf.Rect(x0, y0, x1, y1), color=None, fill=HIGHLIGHT, fill_opacity=0.45,
                        overlay=True)
        return p.get_pixmap(matrix=pymupdf.Matrix(RENDER_ZOOM, RENDER_ZOOM)).tobytes("png")


# --- Appel de la chaîne ---------------------------------------------------------

class ProgressLLM:
    """Transmet les appels au LLM et signale le début et la fin de chaque étape."""

    def __init__(self, llm: LLM, on_step):
        self.llm, self.on_step = llm, on_step

    def chat_json(self, system: str, user: str, kind: str = "analyse") -> dict:
        self.on_step(kind)
        try:
            return self.llm.chat_json(system, user, kind=kind)
        finally:
            self.on_step("recherche" if kind == "analyse" else "verification")


def llm_error_message(exc: Exception) -> str:
    text = str(exc)
    if "non configuré" in text:
        return ("Le modèle de langage n'est pas configuré : renseignez LLM_BASE_URL, LLM_API_KEY et "
                "LLM_MODEL dans le fichier .env.")
    if "413" in text:
        return ("La demande est trop volumineuse pour la limite par minute du fournisseur du modèle. "
                "Essayez une description plus courte, ou configurez un modèle plus généreux "
                "(variables LLM_VERDICT_* du fichier .env).")
    if "401" in text or "403" in text:
        return "Le fournisseur du modèle refuse l'accès : vérifiez la clé API et le modèle dans le fichier .env."
    if "429" in text or "RateLimit" in text or "rate limit" in text.lower():
        return "Le quota du fournisseur du modèle est atteint. Réessayez dans une minute."
    if "Connection" in text or "connexion" in text or "timed out" in text.lower():
        return "Le fournisseur du modèle est injoignable. Vérifiez la connexion réseau et réessayez."
    return "L'appel au modèle de langage a échoué. Réessayez dans quelques instants."


def run_analysis(claim: str, options: list[str] | None) -> None:
    retriever = load_retriever()
    for key in [k for k in st.session_state if str(k).startswith("cit-")]:
        del st.session_state[key]  # pages ouvertes sur le résultat précédent
    done = []
    with st.status(STEPS["analyse"] + "…", expanded=True) as status:
        lines = st.empty()

        def on_step(step: str) -> None:
            if step not in done:
                done.append(step)
            status.update(label=STEPS[step] + "…")
            lines.markdown("\n".join(
                f"{'✅' if i < len(done) - 1 else '⏳'} {STEPS[s]}" for i, s in enumerate(done)
            ))

        try:
            llm = LLM()
            result = decide(claim, ProgressLLM(llm, on_step), retriever, options=options)
        except LLMError as exc:
            status.update(label="Échec de l'analyse", state="error")
            st.session_state.update(result=None, error=llm_error_message(exc))
            return
        except Exception:  # aucune trace Python dans l'interface
            status.update(label="Échec de l'analyse", state="error")
            st.session_state.update(result=None, error="Une erreur inattendue est survenue pendant l'analyse.")
            return
        lines.markdown("\n".join(f"✅ {label}" for label in STEPS.values()))
        status.update(label="Analyse terminée", state="complete", expanded=False)
    result["tokens"] = dict(llm.usage)
    result["modele_verdict"] = endpoint_config("verdict")["model"]
    st.session_state.update(result=result, error=None)


# --- Affichage -------------------------------------------------------------------

def show_badge(result: dict) -> None:
    label, color = VERDICT_STYLES[result["verdict"]]
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin:0.5rem 0 1rem">'
        f'<span style="background:{color};color:#fff;font-weight:700;font-size:1.6rem;'
        f'padding:0.35rem 1.1rem;border-radius:0.5rem;letter-spacing:0.03em">{label}</span>'
        f'<span style="font-size:1rem;opacity:0.8">Confiance : {html.escape(result["confiance"])}</span></div>',
        unsafe_allow_html=True,
    )
    if result.get("verdict_retrograde"):
        st.caption(f"Le modèle avait conclu « {VERDICT_STYLES[result['verdict_llm']][0]} », mais aucune de ses "
                   "citations n'a pu être vérifiée dans le contrat.")


def show_citation(citation: dict, key: str) -> None:
    status = citation.get("statut", "introuvable")
    mark = STATUS_MARKS.get(status, "✗")
    where = (f"{citation['libelle']} · p. {format_pages([citation['page']] if citation.get('page') else citation['pages'])}"
             if citation.get("section") else f"passage {citation.get('passage') or '?'} inconnu")
    details = [STATUS_LABELS.get(status, status)]
    if citation.get("score") and "approchee" in (status, citation.get("correspondance")):
        details.append(f"score {citation['score']:g}")
    if citation.get("passage_origine"):
        details.append(f"cité comme {citation['passage_origine']}")
    st.markdown(
        f'<div style="margin:0.3rem 0 0.2rem 1.2rem"><span title="{html.escape(details[0])}">{mark}</span> '
        f'<b>{html.escape(where)}</b> <span style="opacity:0.65;font-size:0.85rem">({html.escape(", ".join(details))})'
        f'</span><br><i>« {html.escape(citation.get("extrait", ""))} »</i></div>',
        unsafe_allow_html=True,
    )
    if status in VALID and citation.get("lignes"):
        if st.toggle("Voir dans le contrat", key=key):
            by_page = {}
            for box in citation["lignes"]:
                by_page.setdefault(box["page"], []).append((box["x0"], box["y0"], box["x1"], box["y1"]))
            for page, boxes in by_page.items():
                st.image(render_page(str(contract_pdf()), page, tuple(boxes)),
                         caption=f"{citation['libelle']} — page {page}")


def show_reasoning(result: dict) -> None:
    st.subheader("Raisonnement")
    if not result["raisonnement"]:
        st.write("Le modèle n'a détaillé aucune étape.")
    for n, step in enumerate(result["raisonnement"], 1):
        text = html.escape(step["etape"])
        if step.get("non_etayee"):
            st.markdown(f'<div style="color:#8a8f98;margin-top:0.8rem"><b>{n}.</b> {text} '
                        f'<i>(non étayée)</i></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="margin-top:0.8rem"><b>{n}.</b> {text}</div>', unsafe_allow_html=True)
        for m, citation in enumerate(step["citations"]):
            show_citation(citation, key=f"cit-{n}-{m}")


def show_details(result: dict) -> None:
    with st.expander("Détails techniques"):
        if result.get("erreur_llm_analyse"):
            st.warning("L'analyse de la déclaration a échoué : la recherche s'est faite sans elle.")
        analysis = result.get("analyse")
        st.markdown("**Faits extraits**")
        if analysis:
            st.table({"Fait": list(FACT_FIELDS), "Valeur": [analysis["faits"].get(f) or "—" for f in FACT_FIELDS]})
        else:
            st.write("—")
        st.markdown("**Sections retrouvées**")
        st.markdown("\n".join(
            f"- `{p['id']}` {p['libelle']} (p. {format_pages(p['pages'])}) — {p['role']}" for p in result["passages"]
        ))
        checked = result["citations_verifiees"]
        st.markdown(f"**Citations vérifiées** : {checked['valides']}/{checked['total']}  "
                    + "  ".join(f"{STATUS_MARKS.get(s, '?')} {s} : {n}" for s, n in checked["par_statut"].items()))
        tokens = result.get("tokens") or {}
        st.markdown(f"**Tokens consommés** : {tokens.get('total_tokens', 0)} "
                    f"({tokens.get('prompt_tokens', 0)} en entrée, {tokens.get('completion_tokens', 0)} en sortie ; "
                    f"{tokens.get('calls', 0)} appels, {tokens.get('cached_calls', 0)} réponses en cache) — "
                    f"modèle du verdict : `{result.get('modele_verdict', '?')}`")


def show_result(result: dict) -> None:
    show_badge(result)
    if result["resume"]:
        st.markdown(f'<p style="font-size:1.25rem;line-height:1.5">{html.escape(result["resume"])}</p>',
                    unsafe_allow_html=True)
    if result["verdict"] in ("depend", "incertain"):
        with st.container(border=True):
            st.markdown("**Ce qui changerait la réponse**")
            st.markdown("\n".join(f"- {c}" for c in result["conditions_a_verifier"]) or "—")
            st.markdown("**Questions à poser à l'assuré**")
            st.markdown("\n".join(f"- {q}" for q in result["questions_a_poser"]) or "—")
    show_reasoning(result)
    if result["impact_indemnisation"]:
        st.subheader("Impact sur l'indemnisation")
        st.write(result["impact_indemnisation"])
    show_details(result)


# --- Page --------------------------------------------------------------------------

def sidebar(retriever: Retriever) -> list[str] | None:
    """Options souscrites : None si les Conditions particulières sont inconnues."""
    with st.sidebar:
        st.header("À propos")
        st.markdown(
            "Le modèle extrait les faits de la déclaration, puis une recherche retrouve les articles, "
            "clauses et définitions utiles du contrat.\n\n"
            "Il rend ensuite un verdict appuyé uniquement sur ces passages, et chaque citation est "
            "vérifiée mot pour mot dans le contrat.\n\n"
            f"Contrat : *{contract_pdf().stem.replace('-', ' ').capitalize()}*."
        )
        st.header("Options souscrites")
        known = st.checkbox("Conditions particulières connues", help=(
            "Décoché : on ignore quelles options, packs et clauses l'assuré a souscrits."))
        selected = None
        if known:
            options, clauses = subscribable(retriever)
            selected = []
            with st.expander("Options et packs (chapitre 6)", expanded=True):
                selected += [k for k, label in options if st.checkbox(label, key=f"opt-{k}")]
            with st.expander(f"Clauses (article {CLAUSES_ARTICLE})"):
                selected += [k for k, label in clauses if st.checkbox(label, key=f"opt-{k}")]
            st.caption(f"{len(selected)} sélectionnée(s) ; les autres sont considérées comme non souscrites.")
        st.divider()
        st.markdown(f"[Code source sur GitHub]({GITHUB_URL})")
    return selected


def main() -> None:
    st.set_page_config(page_title="Sinistre couvert ?", page_icon="🏠", layout="centered")
    retriever = load_retriever()
    options = sidebar(retriever)

    st.title("Ce sinistre est-il couvert ?")
    st.caption("Décrivez le sinistre : le verdict s'appuie uniquement sur les conditions générales du contrat.")

    st.markdown("**Exemples**")
    examples = example_claims()
    columns = st.columns(len(examples))
    for column, (label, claim) in zip(columns, examples.items()):
        column.button(label, on_click=st.session_state.update, kwargs={"claim": claim}, width="stretch")

    claim = st.text_area("Description du sinistre", key="claim", height=130,
                         placeholder="Ex. : le tuyau de mon lave-linge a lâché et l'eau a abîmé le parquet.")
    if st.button("Analyser", type="primary", disabled=not claim.strip()):
        run_analysis(claim.strip(), options)

    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    elif st.session_state.get("result"):
        st.divider()
        show_result(st.session_state["result"])


main()
