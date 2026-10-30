"""Tests de la vérification des citations (sinistre_cg/citations.py), sur l'article 7 du contrat."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sinistre_cg.citations import (  # noqa: E402
    NOT_VERIFIED, ContractIndex, match_extract, normalize, verify_citation, verify_result,
)

ARTICLE_7 = json.loads((ROOT / "data" / "processed" / "articles_enrichis" / "article_007.json")
                       .read_text(encoding="utf-8"))
TEXT = ARTICLE_7["texte"]
EXCLUSION = ("Nous ne garantissons pas, outre les exclusions énoncées à l’article 41, "
             "les vols, destructions ou détériorations")
PASSAGES = [{"id": "P1", "section": "article 7", "libelle": "Article 7", "pages": ARTICLE_7["pages"],
             "texte": TEXT}]


def thirty_words() -> list[str]:
    start = TEXT.index("B - les disparitions")
    return TEXT[start:].split()[:30]


def test_normalisation_garde_les_accents():
    assert normalize("  Dé-\ngâts  « causés » par l’EAU • ") == 'dégâts " causés " par l\'eau'


def test_apostrophe_droite_exacte():
    assert "’" in EXCLUSION and EXCLUSION in TEXT
    assert match_extract(EXCLUSION.replace("’", "'"), TEXT)["statut"] == "exacte"


def test_retour_a_la_ligne_et_cesure_exacte():
    words = thirty_words()[:15]
    words[3] = words[3][:3] + "-\n" + words[3][3:]  # « disparitions » -> « dis-\nparitions »
    extract = " ".join(words[:8]) + "\n" + " ".join(words[8:])
    assert match_extract(extract, TEXT)["statut"] == "exacte"


def test_un_mot_change_sur_trente_approchee():
    words = thirty_words()
    words[12] = "maisons"
    result = match_extract(" ".join(words), TEXT)
    assert result["statut"] == "approchee"
    assert 92 <= result["score"] < 100


def test_extrait_tronque():
    result = match_extract("Nous ne garantissons pas… les vols, destructions ou détériorations", TEXT)
    assert result["statut"] == "tronquee"
    assert len(result["spans"]) == 2


def test_extrait_tronque_morceau_absent_introuvable():
    extract = "Nous ne garantissons pas… les dégâts causés par les extraterrestres en goguette"
    assert match_extract(extract, TEXT)["statut"] == "introuvable"


def test_phrase_inventee_introuvable():
    extract = "Les dommages causés par les licornes sont garantis sans franchise ni plafond"
    assert match_extract(extract, TEXT)["statut"] == "introuvable"


def test_passage_inconnu():
    by_id = {p["id"]: p for p in PASSAGES}
    assert verify_citation({"passage": "P99", "extrait": EXCLUSION}, by_id)["statut"] == "passage_inconnu"


def test_citation_valide_texte_et_lignes():
    by_id = {p["id"]: p for p in PASSAGES}
    result = verify_citation({"passage": "P1", "extrait": EXCLUSION.replace("’", "'")}, by_id, ContractIndex())
    assert result["texte_contrat"] == EXCLUSION
    assert result["page"] in ARTICLE_7["pages"]
    assert result["lignes"] and {"x0", "y0", "x1", "y1"} <= set(result["lignes"][0])


def _result(verdict: str, extract: str) -> dict:
    return {
        "verdict": verdict,
        "raisonnement": [{"etape": "Le vol est exclu.", "citations": [{"passage": "P1", "extrait": extract}]}],
        "conditions_a_verifier": [],
    }


def test_couvert_sans_citation_valide_devient_incertain():
    result = verify_result(_result("couvert", "Les licornes sont garanties en tous lieux et à toute heure"),
                           PASSAGES)
    assert result["verdict"] == "incertain"
    assert result["verdict_llm"] == "couvert"
    assert result["verdict_retrograde"]
    assert result["raisonnement"][0]["non_etayee"]
    assert result["conditions_a_verifier"] == [NOT_VERIFIED]


def test_couvert_sans_aucune_etape_devient_incertain():
    result = verify_result({"verdict": "couvert", "raisonnement": [], "conditions_a_verifier": []}, PASSAGES)
    assert result["verdict"] == "incertain"


@pytest.mark.parametrize("verdict", ["depend", "incertain"])
def test_verdict_non_ferme_jamais_retrograde(verdict):
    result = verify_result(_result(verdict, "Les licornes sont garanties en tous lieux"), PASSAGES)
    assert result["verdict"] == verdict and not result["verdict_retrograde"]


def test_citation_valide_garde_le_verdict_et_reverification_stable():
    result = verify_result(_result("non_couvert", EXCLUSION), PASSAGES)
    assert result["verdict"] == "non_couvert" and not result["raisonnement"][0]["non_etayee"]
    assert result["citations_verifiees"] == {"valides": 1, "total": 1, "par_statut": {"exacte": 1}}
    # Rejouer la vérification sur un verdict rétrogradé repart du verdict du LLM.
    downgraded = verify_result(_result("couvert", "Les licornes sont garanties partout"), PASSAGES)
    downgraded["raisonnement"][0]["citations"][0]["extrait"] = EXCLUSION
    again = verify_result(downgraded, PASSAGES)
    assert again["verdict"] == "couvert" and again["conditions_a_verifier"] == []


@pytest.mark.parametrize("extract", [f"… {EXCLUSION}", f"{EXCLUSION}...", f"[…] {EXCLUSION} (...)"])
def test_points_de_suspension_au_debut_ou_a_la_fin_exacte(extract):
    assert match_extract(extract, TEXT)["statut"] == "exacte"


def test_mauvais_passage():
    other = {"id": "P2", "section": "article 41", "libelle": "Article 41", "pages": [30],
             "texte": "Les dommages causés par les licornes ne sont jamais garantis."}
    by_id = {p["id"]: p for p in PASSAGES + [other]}
    result = verify_citation({"passage": "P2", "extrait": EXCLUSION}, by_id)
    assert result["statut"] == "mauvais_passage"
    assert result["passage"] == "P1" and result["passage_origine"] == "P2"
    assert result["section"] == "article 7" and result["texte_contrat"] == EXCLUSION
    # Valide pour le verdict, et la vérification rejouée repart du passage cité.
    res = verify_result(_result("non_couvert", EXCLUSION), [other] + PASSAGES)
    res["raisonnement"][0]["citations"][0]["passage"] = "P2"
    res["raisonnement"][0]["citations"][0]["section"] = "article 41"
    res = verify_result(res, [other] + PASSAGES)
    citation = res["raisonnement"][0]["citations"][0]
    assert res["verdict"] == "non_couvert" and citation["statut"] == "mauvais_passage"
    citation["extrait"] = "Les dommages causés par les licornes ne sont jamais garantis."
    citation = verify_result(res, [other] + PASSAGES)["raisonnement"][0]["citations"][0]
    assert citation["statut"] == "exacte" and citation["passage"] == "P2" and citation["section"] == "article 41"
