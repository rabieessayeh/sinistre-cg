"""Évalue la recherche de passages ou le verdict sur eval/cases.json.

--mode retrieval (défaut) : pour chaque cas, vérifie si les sections attendues (articles
et clauses) figurent parmi les passages renvoyés, quel que soit leur rôle.
--mode full : analyse, recherche et verdict ; compare le verdict obtenu au(x) verdict(s)
attendu(s) (« verdict » peut être une liste de verdicts acceptables). Pour chaque erreur,
affiche le résumé, les conditions à vérifier et les questions à poser. À la fin : taux de
citations par statut, étapes non étayées et verdicts rétrogradés par la vérification.
--from-results : réaffiche le dernier eval/results_full.json sans appeler l'API, en
recomptant les verdicts corrects d'après les verdicts attendus actuels de --cases.
Avec --only, les nouveaux résultats remplacent les mêmes cas dans eval/results_full.json.
--reverify : comme --from-results, mais refait d'abord la vérification des citations
(sans LLM) et réécrit eval/results_full.json.

Usage :
    python -m sinistre_cg eval [--mode retrieval|full] [--llm] [--pause N] [--only id1,id2]
                               [--from-results | --reverify] [--cases eval/cases.json]
    (ou python eval/run_eval.py avec les mêmes options)

Les résultats vont dans eval/results.json (BM25 seul), eval/results_llm.json (--llm) ou
eval/results_full.json (--mode full) ; en mode retrieval, si l'autre fichier existe, les deux
rappels moyens sont affichés côte à côte.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from sinistre_cg.retrieval import DATA_DIR, Retriever, article_key, clause_key

ROOT = Path(__file__).resolve().parent.parent

RESULTS = {False: ROOT / "eval" / "results.json", True: ROOT / "eval" / "results_llm.json"}
RESULTS_FULL = ROOT / "eval" / "results_full.json"


def contract_size(data_dir: Path = DATA_DIR) -> int:
    """Nombre de caractères du contrat entier (tous les morceaux, clauses comprises)."""
    return sum(
        len(json.loads(p.read_text(encoding="utf-8"))["texte"])
        for p in (data_dir / "articles_enrichis").glob("*.json")
    )


def evaluate_case(retriever: Retriever, case: dict, llm=None) -> dict:
    expected = [article_key(a) for a in case["articles"]] + [clause_key(c) for c in case["clauses"]]
    result = retriever.retrieve(case["claim"], llm=llm)
    passages, analysis = result["passages"], result["analyse"]
    found = {p["section"] for p in passages if p["role"] != "définition"}
    missing = [e for e in expected if e not in found]
    proposed = set(analysis["sections"]) if analysis else set()
    return {
        "id": case["id"],
        "verdict": case["verdict"],
        "attendues": expected,
        "manquantes": missing,
        "manquantes_proposees_par_llm": [m for m in missing if m in proposed],
        "rappel": (len(expected) - len(missing)) / len(expected) if expected else None,
        "contexte_car": sum(len(p["texte"]) for p in passages),
        "analyse": analysis,
        "erreur_llm": result["erreur_llm"],
        "passages": [
            {k: p[k] for k in ("id", "section", "role", "libelle", "origine", "score") if k in p}
            for p in passages
        ],
    }


def _mean_recall(results: list[dict]) -> float:
    scored = [r["rappel"] for r in results if r["rappel"] is not None]
    return sum(scored) / len(scored)


def accepted_verdicts(case: dict) -> list[str]:
    """Verdicts acceptables, le principal en premier."""
    return case["verdict"] if isinstance(case["verdict"], list) else [case["verdict"]]


def evaluate_verdict(retriever: Retriever, case: dict, llm) -> dict:
    from sinistre_cg.llm import LLMError
    from sinistre_cg.verdict import decide

    expected = accepted_verdicts(case)
    try:
        result, error = decide(case["claim"], llm, retriever), None
    except LLMError as exc:
        result, error = None, str(exc)
    obtained = result["verdict"] if result else "erreur"
    return {
        "id": case["id"],
        "attendus": expected,
        "obtenu": obtained,
        "correct": obtained in expected,
        "erreur_llm": error,
        "resultat": result,
    }


def print_matrix(results: list[dict]) -> None:
    from sinistre_cg.verdict import VERDICTS

    rows = [v for v in VERDICTS if any(r["attendus"][0] == v for r in results)]
    cols = list(VERDICTS) + (["erreur"] if any(r["obtenu"] == "erreur" for r in results) else [])
    corner = "attendu \\ obtenu"
    width = max(map(len, cols)) + 2
    print(corner + "".join(c.rjust(width) for c in cols))
    for row in rows:
        counts = [sum(r["attendus"][0] == row and r["obtenu"] == c for r in results) for c in cols]
        print(row.ljust(len(corner)) + "".join(str(n or ".").rjust(width) for n in counts))


def print_verdict_case(r: dict) -> None:
    status = "OK" if r["correct"] else "ERREUR"
    print(f"{status:<6} {r['id']:<28} obtenu {r['obtenu']:<12} attendu {' ou '.join(r['attendus'])}")
    if r["erreur_llm"]:
        print(f"       [LLM en échec : {r['erreur_llm']}]")
    elif not r["correct"]:
        res = r["resultat"]
        print(f"       résumé     : {res['resume'] or '—'}")
        for title, key in (("conditions", "conditions_a_verifier"), ("questions ", "questions_a_poser")):
            items = res[key] or ["—"]
            print(f"       {title} : {items[0]}")
            for item in items[1:]:
                print(f"                    {item}")
        print()
    sys.stdout.flush()


def print_citation_stats(results: list[dict]) -> None:
    from sinistre_cg.citations import STATUSES, VALID

    verified = [r["resultat"] for r in results if r["resultat"] and "citations_verifiees" in r["resultat"]]
    if not verified:
        print("\nCitations : non vérifiées (résultats antérieurs à la vérification, voir --reverify)")
        return
    counts = Counter()
    for res in verified:
        counts.update(res["citations_verifiees"]["par_statut"])
    total = sum(counts.values())
    valid = sum(counts[s] for s in VALID)
    steps = sum(len(res["raisonnement"]) for res in verified)
    unsupported = sum(step["non_etayee"] for res in verified for step in res["raisonnement"])
    print(f"\nCitations valides : {valid}/{total} ({valid / total:.0%})" if total else "\nCitations : aucune")
    for status in STATUSES:
        if total:
            print(f"  {status:<16} {counts[status]:>4}  ({counts[status] / total:.0%})")
    print(f"Étapes non étayées : {unsupported}/{steps}")
    print(f"Verdicts rétrogradés : {sum(res['verdict_retrograde'] for res in verified)}/{len(verified)}")


def print_verdict_summary(results: list[dict], usage: dict) -> None:
    correct = sum(r["correct"] for r in results)
    print(f"\nVerdicts corrects : {correct}/{len(results)}")
    llm_verdicts = [r["resultat"].get("verdict_llm", r["obtenu"]) if r["resultat"] else r["obtenu"]
                    for r in results]
    llm_correct = sum(v in r["attendus"] for v, r in zip(llm_verdicts, results))
    print(f"Verdicts du LLM corrects (avant vérification des citations) : {llm_correct}/{len(results)}\n")
    print_matrix(results)
    print_citation_stats(results)
    print(f"\nTokens consommés : {usage['total_tokens']} ({usage['prompt_tokens']} entrée + "
          f"{usage['completion_tokens']} sortie, {usage['calls']} appels, {usage['cached_calls']} en cache) ; "
          f"échecs LLM : {sum(bool(r['erreur_llm']) for r in results)}")


def show_full_results(cases: list[dict], only: list[str] | None, reverify: bool = False) -> None:
    """Réaffiche eval/results_full.json sans appeler l'API, rescoré avec `cases`. Avec
    `reverify`, refait d'abord la vérification des citations et réécrit le fichier."""
    data = json.loads(RESULTS_FULL.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in cases}
    index = None
    if reverify:
        from sinistre_cg.citations import ContractIndex, verify_result
        index = ContractIndex()
    selected = [r for r in data["cas"] if not only or r["id"] in only]
    for r in selected:
        if index is not None and r["resultat"]:
            verify_result(r["resultat"], r["resultat"]["passages"], index)
            r["obtenu"] = r["resultat"]["verdict"]
        if r["id"] in by_id:
            r["attendus"] = accepted_verdicts(by_id[r["id"]])
            r["correct"] = r["obtenu"] in r["attendus"]
        print_verdict_case(r)
    print_verdict_summary(selected, data["resume"]["tokens"])
    if reverify:
        data["resume"]["verdicts_corrects"] = sum(r["correct"] for r in data["cas"])
        RESULTS_FULL.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Détail -> {RESULTS_FULL.relative_to(ROOT)}")


def run_full(cases: list[dict], pause: float, partial: bool = False) -> None:
    from sinistre_cg.llm import LLM, LLMError
    try:
        llm = LLM()
    except LLMError as exc:
        sys.exit(f"Erreur LLM : {exc}")
    retriever = Retriever()

    results = []
    for n, case in enumerate(cases):
        if n and pause:
            time.sleep(pause)
        r = evaluate_verdict(retriever, case, llm)
        results.append(r)
        print_verdict_case(r)

    u = llm.usage
    print_verdict_summary(results, u)
    if partial and RESULTS_FULL.exists():
        # Résultats partiels (--only) : ils remplacent les mêmes cas dans le fichier existant.
        new = {r["id"]: r for r in results}
        previous = json.loads(RESULTS_FULL.read_text(encoding="utf-8"))["cas"]
        results = [new.pop(r["id"], r) for r in previous] + list(new.values())
    summary = {"verdicts_corrects": sum(r["correct"] for r in results), "cas": len(results), "tokens": u}
    RESULTS_FULL.write_text(json.dumps({"resume": summary, "cas": results}, ensure_ascii=False, indent=1),
                            encoding="utf-8")
    print(f"Détail -> {RESULTS_FULL.relative_to(ROOT)}"
          + (f" ({len(results)} cas après fusion)" if partial else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation de la recherche ou du verdict.")
    parser.add_argument("--cases", type=Path, default=ROOT / "eval" / "cases.json")
    parser.add_argument("--mode", choices=("retrieval", "full"), default="retrieval",
                        help="retrieval : rappel de la recherche ; full : verdict complet (LLM requis)")
    parser.add_argument("--llm", action="store_true", help="analyse par LLM avant la recherche")
    parser.add_argument("--pause", type=float, default=0, help="secondes d'attente entre deux cas")
    parser.add_argument("--only", type=lambda v: [i.strip() for i in v.split(",") if i.strip()],
                        metavar="ID1,ID2", help="ne lance que ces cas")
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--from-results", action="store_true",
                       help=f"réaffiche {RESULTS_FULL.relative_to(ROOT)} sans appeler l'API")
    reuse.add_argument("--reverify", action="store_true",
                       help=f"refait seulement la vérification des citations de {RESULTS_FULL.relative_to(ROOT)}")
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.only:
        unknown = set(args.only) - {c["id"] for c in cases}
        if unknown:
            sys.exit(f"Cas inconnus : {', '.join(sorted(unknown))}")
    if args.from_results or args.reverify:
        show_full_results(cases, args.only, reverify=args.reverify)
        return
    if args.only:
        cases = [c for c in cases if c["id"] in args.only]
    if args.mode == "full":
        run_full(cases, args.pause, partial=bool(args.only))
        return

    llm = None
    if args.llm:
        from sinistre_cg.llm import LLM, LLMError
        try:
            llm = LLM()
        except LLMError as exc:
            sys.exit(f"Erreur LLM : {exc}")

    retriever = Retriever()
    total_chars = contract_size()

    results = []
    for n, case in enumerate(cases):
        if n and args.pause:
            time.sleep(args.pause)
        r = evaluate_case(retriever, case, llm)
        results.append(r)
        if r["rappel"] is None:
            status, recall = "n/a", "   —"
        else:
            status, recall = ("OK" if not r["manquantes"] else "--"), f"{r['rappel']:4.0%}"
        missing = ", ".join(
            m + (" (proposée par le LLM)" if m in r["manquantes_proposees_par_llm"]
                 else " (non proposée par le LLM)" if r["analyse"] else "")
            for m in r["manquantes"]
        )
        if r["erreur_llm"]:
            missing += f"  [LLM en échec : {r['erreur_llm'][:80]}]"
        print(f"{status:<3} {r['id']:<28} {recall}  {r['contexte_car'] / 1000:5.1f}k  {missing}", flush=True)

    scored = [r for r in results if r["rappel"] is not None]
    mean_recall = _mean_recall(results)
    complete = sum(not r["manquantes"] for r in scored)
    mean_context = sum(r["contexte_car"] for r in results) / len(results)
    summary = {
        "llm": bool(llm),
        "rappel_moyen": round(mean_recall, 4),
        "cas_complets": complete,
        "cas_evalues": len(scored),
        "cas_sans_section_attendue": len(results) - len(scored),
        "contexte_moyen_car": round(mean_context),
        "contrat_car": total_chars,
        "contexte_moyen_pct_contrat": round(mean_context / total_chars, 4),
        "echecs_llm": sum(bool(r["erreur_llm"]) for r in results),
        "tokens": llm.usage if llm else None,
    }

    print(f"\nRappel moyen : {mean_recall:.1%} ({len(scored)} cas avec sections attendues)")
    print(f"Cas complets : {complete}/{len(scored)}")
    print(f"Contexte moyen : {mean_context / 1000:.1f}k car., soit {mean_context / total_chars:.1%} "
          f"du contrat ({total_chars / 1000:.0f}k car.)")
    if llm:
        u = llm.usage
        print(f"Tokens consommés : {u['total_tokens']} ({u['prompt_tokens']} entrée + "
              f"{u['completion_tokens']} sortie, {u['calls']} appels) ; échecs LLM : {summary['echecs_llm']}")

    out = RESULTS[bool(llm)]
    out.write_text(json.dumps({"resume": summary, "cas": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Détail -> {out.relative_to(ROOT)}")

    other = RESULTS[not llm]
    if other.exists():
        other_recall = json.loads(other.read_text(encoding="utf-8"))["resume"]["rappel_moyen"]
        bm25, with_llm = (other_recall, mean_recall) if llm else (mean_recall, other_recall)
        print(f"\nRappel moyen  BM25 seul : {bm25:.1%}   |   avec LLM : {with_llm:.1%}")


if __name__ == "__main__":
    main()
