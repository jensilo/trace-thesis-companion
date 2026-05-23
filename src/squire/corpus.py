import asyncio
import datetime
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from rich.console import Console

from squire.config import (
    OUTPUT_DIR,
    DEFAULT_SYNTHESIS_MODEL,
    DEFAULT_QUALITY_JUDGE_MODEL,
    DEFAULT_META_JUDGE_MODEL,
)

console = Console(force_terminal=True)

CORPUS_PROJECT_IDS = ["3", "4", "5", "6", "8"]
CORPUS_INTERVIEWER_IDS = ["re_novice", "re_experienced"]
CORPUS_STAKEHOLDER_IDS = [
    "stakeholder_technical_senior",
    "stakeholder_technical_junior",
    "stakeholder_nontechnical_manager",
    "stakeholder_nontechnical_enduser",
]
CORPUS_CONCURRENCY = 10

QUALITY_DIMS = ["structuring", "clarity", "responsiveness", "rigor"]
META_DIMS = ["completeness", "realism"]

_INTERVIEWER_SHORT: dict[str, str] = {"re_novice": "novice", "re_experienced": "expert"}
_STAKEHOLDER_SHORT: dict[str, str] = {
    "stakeholder_technical_senior": "tech_sr",
    "stakeholder_technical_junior": "tech_jr",
    "stakeholder_nontechnical_manager": "mgr",
    "stakeholder_nontechnical_enduser": "user",
}


def _short_label(project_id: str, interviewer_id: str, stakeholder_id: str) -> str:
    iid = _INTERVIEWER_SHORT.get(interviewer_id, interviewer_id[:6])
    sid = _STAKEHOLDER_SHORT.get(stakeholder_id, stakeholder_id[:6])
    return f"[{project_id}/{iid}/{sid}]"


def run_synthesize_corpus(
    synthesis_model: str = DEFAULT_SYNTHESIS_MODEL,
    quality_model: str = DEFAULT_QUALITY_JUDGE_MODEL,
    meta_model: str = DEFAULT_META_JUDGE_MODEL,
    single_shot: bool = False,
    no_refine: bool = False,
    quality_samples: int = 5,
    meta_samples: int = 5,
    concurrency: int = CORPUS_CONCURRENCY,
) -> None:
    asyncio.run(_async_run_corpus(
        synthesis_model=synthesis_model,
        quality_model=quality_model,
        meta_model=meta_model,
        single_shot=single_shot,
        no_refine=no_refine,
        quality_samples=quality_samples,
        meta_samples=meta_samples,
        concurrency=concurrency,
    ))


async def _async_run_corpus(
    synthesis_model: str,
    quality_model: str,
    meta_model: str,
    single_shot: bool,
    no_refine: bool,
    quality_samples: int,
    meta_samples: int,
    concurrency: int,
) -> None:
    corpus_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    corpus_dir = OUTPUT_DIR / f"corpus-{corpus_timestamp}"
    corpus_dir.mkdir(parents=True)

    combinations = [
        (pid, iid, sid)
        for pid in CORPUS_PROJECT_IDS
        for iid in CORPUS_INTERVIEWER_IDS
        for sid in CORPUS_STAKEHOLDER_IDS
    ]
    total = len(combinations)
    mode = "single-shot" if single_shot else ("3-step" if no_refine else "4-step")

    console.print(f"\n[bold cyan]══════════ SQuIRE Corpus Synthesis ══════════[/bold cyan]")
    console.print(f"  Corpus:      corpus-{corpus_timestamp}")
    console.print(f"  Output:      {corpus_dir}")
    console.print(f"  Transcripts: {total}  ({len(CORPUS_PROJECT_IDS)} projects × {len(CORPUS_INTERVIEWER_IDS)} interviewers × {len(CORPUS_STAKEHOLDER_IDS)} stakeholders)")
    console.print(f"  Synthesizer: {synthesis_model}  [{mode}]")
    console.print(f"  Evaluator:   quality={quality_model}  meta={meta_model}")
    console.print(f"  Samples:     quality={quality_samples}  meta={meta_samples}")
    console.print(f"  Concurrency: {concurrency}")
    console.print()

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    wall_start = time.time()

    async def run_one(project_id: str, interviewer_id: str, stakeholder_id: str) -> dict:
        nonlocal completed
        run_id = f"{interviewer_id}_{stakeholder_id}"
        async with semaphore:
            console.print(f"  [dim]→ {run_id}[/dim]")
            result = await asyncio.to_thread(
                _synthesize_and_evaluate_one,
                project_id, interviewer_id, stakeholder_id,
                synthesis_model, quality_model, meta_model,
                single_shot, no_refine, quality_samples, meta_samples,
                corpus_dir, run_id,
            )
        async with lock:
            completed += 1
            elapsed = result.get("elapsed_seconds", 0)
            if "error" in result:
                console.print(f"  [red]✗[/red] [{completed}/{total}] {run_id}  ({elapsed:.1f}s)  error: {result['error']}")
            else:
                q_mean = result.get("evaluation", {}).get("g_eval_quality", {}).get("overall_mean")
                m_mean = result.get("evaluation", {}).get("g_eval_meta", {}).get("overall_mean")
                bert_f1 = (result.get("evaluation", {}).get("bert_score") or {}).get("f1")
                score_str = ""
                if q_mean is not None:
                    score_str += f"  quality={q_mean:.2f}"
                if m_mean is not None:
                    score_str += f"  meta={m_mean:.2f}"
                if bert_f1 is not None:
                    score_str += f"  bert={bert_f1:.3f}"
                console.print(f"  [green]✓[/green] [{completed}/{total}] {run_id}  ({elapsed:.1f}s){score_str}")
        return result

    tasks = [run_one(*combo) for combo in combinations]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - wall_start

    # Normalise: exceptions become error dicts
    result_dicts: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            pid, iid, sid = combinations[i]
            result_dicts.append({
                "run_id": f"{pid}_{iid}_{sid}",
                "project_id": pid,
                "interviewer_id": iid,
                "stakeholder_id": sid,
                "error": str(r),
                "elapsed_seconds": 0,
            })
        else:
            result_dicts.append(r)

    successful = [r for r in result_dicts if "error" not in r]
    failed = [r for r in result_dicts if "error" in r]

    console.print()
    console.print(f"[bold]Corpus complete[/bold] in {total_elapsed:.1f}s")
    console.print(f"  Successful: [green]{len(successful)}[/green] / {total}")
    if failed:
        console.print(f"  Failed:     [red]{len(failed)}[/red]")
        for f in failed:
            console.print(f"    [red]✗[/red] {f['run_id']}: {f['error']}")

    console.print("\n[bold cyan]Computing corpus meta-analysis...[/bold cyan]")
    meta = _compute_corpus_meta(
        corpus_id=f"corpus-{corpus_timestamp}",
        combinations=combinations,
        results=result_dicts,
        synthesis_model=synthesis_model,
        quality_model=quality_model,
        meta_model=meta_model,
        single_shot=single_shot,
        no_refine=no_refine,
        quality_samples=quality_samples,
        meta_samples=meta_samples,
        total_elapsed_seconds=total_elapsed,
    )

    meta_path = corpus_dir / "corpus_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    console.print(f"  [bold green]Corpus meta:[/bold green] {meta_path}")


def _synthesize_and_evaluate_one(
    project_id: str,
    interviewer_id: str,
    stakeholder_id: str,
    synthesis_model: str,
    quality_model: str,
    meta_model: str,
    single_shot: bool,
    no_refine: bool,
    quality_samples: int,
    meta_samples: int,
    corpus_dir: Path,
    run_id: str,
) -> dict:
    # Deferred imports: each thread must not share module-level agent state
    from squire.synthesis import run_synthesis
    from squire.evaluation import run_evaluation

    job_label = _short_label(project_id, interviewer_id, stakeholder_id)
    start = time.time()
    try:
        synthesis_result = run_synthesis(
            project_id=project_id,
            model=synthesis_model,
            stakeholder_id=stakeholder_id,
            interviewer_id=interviewer_id,
            single_shot=single_shot,
            no_refine=no_refine,
            debug=False,
            output_dir=corpus_dir,
            run_id=run_id,
        )

        run_evaluation(
            transcript_file=synthesis_result.transcript_path,
            quality_model=quality_model,
            meta_model=meta_model,
            quality_samples=quality_samples,
            meta_samples=meta_samples,
            output_dir=corpus_dir,
            job_label=job_label,
        )

        elapsed = round(time.time() - start, 1)
        metadata = json.loads(synthesis_result.metadata_path.read_text(encoding="utf-8"))

        return {
            "run_id": run_id,
            "project_id": project_id,
            "interviewer_id": interviewer_id,
            "stakeholder_id": stakeholder_id,
            "transcript_path": str(synthesis_result.transcript_path),
            "metadata_path": str(synthesis_result.metadata_path),
            "elapsed_seconds": elapsed,
            "evaluation": metadata.get("evaluation", {}),
            "selected_requirements": metadata.get("selected_requirements", []),
        }
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return {
            "run_id": run_id,
            "project_id": project_id,
            "interviewer_id": interviewer_id,
            "stakeholder_id": stakeholder_id,
            "error": str(e),
            "elapsed_seconds": elapsed,
        }


def _compute_corpus_meta(
    corpus_id: str,
    combinations: list[tuple],
    results: list[dict],
    synthesis_model: str,
    quality_model: str,
    meta_model: str,
    single_shot: bool,
    no_refine: bool,
    quality_samples: int,
    meta_samples: int,
    total_elapsed_seconds: float,
) -> dict[str, Any]:
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    def _agg(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"mean": None, "std_dev": None, "count": 0}
        return {
            "mean": round(statistics.mean(values), 3),
            "std_dev": round(statistics.stdev(values), 3) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    def _group_stats(subset: list[dict]) -> dict[str, Any]:
        q_overall: list[float] = []
        m_overall: list[float] = []
        q_dims: dict[str, list[float]] = {d: [] for d in QUALITY_DIMS}
        m_dims: dict[str, list[float]] = {d: [] for d in META_DIMS}
        bert_f1: list[float] = []
        elapsed: list[float] = []

        for r in subset:
            ev = r.get("evaluation", {})
            gq = ev.get("g_eval_quality", {})
            gm = ev.get("g_eval_meta", {})

            if gq.get("overall_mean") is not None:
                q_overall.append(gq["overall_mean"])
            if gm.get("overall_mean") is not None:
                m_overall.append(gm["overall_mean"])

            for d in QUALITY_DIMS:
                v = gq.get("dimensions", {}).get(d, {}).get("mean")
                if v is not None:
                    q_dims[d].append(v)
            for d in META_DIMS:
                v = gm.get("dimensions", {}).get(d, {}).get("mean")
                if v is not None:
                    m_dims[d].append(v)

            f1 = (ev.get("bert_score") or {}).get("f1")
            if f1 is not None:
                bert_f1.append(f1)

            if r.get("elapsed_seconds") is not None:
                elapsed.append(r["elapsed_seconds"])

        return {
            "count": len(subset),
            "avg_elapsed_seconds": round(statistics.mean(elapsed), 1) if elapsed else None,
            "quality_overall": _agg(q_overall),
            "quality_dimensions": {d: _agg(q_dims[d]) for d in QUALITY_DIMS},
            "meta_overall": _agg(m_overall),
            "meta_dimensions": {d: _agg(m_dims[d]) for d in META_DIMS},
            "bert_score_f1": _agg(bert_f1),
        }

    # ── Requirement frequency across corpus ───────────────────────────────────
    req_counts: dict[int, dict] = {}
    for r in successful:
        for req in r.get("selected_requirements", []):
            line = req.get("line")
            if line is None:
                continue
            if line not in req_counts:
                req_counts[line] = {
                    "line": line,
                    "text": req.get("text", ""),
                    "categories": req.get("categories", []),
                    "count": 0,
                    "by_stakeholder": defaultdict(int),
                    "by_interviewer": defaultdict(int),
                    "by_project": defaultdict(int),
                }
            req_counts[line]["count"] += 1
            req_counts[line]["by_stakeholder"][r["stakeholder_id"]] += 1
            req_counts[line]["by_interviewer"][r["interviewer_id"]] += 1
            req_counts[line]["by_project"][r["project_id"]] += 1

    req_freq = sorted(req_counts.values(), key=lambda x: x["count"], reverse=True)
    for rf in req_freq:
        rf["by_stakeholder"] = dict(rf["by_stakeholder"])
        rf["by_interviewer"] = dict(rf["by_interviewer"])
        rf["by_project"] = dict(rf["by_project"])

    # ── Per-stakeholder requirement category affinity ─────────────────────────
    persona_affinity: dict[str, dict] = {}
    for sid in CORPUS_STAKEHOLDER_IDS:
        subset = [r for r in successful if r["stakeholder_id"] == sid]
        cat_counts: dict[str, int] = defaultdict(int)
        req_line_counts: dict[int, dict] = {}

        for r in subset:
            for req in r.get("selected_requirements", []):
                for cat in req.get("categories", []):
                    cat_counts[cat] += 1
                line = req.get("line")
                if line:
                    if line not in req_line_counts:
                        req_line_counts[line] = {"count": 0, "text": req.get("text", "")}
                    req_line_counts[line]["count"] += 1

        top_lines = sorted(req_line_counts, key=lambda l: req_line_counts[l]["count"], reverse=True)[:10]
        persona_affinity[sid] = {
            "top_categories": dict(sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)),
            "top_requirements": [
                {"line": l, "count": req_line_counts[l]["count"], "text": req_line_counts[l]["text"]}
                for l in top_lines
            ],
        }

    return {
        "corpus_id": corpus_id,
        "generated_at": datetime.datetime.now().isoformat(),
        "total_elapsed_seconds": round(total_elapsed_seconds, 1),
        "configuration": {
            "project_ids": CORPUS_PROJECT_IDS,
            "interviewer_ids": CORPUS_INTERVIEWER_IDS,
            "stakeholder_ids": CORPUS_STAKEHOLDER_IDS,
            "synthesis_model": synthesis_model,
            "quality_model": quality_model,
            "meta_model": meta_model,
            "single_shot": single_shot,
            "no_refine": no_refine,
            "quality_samples": quality_samples,
            "meta_samples": meta_samples,
        },
        "summary": {
            "total": len(combinations),
            "successful": len(successful),
            "failed": len(failed),
            "failures": [{"run_id": r["run_id"], "error": r.get("error", "")} for r in failed],
        },
        "overall": _group_stats(successful),
        "by_interviewer": {
            iid: _group_stats([r for r in successful if r["interviewer_id"] == iid])
            for iid in CORPUS_INTERVIEWER_IDS
        },
        "by_stakeholder": {
            sid: _group_stats([r for r in successful if r["stakeholder_id"] == sid])
            for sid in CORPUS_STAKEHOLDER_IDS
        },
        "by_project": {
            pid: _group_stats([r for r in successful if r["project_id"] == pid])
            for pid in CORPUS_PROJECT_IDS
        },
        "requirement_frequency": req_freq,
        "persona_requirement_affinity": persona_affinity,
    }


# ── Corpus re-evaluation ──────────────────────────────────────────────────────


def _find_corpus_dir(corpus_arg: str | None) -> Path:
    """Resolve a corpus directory from a name, timestamp, path, or None (→ latest)."""
    if corpus_arg is None:
        dirs = [
            d for d in OUTPUT_DIR.iterdir()
            if d.is_dir() and d.name.startswith("corpus-") and "-reeval-" not in d.name
        ]
        if not dirs:
            raise FileNotFoundError("No original corpus directories found in outputs/")
        return max(dirs, key=lambda d: d.stat().st_mtime)

    p = Path(corpus_arg)
    if p.exists() and p.is_dir():
        return p

    name = corpus_arg if corpus_arg.startswith("corpus-") else f"corpus-{corpus_arg}"
    p = OUTPUT_DIR / name
    if p.exists() and p.is_dir():
        return p

    raise FileNotFoundError(f"Corpus directory not found: {corpus_arg!r}. Looked in {OUTPUT_DIR}/")


def _load_original_results(corpus_dir: Path) -> list[dict]:
    """Read all transcript metadata JSONs from a corpus dir and return normalised result dicts."""
    results: list[dict] = []
    for metadata_path in sorted(corpus_dir.glob("transcript_*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        project_id = metadata.get("project_id", "")
        interviewer_id = metadata.get("interviewer_persona", "")
        stakeholder_id = metadata.get("stakeholder_persona", "")
        results.append({
            "run_id": f"{interviewer_id}_{stakeholder_id}",
            "project_id": project_id,
            "interviewer_id": interviewer_id,
            "stakeholder_id": stakeholder_id,
            "evaluation": metadata.get("evaluation", {}),
            "selected_requirements": metadata.get("selected_requirements", []),
        })
    return results


def _reevaluate_one(
    metadata_path: Path,
    quality_model: str,
    meta_model: str,
    quality_samples: int,
    meta_samples: int,
    reeval_dir: Path,
    job_label: str,
    skip_bert: bool = False,
) -> dict:
    """Blocking: re-evaluate one transcript without modifying original metadata."""
    from squire.evaluation import run_evaluation

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "run_id": metadata_path.stem,
            "project_id": "", "interviewer_id": "", "stakeholder_id": "",
            "error": f"Failed to read metadata: {e}",
            "elapsed_seconds": 0.0,
        }

    project_id = metadata.get("project_id", "")
    interviewer_id = metadata.get("interviewer_persona", "")
    stakeholder_id = metadata.get("stakeholder_persona", "")
    run_id = f"{interviewer_id}_{stakeholder_id}"

    start = time.time()
    try:
        transcript_path = metadata_path.with_suffix(".txt")
        eval_result = run_evaluation(
            transcript_file=transcript_path,
            quality_model=quality_model,
            meta_model=meta_model,
            quality_samples=quality_samples,
            meta_samples=meta_samples,
            output_dir=reeval_dir,
            save_metadata=False,
            skip_bert=skip_bert,
            job_label=job_label,
        )
        return {
            "run_id": run_id,
            "project_id": project_id,
            "interviewer_id": interviewer_id,
            "stakeholder_id": stakeholder_id,
            "transcript_path": str(transcript_path),
            "metadata_path": str(metadata_path),
            "elapsed_seconds": round(time.time() - start, 1),
            "evaluation": eval_result,
            "selected_requirements": metadata.get("selected_requirements", []),
        }
    except Exception as e:
        return {
            "run_id": run_id,
            "project_id": project_id,
            "interviewer_id": interviewer_id,
            "stakeholder_id": stakeholder_id,
            "error": str(e),
            "elapsed_seconds": round(time.time() - start, 1),
        }


def _compute_corpus_meta_diff(
    source_corpus_dir: Path,
    original_meta: dict,
    reeval_meta: dict,
    reeval_results: list[dict],
) -> dict[str, Any]:
    """Diff two corpus_meta structures, including per-transcript deltas and a stability verdict."""
    original_results = _load_original_results(source_corpus_dir)
    orig_by_key: dict[str, dict] = {
        f"{r['project_id']}_{r['run_id']}": r for r in original_results
    }

    def _diff_group(orig: dict, reev: dict) -> dict:
        out: dict[str, Any] = {}
        for key in ("quality_overall", "meta_overall", "bert_score_f1"):
            o, r = orig.get(key, {}), reev.get(key, {})
            o_m, r_m = o.get("mean"), r.get("mean")
            out[key] = {
                "original_mean": o_m,
                "reeval_mean": r_m,
                "delta": round(r_m - o_m, 3) if (o_m is not None and r_m is not None) else None,
                "original_std_dev": o.get("std_dev"),
                "reeval_std_dev": r.get("std_dev"),
            }
        for group_key in ("quality_dimensions", "meta_dimensions"):
            o_dims, r_dims = orig.get(group_key, {}), reev.get(group_key, {})
            out[group_key] = {}
            for dim in set(o_dims) | set(r_dims):
                o_m = o_dims.get(dim, {}).get("mean")
                r_m = r_dims.get(dim, {}).get("mean")
                out[group_key][dim] = {
                    "original_mean": o_m,
                    "reeval_mean": r_m,
                    "delta": round(r_m - o_m, 3) if (o_m is not None and r_m is not None) else None,
                }
        return out

    per_transcript: list[dict] = []
    for rr in reeval_results:
        if "error" in rr:
            continue
        key = f"{rr['project_id']}_{rr['run_id']}"
        orig_r = orig_by_key.get(key)
        if orig_r is None:
            continue
        o_ev, r_ev = orig_r.get("evaluation", {}), rr.get("evaluation", {})
        o_q = o_ev.get("g_eval_quality", {}).get("overall_mean")
        r_q = r_ev.get("g_eval_quality", {}).get("overall_mean")
        o_m = o_ev.get("g_eval_meta", {}).get("overall_mean")
        r_m = r_ev.get("g_eval_meta", {}).get("overall_mean")
        o_b = (o_ev.get("bert_score") or {}).get("f1")
        r_b = (r_ev.get("bert_score") or {}).get("f1")
        per_transcript.append({
            "run_id": rr["run_id"],
            "project_id": rr["project_id"],
            "interviewer_id": rr["interviewer_id"],
            "stakeholder_id": rr["stakeholder_id"],
            "quality": {
                "original": o_q, "reeval": r_q,
                "delta": round(r_q - o_q, 3) if (o_q is not None and r_q is not None) else None,
            },
            "meta": {
                "original": o_m, "reeval": r_m,
                "delta": round(r_m - o_m, 3) if (o_m is not None and r_m is not None) else None,
            },
            "bert_f1": {
                "original": o_b, "reeval": r_b,
                "delta": round(r_b - o_b, 4) if (o_b is not None and r_b is not None) else None,
            },
        })

    per_transcript.sort(key=lambda x: abs(x["quality"].get("delta") or 0), reverse=True)

    q_deltas = [t["quality"]["delta"] for t in per_transcript if t["quality"]["delta"] is not None]
    m_deltas = [t["meta"]["delta"] for t in per_transcript if t["meta"]["delta"] is not None]
    b_deltas = [t["bert_f1"]["delta"] for t in per_transcript if t["bert_f1"]["delta"] is not None]

    def _delta_stats(deltas: list[float]) -> dict:
        if not deltas:
            return {"mean_delta": None, "std_dev_delta": None, "max_abs_delta": None}
        return {
            "mean_delta": round(statistics.mean(deltas), 3),
            "std_dev_delta": round(statistics.stdev(deltas), 3) if len(deltas) > 1 else 0.0,
            "max_abs_delta": round(max(abs(d) for d in deltas), 3),
        }

    q_stats = _delta_stats(q_deltas)
    m_stats = _delta_stats(m_deltas)
    stable = (
        abs(q_stats.get("mean_delta") or 0) < 0.5
        and abs(m_stats.get("mean_delta") or 0) < 0.5
    )

    orig_conf = original_meta.get("configuration", {})
    reeval_conf = reeval_meta.get("configuration", {})
    conf_diff = {
        k: {"original": orig_conf.get(k), "reeval": reeval_conf.get(k)}
        for k in ("quality_model", "meta_model", "quality_samples", "meta_samples")
        if orig_conf.get(k) != reeval_conf.get(k)
    }

    return {
        "original_corpus_id": original_meta.get("corpus_id"),
        "reeval_corpus_id": reeval_meta.get("corpus_id"),
        "generated_at": datetime.datetime.now().isoformat(),
        "configuration_diff": conf_diff,
        "overall": _diff_group(original_meta.get("overall", {}), reeval_meta.get("overall", {})),
        "by_interviewer": {
            iid: _diff_group(
                original_meta.get("by_interviewer", {}).get(iid, {}),
                reeval_meta.get("by_interviewer", {}).get(iid, {}),
            )
            for iid in CORPUS_INTERVIEWER_IDS
        },
        "by_stakeholder": {
            sid: _diff_group(
                original_meta.get("by_stakeholder", {}).get(sid, {}),
                reeval_meta.get("by_stakeholder", {}).get(sid, {}),
            )
            for sid in CORPUS_STAKEHOLDER_IDS
        },
        "by_project": {
            pid: _diff_group(
                original_meta.get("by_project", {}).get(pid, {}),
                reeval_meta.get("by_project", {}).get(pid, {}),
            )
            for pid in CORPUS_PROJECT_IDS
        },
        "per_transcript": per_transcript,
        "stability": {
            "quality": q_stats,
            "meta": m_stats,
            "bert_f1": _delta_stats(b_deltas),
            "verdict": "stable" if stable else "unstable",
        },
    }


async def _async_reevaluate_corpus(
    source_corpus_dir: Path,
    quality_model: str,
    meta_model: str,
    quality_samples: int,
    meta_samples: int,
    concurrency: int,
    skip_bert: bool = False,
) -> None:
    source_corpus_id = source_corpus_dir.name
    reeval_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    reeval_corpus_id = f"{source_corpus_id}-reeval-{reeval_timestamp}"
    reeval_dir = OUTPUT_DIR / reeval_corpus_id
    reeval_dir.mkdir(parents=True)

    metadata_files = sorted(source_corpus_dir.glob("transcript_*.json"))
    total = len(metadata_files)

    if total == 0:
        console.print(f"[red]No transcript metadata files found in {source_corpus_dir}[/red]")
        return

    console.print(f"\n[bold cyan]══════════ SQuIRE Corpus Re-Evaluation ══════════[/bold cyan]")
    console.print(f"  Source:      {source_corpus_id}")
    console.print(f"  Re-eval ID:  {reeval_corpus_id}")
    console.print(f"  Output:      {reeval_dir}")
    console.print(f"  Transcripts: {total}")
    console.print(f"  Quality:     {quality_model}  (n={quality_samples})")
    console.print(f"  Meta:        {meta_model}  (n={meta_samples})")
    console.print(f"  Concurrency: {concurrency}")
    if skip_bert:
        console.print(f"  BERTScore:   [dim]skipped[/dim]")
    console.print()

    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    lock = asyncio.Lock()
    wall_start = time.time()

    async def reeval_one(metadata_path: Path) -> dict:
        nonlocal completed
        # Peek at metadata for the short label — cheap read before acquiring semaphore
        try:
            meta_peek = json.loads(metadata_path.read_text(encoding="utf-8"))
            pid = meta_peek.get("project_id", "?")
            iid = meta_peek.get("interviewer_persona", "?")
            sid = meta_peek.get("stakeholder_persona", "?")
        except Exception:
            pid, iid, sid = "?", "?", "?"

        job_label = _short_label(pid, iid, sid)

        async with semaphore:
            console.print(f"  [dim]→ {job_label}[/dim]")
            result = await asyncio.to_thread(
                _reevaluate_one,
                metadata_path, quality_model, meta_model,
                quality_samples, meta_samples, reeval_dir, job_label, skip_bert,
            )

        async with lock:
            completed += 1
            elapsed = result.get("elapsed_seconds", 0)
            if "error" in result:
                console.print(f"  [red]✗[/red] [{completed}/{total}] {job_label}  ({elapsed:.1f}s)  error: {result['error']}")
            else:
                q_mean = result.get("evaluation", {}).get("g_eval_quality", {}).get("overall_mean")
                m_mean = result.get("evaluation", {}).get("g_eval_meta", {}).get("overall_mean")
                bert_f1 = (result.get("evaluation", {}).get("bert_score") or {}).get("f1")
                score_str = ""
                if q_mean is not None:
                    score_str += f"  quality={q_mean:.2f}"
                if m_mean is not None:
                    score_str += f"  meta={m_mean:.2f}"
                if bert_f1 is not None:
                    score_str += f"  bert={bert_f1:.3f}"
                console.print(f"  [green]✓[/green] [{completed}/{total}] {job_label}  ({elapsed:.1f}s){score_str}")
        return result

    tasks = [reeval_one(mf) for mf in metadata_files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_elapsed = time.time() - wall_start

    result_dicts: list[dict] = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            result_dicts.append({
                "run_id": metadata_files[i].stem,
                "project_id": "", "interviewer_id": "", "stakeholder_id": "",
                "error": str(r),
                "elapsed_seconds": 0,
            })
        else:
            result_dicts.append(r)

    successful = [r for r in result_dicts if "error" not in r]
    failed = [r for r in result_dicts if "error" in r]

    console.print()
    console.print(f"[bold]Re-evaluation complete[/bold] in {total_elapsed:.1f}s")
    console.print(f"  Successful: [green]{len(successful)}[/green] / {total}")
    if failed:
        console.print(f"  Failed:     [red]{len(failed)}[/red]")
        for f in failed:
            console.print(f"    [red]✗[/red] {f['run_id']}: {f.get('error', '')}")

    original_meta_path = source_corpus_dir / "corpus_meta.json"
    original_meta = (
        json.loads(original_meta_path.read_text(encoding="utf-8"))
        if original_meta_path.exists() else {}
    )
    orig_conf = original_meta.get("configuration", {})

    console.print("\n[bold cyan]Computing corpus meta-analysis...[/bold cyan]")
    combinations = [(r["project_id"], r["interviewer_id"], r["stakeholder_id"]) for r in result_dicts]
    reeval_meta = _compute_corpus_meta(
        corpus_id=reeval_corpus_id,
        combinations=combinations,
        results=result_dicts,
        synthesis_model=orig_conf.get("synthesis_model", ""),
        quality_model=quality_model,
        meta_model=meta_model,
        single_shot=orig_conf.get("single_shot", False),
        no_refine=orig_conf.get("no_refine", True),
        quality_samples=quality_samples,
        meta_samples=meta_samples,
        total_elapsed_seconds=total_elapsed,
    )

    meta_path = reeval_dir / "corpus_meta.json"
    meta_path.write_text(json.dumps(reeval_meta, indent=2, default=str), encoding="utf-8")
    console.print(f"  [bold green]Corpus meta:[/bold green] {meta_path}")

    console.print("[bold cyan]Computing corpus meta diff...[/bold cyan]")
    diff = _compute_corpus_meta_diff(source_corpus_dir, original_meta, reeval_meta, result_dicts)
    diff_path = reeval_dir / "corpus_meta_diff.json"
    diff_path.write_text(json.dumps(diff, indent=2, default=str), encoding="utf-8")
    console.print(f"  [bold green]Corpus diff:[/bold green] {diff_path}")

    stability = diff.get("stability", {})
    verdict = stability.get("verdict", "unknown")
    color = "green" if verdict == "stable" else "red"
    console.print(f"\n[bold]Stability verdict: [{color}]{verdict}[/{color}][/bold]")
    q_s = stability.get("quality", {})
    m_s = stability.get("meta", {})
    if q_s.get("mean_delta") is not None:
        console.print(f"  Quality Δ̄={q_s['mean_delta']:+.3f}  σ={q_s.get('std_dev_delta', 0):.3f}  max|Δ|={q_s.get('max_abs_delta', 0):.3f}")
    if m_s.get("mean_delta") is not None:
        console.print(f"  Meta    Δ̄={m_s['mean_delta']:+.3f}  σ={m_s.get('std_dev_delta', 0):.3f}  max|Δ|={m_s.get('max_abs_delta', 0):.3f}")


def run_reevaluate_corpus(
    corpus: str | None = None,
    quality_model: str = DEFAULT_QUALITY_JUDGE_MODEL,
    meta_model: str = DEFAULT_META_JUDGE_MODEL,
    quality_samples: int = 5,
    meta_samples: int = 5,
    concurrency: int = CORPUS_CONCURRENCY,
    skip_bert: bool = False,
) -> None:
    source_dir = _find_corpus_dir(corpus)
    console.print(f"[bold]Re-evaluating corpus:[/bold] {source_dir.name}")
    asyncio.run(_async_reevaluate_corpus(
        source_corpus_dir=source_dir,
        quality_model=quality_model,
        meta_model=meta_model,
        quality_samples=quality_samples,
        meta_samples=meta_samples,
        concurrency=concurrency,
        skip_bert=skip_bert,
    ))
