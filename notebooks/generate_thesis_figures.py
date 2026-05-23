"""
Thesis Figure Generation Script
================================
Generates polished, thesis-ready figures for Chapter 5 (Results).
This script has NO analytical value; it is a figure export utility.
Run from the `codeproj/` directory:

    uv run python notebooks/generate_thesis_figures.py

Output PNGs are written to notebooks/ — move them to thesis/figures/.

Depends on: matplotlib, seaborn, pandas, numpy, scipy, scikit-learn
Does NOT require: sentence_transformers (no embedding model needed)

fig_tsne_embedding() requires pre-computed tsne_coords.npz + tsne_meta.json.
Generate them by running cell 63 (tsne-save-coords) in thesis_analysis.ipynb
after cells 02–10 have executed.
"""

from __future__ import annotations
import json, re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_cm

# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUTS      = Path("../outputs")
FIG_OUT      = Path(".")   # save here, then move to thesis/figures/
SQUIRE_SS    = OUTPUTS / "corpus-20260227_105946"
SQUIRE_3S    = OUTPUTS / "corpus-20260226_162106"

PROJECT_NAMES = {
    "3": "Nursing\nScheduling",
    "4": "Credit Card\nDisputes",
    "5": "Recycled Parts\nMgmt",
    "6": "Conference\nRoom Mgmt",
    "8": "IzognMovies\nStreaming",
}
INTERVIEWER_LABELS = {
    "re_novice":      "Novice RE",
    "re_experienced": "Experienced RE",
}

TRACE_MANIFEST = [
    {"label": "Mistral Small\nT=0.0",   "group": "model", "phase": "exploratory", "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.0,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_165429"},
    {"label": "Mistral Small\nT=0.3",   "group": "temp",  "phase": "exploratory", "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.3,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_175157"},
    {"label": "Mistral Small\nT=0.7",   "group": "temp",  "phase": "exploratory", "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.7,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_174709"},
    {"label": "Mistral Medium\nT=0.0",  "group": "model", "phase": "exploratory", "corpus": "single-shot",
     "model": "mistral-medium", "temp": 0.0,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-medium_trace_20260302_170431"},
    {"label": "GPT-5-mini\nlow-effort", "group": "model", "phase": "exploratory", "corpus": "single-shot",
     "model": "gpt-5-mini",     "temp": None,
     "path": OUTPUTS / "corpus_20260227_105946_gpt-5-mini_trace_20260302_172047"},
    {"label": "Mistral Small\nT=0.5 #1","group": "temp",  "phase": "primary",     "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.5,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_173239"},
    {"label": "Mistral Small\nT=0.5 #2","group": "temp",  "phase": "primary",     "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.5,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_175542"},
    {"label": "Mistral Small\nT=0.5 #3","group": "temp",  "phase": "primary",     "corpus": "single-shot",
     "model": "mistral-small",  "temp": 0.5,
     "path": OUTPUTS / "corpus_20260227_105946_mistral-small_trace_20260302_180009"},
    {"label": "Mistral Small\nT=0.5 3-step #1","group": "3step","phase": "primary","corpus": "3-step",
     "model": "mistral-small",  "temp": 0.5,
     "path": OUTPUTS / "corpus_20260226_162106_mistral-small_trace_20260302_182442"},
    {"label": "Mistral Small\nT=0.5 3-step #2","group": "3step","phase": "primary","corpus": "3-step",
     "model": "mistral-small",  "temp": 0.5,
     "path": OUTPUTS / "corpus_20260226_162106_mistral-small_trace_20260302_182911"},
]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})
C_SS  = "#2196F3"   # two-step blue
C_3S  = "#FF5722"   # three-step orange
C_EXP = "#4CAF50"   # experienced green
C_NOV = "#9C27B0"   # novice purple


# ── Helpers ───────────────────────────────────────────────────────────────────
def _texts_match(a: str, b: str, min_len: int = 20) -> bool:
    a, b = a.strip().strip("'\""), b.strip().strip("'\"")
    return a == b or (len(a) >= min_len and (a in b or b in a))

def _bstrap_ci(values, n_boot: int = 10_000, alpha: float = 0.05):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, (n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))

def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    from scipy.stats import norm as _norm
    z = _norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)

def load_trace_metrics(run: dict) -> pd.DataFrame:
    records = []
    for meta_f in sorted(run["path"].glob("trace_transcript_*_eval.json")):
        d = json.loads(meta_f.read_text())
        m = d.get("metrics", {})
        if not m:
            continue
        fname = meta_f.stem
        parts = fname.replace("trace_transcript_", "").replace("_eval", "").split("_", 4)
        records.append({
            "label": run["label"], "corpus": run["corpus"],
            "model": run["model"], "temp": run["temp"],
            "f1": m.get("f1", float("nan")),
            "precision": m.get("precision", float("nan")),
            "recall": m.get("recall", float("nan")),
            "quality_attribute_accuracy": m.get("quality_attribute_accuracy", float("nan")),
            "extracted_count": m.get("extracted_count", float("nan")),
        })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Extraction Count vs. F1 and Precision scatter
# ════════════════════════════════════════════════════════════════════════════
def fig_extraction_count_scatter():
    print("Generating: thesis_fig_extraction_scatter.png")
    from scipy import stats as _stats

    frames = []
    for run in TRACE_MANIFEST:
        if run["corpus"] != "single-shot":
            continue
        df = load_trace_metrics(run)
        if not df.empty:
            frames.append(df)
    df_ss = pd.concat(frames, ignore_index=True)

    MODEL_LABELS = {
        "mistral-small":  "Mistral Small 3.2",
        "mistral-medium": "Mistral Medium",
        "gpt-5-mini":     "GPT-5-mini",
    }
    MODEL_COLORS = {
        "mistral-small":  C_SS,
        "mistral-medium": "#607D8B",
        "gpt-5-mini":     "#795548",
    }
    MODEL_MARKERS = {
        "mistral-small":  "o",
        "mistral-medium": "s",
        "gpt-5-mini":     "^",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, (y_col, y_label) in zip(axes, [("f1", "F1"), ("precision", "Precision")]):
        for model, grp in df_ss.groupby("model"):
            ax.scatter(
                grp["extracted_count"], grp[y_col],
                color=MODEL_COLORS[model],
                marker=MODEL_MARKERS[model],
                label=MODEL_LABELS[model],
                alpha=0.55, s=28, linewidths=0,
            )

        x_all = df_ss["extracted_count"].values
        y_all = df_ss[y_col].values
        mask  = ~(np.isnan(x_all) | np.isnan(y_all))
        slope, intercept, r, _, _ = _stats.linregress(x_all[mask], y_all[mask])
        xs = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.2, zorder=3)
        ax.text(
            0.97, 0.97, f"$r = {r:.3f}$",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="top",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )

        ax.set_xlabel("Extracted requirement count per transcript", fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.set_title(f"Extraction count vs. {y_label}", fontsize=9)
        ax.set_ylim(0, 1.05)

    handles, labels_leg = axes[0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels_leg):
        seen.setdefault(l, h)
    fig.legend(
        list(seen.values()), list(seen.keys()),
        loc="lower center", ncol=3, fontsize=8,
        bbox_to_anchor=(0.5, -0.07), frameon=False,
    )
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_extraction_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_extraction_scatter.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Confusion Matrix — two-step, row-normalised (thesis version)
# ════════════════════════════════════════════════════════════════════════════
def fig_confusion_matrix():
    print("Generating: thesis_fig_confusion_ss.png")
    _EXCLUDE = {"quality", "functional"}

    records = []
    primary_ss = [r for r in TRACE_MANIFEST if r["phase"] == "primary" and r["corpus"] == "single-shot"]
    for run in primary_ss:
        corpus_dir = run["path"]
        for eval_f in sorted(corpus_dir.glob("trace_transcript_*_eval.json")):
            eval_d  = json.loads(eval_f.read_text())
            stem    = eval_f.stem[:eval_f.stem.rfind("_eval")]
            xfile   = corpus_dir / f"{stem}.json"
            if not xfile.exists():
                continue
            extract_d = json.loads(xfile.read_text())
            gt_src = eval_d.get("ground_truth_source", "")
            m = re.search(r"transcript_[^\s/]+\.json", gt_src)
            if not m:
                continue
            sidecar = SQUIRE_SS / m.group()
            if not sidecar.exists():
                continue
            gt_reqs   = json.loads(sidecar.read_text()).get("selected_requirements", [])
            ext_reqs  = extract_d.get("requirements", [])
            for i, ev in enumerate(eval_d.get("per_requirement", [])):
                if not ev.get("is_hit"):
                    continue
                mgt   = (ev.get("matched_ground_truth") or "").strip().strip("'\"")
                gt_idx = next((j for j, g in enumerate(gt_reqs) if _texts_match(g["text"], mgt)), None)
                if gt_idx is None or i >= len(ext_reqs):
                    continue
                predicted = (
                    (ext_reqs[i].get("quality_attribute") or "unknown")
                    .lower().strip().replace(" ", "_").replace("-", "_")
                )
                for cat in gt_reqs[gt_idx].get("categories", []):
                    if cat in _EXCLUDE:
                        continue
                    records.append({"gt": cat, "pred": predicted})

    if not records:
        print("  No data — skipping confusion matrix")
        return

    df = pd.DataFrame(records)
    cats = sorted(df["gt"].unique())
    raw  = pd.DataFrame(sk_cm(df["gt"], df["pred"], labels=cats), index=cats, columns=cats)
    norm = raw.div(raw.sum(axis=1), axis=0).fillna(0)

    # Clean category labels
    label_map = {
        "availability": "Availability", "fault_tolerance": "Fault\nTolerance",
        "look_and_feel": "Look & Feel", "maintainability": "Maintainability",
        "operability": "Operability", "performance": "Performance",
        "portability": "Portability", "scalability": "Scalability",
        "security": "Security", "usability": "Usability", "legal": "Legal",
    }
    tick_labels = [label_map.get(c, c) for c in cats]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        norm, ax=ax, cmap="Blues", annot=True, fmt=".2f",
        linewidths=0.3, linecolor="#ddd",
        xticklabels=tick_labels, yticklabels=tick_labels,
        vmin=0, vmax=1,
        annot_kws={"size": 7},
        cbar_kws={"label": "Row-normalised fraction", "shrink": 0.8},
    )
    ax.set_xlabel("TRACE predicted attribute", fontsize=9)
    ax.set_ylabel("Ground-truth category", fontsize=9)
    ax.set_title(
        "Quality Attribute Classification — Row-Normalised Confusion Matrix\n"
        "Two-step corpus, Mistral Small T=0.5 (×3 runs pooled, hits only)",
        fontsize=9,
    )
    ax.tick_params(axis="x", rotation=40, labelsize=7.5)
    ax.tick_params(axis="y", rotation=0,  labelsize=7.5)
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_confusion_ss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_confusion_ss.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3: ISO Concordance — Jaccard per NICE label
# ════════════════════════════════════════════════════════════════════════════
def fig_iso_concordance():
    print("Generating: thesis_fig_iso_concordance.png")
    ISO_TAXONOMY = {
        "Functional suitability": ["Functional completeness", "Functional appropriateness", "Functional correctness"],
        "Performance efficiency": ["Time behaviour", "Resource utilization", "Capacity"],
        "Compatibility":          ["Co-existence", "Interoperability"],
        "Interaction capability": ["Appropriateness recognizability", "Learnability", "Operability",
                                   "User error protection", "User engagement", "Inclusivity",
                                   "User assistance", "Self descriptiveness"],
        "Reliability":            ["Faultlessness", "Availability", "Fault tolerance", "Recoverability"],
        "Security":               ["Confidentiality", "Integrity", "Non-repudiation",
                                   "Accountability", "Authenticity", "Resistance"],
        "Maintainability":        ["Modularity", "Reusability", "Analysability", "Modifiability", "Testability"],
        "Flexibility":            ["Adaptability", "Scalability", "Installability", "Replaceability"],
        "Safety":                 ["Operational constraint", "Risk identification", "Fail safe",
                                   "Hazard warning", "Safe integration"],
    }
    SUB_TO_CHAR = {sub: char for char, subs in ISO_TAXONOMY.items() for sub in subs}
    NICE_TO_ISO = {
        "Availability (A)":     ["Reliability"],
        "Fault Tolerance (FT)": ["Reliability", "Safety"],
        "Look & Feel (LF)":     ["Interaction capability"],
        "Maintainability (MN)": ["Maintainability"],
        "Operability (O)":      ["Interaction capability"],
        "Performance (PE)":     ["Performance efficiency"],
        "Portability (PO)":     ["Flexibility", "Compatibility"],
        "Scalability (SC)":     ["Flexibility"],
        "Security (SE)":        ["Security"],
        "Usability (US)":       ["Interaction capability"],
    }
    import glob as _glob
    run_files = sorted(_glob.glob(str(OUTPUTS / "iso_label_*/results.json")))
    if not run_files:
        print("  No ISO label runs found — skipping")
        return
    records = []
    for run_id, rf in enumerate(run_files, 1):
        for item in json.loads(Path(rf).read_text())["results"]:
            records.append({
                "run_id": run_id,
                "nice_labels": item["nice_labels"],
                "iso_sub_chars": item["iso_sub_chars"],
            })
    df = pd.DataFrame(records)

    def _score(row):
        pred = {SUB_TO_CHAR[s] for s in row["iso_sub_chars"] if s in SUB_TO_CHAR}
        exp  = {c for lbl in row["nice_labels"] for c in NICE_TO_ISO.get(lbl, [])}
        if not exp:
            return None, None
        inter = pred & exp; union = pred | exp
        j = len(inter) / len(union) if union else 0.0
        r = len(inter) / len(exp)
        return j, r

    df["jaccard"], df["recall"] = zip(*df.apply(_score, axis=1))
    df = df.dropna(subset=["jaccard"])

    agg = []
    for lbl in NICE_TO_ISO:
        mask = df["nice_labels"].apply(lambda ls: lbl in ls)
        sub = df[mask]["jaccard"].values
        if len(sub) == 0:
            continue
        mean_j = float(np.mean(sub))
        sd_j   = float(np.std(sub, ddof=1)) if len(sub) > 1 else 0.0
        ci95   = 1.96 * float(np.std(sub, ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else 0.0
        agg.append({"label": lbl, "mean_j": mean_j, "sd_j": sd_j, "ci95": ci95, "n": len(sub)})

    df_agg = pd.DataFrame(agg).sort_values("mean_j", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(df_agg))
    ax.bar(x, df_agg["mean_j"], color="#4472C4", alpha=0.85, width=0.6)
    ax.errorbar(x, df_agg["mean_j"], yerr=df_agg["sd_j"],
                fmt="none", ecolor="#333", elinewidth=1.2, capsize=4, capthick=1.2)
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--", alpha=0.7, label="J = 0.50 (midpoint)")
    ax.set_xticks(x)
    # Shorten label names
    short_labels = [l.split(" (")[0].replace(" & ", "/") for l in df_agg["label"]]
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean Jaccard (ISO characteristic level)")
    ax.set_title(
        "ISO/IEC 25010:2023 Re-labelling Concordance per NICE Label\n"
        "n=5 runs · Mistral Small 4 · T=0.5 · error bars = ±1 SD",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_iso_concordance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_iso_concordance.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4: By-project F1 — two-step vs three-step side by side
# ════════════════════════════════════════════════════════════════════════════
def fig_by_project():
    print("Generating: thesis_fig_project_f1.png")
    frames = []
    for run in TRACE_MANIFEST:
        if run["phase"] != "primary":
            continue
        df = load_trace_metrics(run)
        if not df.empty:
            frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)

    # Parse project_id from transcript name — load it from the eval files directly
    project_f1 = {corpus: {} for corpus in ["single-shot", "3-step"]}
    primary_runs = [r for r in TRACE_MANIFEST if r["phase"] == "primary"]
    for run in primary_runs:
        corpus_dir = run["path"]
        corpus     = run["corpus"]
        for eval_f in sorted(corpus_dir.glob("trace_transcript_*_eval.json")):
            m  = re.search(r"trace_transcript_(\d+)_", eval_f.name)
            if not m:
                continue
            pid = m.group(1)
            d = json.loads(eval_f.read_text())
            f1 = d.get("metrics", {}).get("f1", float("nan"))
            if not np.isnan(f1):
                project_f1[corpus].setdefault(pid, []).append(f1)

    _corpus_display = {"single-shot": "Two-step", "3-step": "Three-step"}
    proj_ids = sorted(PROJECT_NAMES.keys())
    x = np.arange(len(proj_ids))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for offset, corpus, color in [(-w/2, "single-shot", C_SS), (w/2, "3-step", C_3S)]:
        means, lo_errs, hi_errs = [], [], []
        for pid in proj_ids:
            vals = project_f1[corpus].get(pid, [])
            m = np.mean(vals) if vals else float("nan")
            lo, hi = _bstrap_ci(vals) if len(vals) > 1 else (m, m)
            means.append(m); lo_errs.append(m - lo); hi_errs.append(hi - m)
        ax.bar(x + offset, means, w, color=color, alpha=0.85, label=_corpus_display.get(corpus, corpus),
               yerr=[lo_errs, hi_errs], capsize=3, error_kw={"elinewidth": 1, "ecolor": "#333"})
    short_names = ["Nursing", "Credit Card", "Recycled\nParts", "Conf. Room", "IzognMovies"]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8.5)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("Mean F1 [95% bootstrap CI]")
    ax.set_title("TRACE F1 by Project Domain — Mistral Small T=0.5\n"
                 "(primary runs; two-step ×3, three-step ×2)", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_project_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_project_f1.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5: t-SNE Embedding — by Project and by Stakeholder (2×2)
# ════════════════════════════════════════════════════════════════════════════
def fig_tsne_embedding():
    print("Generating: thesis_fig_tsne_embedding.png")
    COORDS = Path("tsne_coords.npz")
    META   = Path("tsne_meta.json")
    if not COORDS.exists():
        print("  SKIP — tsne_coords.npz not found.")
        print("  Run cell 63 (tsne-save-coords) in thesis_analysis.ipynb first.")
        return

    data     = np.load(COORDS, allow_pickle=True)
    meta     = json.loads(META.read_text()) if META.exists() else {}

    X_ss     = data["X_tsne_ss"]
    X_3s     = data["X_tsne_3s"]
    pids_ss  = data["project_ids_ss"].astype(str)
    pids_3s  = data["project_ids_3s"].astype(str)
    stk_ss   = data["stakeholders_ss"].astype(str)
    stk_3s   = data["stakeholders_3s"].astype(str)

    # Okabe-Ito palette — colorblind-safe, high contrast
    OI_PROJECT    = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    OI_STAKEHOLDER= ["#0072B2", "#E69F00", "#009E73", "#D55E00"]

    STAKEHOLDER_LABELS = {
        "nontechnical_enduser": "Non-tech. End-user",
        "nontechnical_manager": "Non-tech. Manager",
        "technical_junior":     "Technical Junior",
        "technical_senior":     "Technical Senior",
    }
    PROJECT_SHORT = {
        "3": "Nursing Scheduling",
        "4": "Credit Card Disputes",
        "5": "Recycled Parts Mgmt",
        "6": "Conf. Room Mgmt",
        "8": "IzognMovies Streaming",
    }

    corpora = [
        ("Two-step",   X_ss, pids_ss, stk_ss, "ss"),
        ("Three-step", X_3s, pids_3s, stk_3s, "3s"),
    ]
    groupings = [
        ("project",     OI_PROJECT,     PROJECT_SHORT,     "project"),
        ("stakeholder", OI_STAKEHOLDER, STAKEHOLDER_LABELS, "stakeholder"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    for col, (corpus_name, X, pids, stakes, ckey) in enumerate(corpora):
        for row, (gname, palette, label_map, gkey) in enumerate(groupings):
            ax = axes[row, col]
            values = pids if gkey == "project" else stakes
            groups = sorted(set(values))
            colors = {g: palette[i] for i, g in enumerate(groups)}
            sil = meta.get(f"sil_{gname}_{ckey}") if gkey == "project" else None

            for g in groups:
                mask  = values == g
                ax.scatter(
                    X[mask, 0], X[mask, 1],
                    color=colors[g],
                    label=label_map.get(g, g),
                    s=52, alpha=0.88,
                    linewidths=0.4, edgecolors="white",
                )

            sil_str = f"  (sil = {sil:+.3f})" if sil is not None else ""
            ax.set_title(
                f"{corpus_name} — by {gname.capitalize()}{sil_str}",
                fontsize=9, pad=6,
            )
            ax.set_xlabel("t-SNE dim. 1", fontsize=8)
            ax.set_ylabel("t-SNE dim. 2", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(
                fontsize=7, loc="best", framealpha=0.45,
                markerscale=1.1, handlelength=1.0, borderpad=0.5,
            )

    plt.savefig(FIG_OUT / "thesis_fig_tsne_embedding.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_tsne_embedding.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6: G-Eval dimensions — two-step vs three-step grouped bar chart
# ════════════════════════════════════════════════════════════════════════════
def _load_squire_transcripts(corpus_dir: Path, pipeline_label: str) -> pd.DataFrame:
    records = []
    for f in sorted(corpus_dir.glob("transcript_*.json")):
        d = json.loads(f.read_text())
        ev = d["evaluation"]
        gq = ev["g_eval_quality"]["dimensions"]
        gm = ev["g_eval_meta"]["dimensions"]
        records.append({
            "pipeline":       pipeline_label,
            "structuring":    gq["structuring"]["mean"],
            "clarity":        gq["clarity"]["mean"],
            "responsiveness": gq["responsiveness"]["mean"],
            "rigor":          gq["rigor"]["mean"],
            "completeness":   gm["completeness"]["mean"],
            "realism":        gm["realism"]["mean"],
        })
    return pd.DataFrame(records)


def fig_geval_dimensions():
    print("Generating: thesis_fig_geval_dimensions.png")
    import matplotlib.ticker as _ticker

    df_ss = _load_squire_transcripts(SQUIRE_SS, "Two-step")
    df_3s = _load_squire_transcripts(SQUIRE_3S, "Three-step")
    df_squire = pd.concat([df_ss, df_3s], ignore_index=True)

    q_dims = ["structuring", "clarity", "responsiveness", "rigor"]
    m_dims = ["completeness", "realism"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    pipeline_colors = {"Two-step": C_SS, "Three-step": C_3S}

    for ax, (dims, title) in zip(axes, [
        (q_dims, "Quality G-Eval (blind)"),
        (m_dims, "Meta G-Eval (full context)"),
    ]):
        data = df_squire.groupby("pipeline")[dims].mean().reindex(["Two-step", "Three-step"])
        x = np.arange(len(dims))
        w = 0.35
        for i, (pipeline, row) in enumerate(data.iterrows()):
            offset = -w / 2 + i * w
            bars = ax.bar(
                x + offset, row.values, w,
                color=pipeline_colors[pipeline],
                alpha=0.85,
                label=pipeline,
            )
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels([d.capitalize() for d in dims], fontsize=8.5)
        ax.set_ylim(0, 11)
        ax.set_ylabel("Score (1–10)", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.yaxis.set_major_locator(_ticker.MultipleLocator(1))
        ax.legend(fontsize=8)

    plt.suptitle(
        "SQuIRE G-Eval Scores by Pipeline — Mistral Small T=0.5 (n=40 per pipeline)",
        fontsize=9, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_geval_dimensions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_geval_dimensions.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Recall gaps by quality attribute category
# ════════════════════════════════════════════════════════════════════════════
_FNAME_RE = re.compile(
    r"(?:trace_)?transcript_(\d+)_(re_(?:novice|experienced))_(stakeholder_\w+)_\d{8}_\d{6}"
)

def _load_recall_gaps(run: dict, squire_corpus_dir: Path) -> pd.DataFrame:
    _NON_SEMANTIC = {"quality", "functional"}
    records = []
    for eval_f in sorted(run["path"].glob("trace_transcript_*_eval.json")):
        eval_d = json.loads(eval_f.read_text())
        gt_source = eval_d.get("ground_truth_source", "")
        m = re.search(r"transcript_[^\s/]+\.json", gt_source)
        if not m:
            continue
        sidecar_path = squire_corpus_dir / m.group()
        if not sidecar_path.exists():
            continue
        sidecar = json.loads(sidecar_path.read_text())
        gt_reqs = sidecar.get("selected_requirements", [])
        matched_mgt_list = [
            req["matched_ground_truth"]
            for req in eval_d["per_requirement"]
            if req["is_hit"]
        ]
        for gt in gt_reqs:
            gt_text = gt["text"].strip("'\"")
            is_matched = any(_texts_match(gt_text, mgt) for mgt in matched_mgt_list)
            for cat in gt.get("categories", ["unknown"]):
                if cat not in _NON_SEMANTIC:
                    records.append({"category": cat, "is_matched": is_matched})
    return pd.DataFrame(records)


def fig_recall_gaps():
    print("Generating: thesis_fig_recall_gaps.png")
    t05_ss_runs = [r for r in TRACE_MANIFEST if r["model"] == "mistral-small" and r["temp"] == 0.5 and r["corpus"] == "single-shot"]
    t05_3s_runs = [r for r in TRACE_MANIFEST if r["model"] == "mistral-small" and r["temp"] == 0.5 and r["corpus"] == "3-step"]

    def _recall_table(runs, squire_dir):
        df = pd.concat([_load_recall_gaps(r, squire_dir) for r in runs], ignore_index=True)
        tbl = (
            df.groupby("category")["is_matched"]
            .agg(total="count", matched="sum")
            .assign(recall=lambda x: x["matched"] / x["total"])
        )
        tbl["ci_lo"] = tbl.apply(lambda r: _wilson_ci(int(r["matched"]), int(r["total"]))[0], axis=1)
        tbl["ci_hi"] = tbl.apply(lambda r: _wilson_ci(int(r["matched"]), int(r["total"]))[1], axis=1)
        return tbl.sort_values("recall")

    stats_ss = _recall_table(t05_ss_runs, SQUIRE_SS)
    stats_3s = _recall_table(t05_3s_runs, SQUIRE_3S)

    all_cats = stats_ss.index.union(stats_3s.index)
    order = stats_ss.reindex(all_cats)["recall"].sort_values().index

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (stats_df, label) in zip(axes, [(stats_ss, "Two-step"), (stats_3s, "Three-step")]):
        s = stats_df.reindex(order).dropna()
        colors = [
            "#c0392b" if r < 0.5 else ("#4c9c6b" if r > 0.7 else "steelblue")
            for r in s["recall"]
        ]
        ax.barh(s.index, s["recall"], color=colors)
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Recall (fraction of GT requirements retrieved)", fontsize=9)
        ax.set_title(
            f"Recall by Quality Attribute — {label}\n"
            f"Mistral Small T=0.5 ({'×3' if label == 'Two-step' else '×2'} runs pooled)",
            fontsize=9,
        )
        ax.set_xlim(0, 1.4)
        for i, (cat, row) in enumerate(s.iterrows()):
            ax.text(
                row["recall"] + 0.01, i,
                f"{row['recall']:.2f} [{row['ci_lo']:.2f}–{row['ci_hi']:.2f}] "
                f"({int(row['matched'])}/{int(row['total'])})",
                va="center", fontsize=7.5,
            )

    axes[0].set_ylabel("Quality attribute category", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_recall_gaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_recall_gaps.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Confidence calibration — 2-panel (two-step | three-step)
# ════════════════════════════════════════════════════════════════════════════
def _load_confidence_overlap(run: dict) -> pd.DataFrame:
    """Join extraction confidence with eval overlap_score by positional pairing."""
    import warnings
    records = []
    for ext_f in sorted(run["path"].glob("trace_transcript_*.json")):
        if "_eval" in ext_f.name:
            continue
        eval_f = ext_f.with_name(ext_f.stem + "_eval.json")
        if not eval_f.exists():
            continue
        ext_d  = json.loads(ext_f.read_text())
        eval_d = json.loads(eval_f.read_text())
        reqs_ext  = ext_d.get("requirements", [])
        reqs_eval = eval_d.get("per_requirement", [])
        if len(reqs_ext) != len(reqs_eval):
            warnings.warn(
                f"Length mismatch in {ext_f.name}: "
                f"extracted={len(reqs_ext)}, eval={len(reqs_eval)} — skipping",
                stacklevel=2,
            )
            continue
        for req_ext, req_eval in zip(reqs_ext, reqs_eval):
            records.append({
                "confidence":    req_ext.get("confidence", 3),
                "overlap_score": req_eval["overlap_score"],
                "is_hit":        req_eval["is_hit"],
            })
    return pd.DataFrame(records)


def fig_confidence_calibration():
    print("Generating: thesis_fig_confidence_calibration.png")
    t05_ss_runs = [r for r in TRACE_MANIFEST if r["model"] == "mistral-small" and r["temp"] == 0.5 and r["corpus"] == "single-shot"]
    t05_3s_runs = [r for r in TRACE_MANIFEST if r["model"] == "mistral-small" and r["temp"] == 0.5 and r["corpus"] == "3-step"]

    df_ss = pd.concat([_load_confidence_overlap(r) for r in t05_ss_runs], ignore_index=True)
    df_3s = pd.concat([_load_confidence_overlap(r) for r in t05_3s_runs], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (df, label, color) in zip(axes, [
        (df_ss, "Two-step",   C_SS),
        (df_3s, "Three-step", C_3S),
    ]):
        calib = df.groupby("confidence").agg(
            mean_overlap=("overlap_score", "mean"),
            hit_rate=("is_hit", "mean"),
            n=("is_hit", "count"),
        )
        calib = calib[calib["n"] >= 20]

        x_pos = np.arange(len(calib))
        bars = ax.bar(
            x_pos, calib["mean_overlap"],
            color=color, alpha=0.75, width=0.55,
        )

        # hit threshold line
        ax.axhline(3.0, color="black", linestyle=":", linewidth=1.2,
                   label="hit threshold (3.0)")
        # overall mean overlap
        overall_mean = df["overlap_score"].mean()
        ax.axhline(overall_mean, color="#555", linestyle="--", linewidth=1.2,
                   label=f"overall mean = {overall_mean:.2f}")

        # bar labels: mean value, n, hit rate
        for i, (conf_val, row) in enumerate(calib.iterrows()):
            ax.text(
                i, row["mean_overlap"] + 0.08,
                f"{row['mean_overlap']:.2f}\n(n={row['n']})",
                ha="center", va="bottom", fontsize=8,
            )
            ax.text(
                i, 1.1,
                f"hit rate: {row['hit_rate']:.0%}",
                ha="center", va="bottom", fontsize=7.5, color="#333",
                style="italic",
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(c) for c in calib.index], fontsize=9)
        ax.set_xlabel("Confidence assigned by Mistral Small (1–5)", fontsize=9)
        ax.set_ylabel("Mean Overlap Score (judge-assessed)", fontsize=9)
        ax.set_ylim(1, 5.5)
        ax.set_title(
            f"Confidence Calibration — {label}\n"
            f"Mistral Small T=0.5, n≥20 per group",
            fontsize=9,
        )
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_confidence_calibration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_confidence_calibration.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 9: t-SNE — three-step by Stakeholder (single panel, presentation)
# Shows both structure levels at once: spatial = project domain, colour =
# stakeholder persona.  Requires tsne_coords.npz / tsne_meta.json (cell 63).
# ════════════════════════════════════════════════════════════════════════════
def fig_tsne_3s_stakeholder():
    print("Generating: thesis_fig_tsne_3s_stakeholder.png")
    COORDS = Path("tsne_coords.npz")
    META   = Path("tsne_meta.json")
    if not COORDS.exists():
        print("  SKIP — tsne_coords.npz not found.")
        print("  Run cell 63 (tsne-save-coords) in thesis_analysis.ipynb first.")
        return

    data    = np.load(COORDS, allow_pickle=True)
    meta    = json.loads(META.read_text()) if META.exists() else {}

    X_3s    = data["X_tsne_3s"]
    stk_3s  = data["stakeholders_3s"].astype(str)

    OI_STAKEHOLDER = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
    STAKEHOLDER_LABELS = {
        "nontechnical_enduser": "Non-tech. End-user",
        "nontechnical_manager": "Non-tech. Manager",
        "technical_junior":     "Technical Junior",
        "technical_senior":     "Technical Senior",
    }

    fig, ax = plt.subplots(figsize=(5, 4.5))
    groups  = sorted(set(stk_3s))
    colors  = {g: OI_STAKEHOLDER[i] for i, g in enumerate(groups)}

    for g in groups:
        mask = stk_3s == g
        ax.scatter(
            X_3s[mask, 0], X_3s[mask, 1],
            color=colors[g],
            label=STAKEHOLDER_LABELS.get(g, g),
            s=52, alpha=0.88,
            linewidths=0.4, edgecolors="white",
        )

    ax.set_title("Three-step corpus -- by Stakeholder", fontsize=9, pad=6)
    ax.set_xlabel("t-SNE dim. 1", fontsize=8)
    ax.set_ylabel("t-SNE dim. 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(
        fontsize=7, loc="best", framealpha=0.45,
        markerscale=1.1, handlelength=1.0, borderpad=0.5,
    )

    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_tsne_3s_stakeholder.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_tsne_3s_stakeholder.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 10: Extraction Count vs. F1 — single panel (presentation)
# ════════════════════════════════════════════════════════════════════════════
def fig_extraction_scatter_f1():
    print("Generating: thesis_fig_extraction_scatter_f1.png")
    from scipy import stats as _stats

    frames = []
    for run in TRACE_MANIFEST:
        if run["corpus"] != "single-shot":
            continue
        df = load_trace_metrics(run)
        if not df.empty:
            frames.append(df)
    df_ss = pd.concat(frames, ignore_index=True)

    MODEL_LABELS = {
        "mistral-small":  "Mistral Small 3.2",
        "mistral-medium": "Mistral Medium",
        "gpt-5-mini":     "GPT-5-mini",
    }
    MODEL_COLORS = {
        "mistral-small":  C_SS,
        "mistral-medium": "#607D8B",
        "gpt-5-mini":     "#795548",
    }
    MODEL_MARKERS = {
        "mistral-small":  "o",
        "mistral-medium": "s",
        "gpt-5-mini":     "^",
    }

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for model, grp in df_ss.groupby("model"):
        ax.scatter(
            grp["extracted_count"], grp["f1"],
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            label=MODEL_LABELS[model],
            alpha=0.55, s=28, linewidths=0,
        )

    x_all = df_ss["extracted_count"].values
    y_all = df_ss["f1"].values
    mask  = ~(np.isnan(x_all) | np.isnan(y_all))
    slope, intercept, r, _, _ = _stats.linregress(x_all[mask], y_all[mask])
    xs = np.linspace(x_all[mask].min(), x_all[mask].max(), 200)
    ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.2, zorder=3)

    ax.set_xlabel("Extracted requirement count per transcript", fontsize=9)
    ax.set_ylabel("F1", fontsize=9)
    ax.set_title("Extraction count vs. F1", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, framealpha=0.45, loc="upper right")

    plt.tight_layout()
    plt.savefig(FIG_OUT / "thesis_fig_extraction_scatter_f1.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: thesis_fig_extraction_scatter_f1.png")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent)
    fig_extraction_count_scatter()
    fig_confusion_matrix()
    fig_iso_concordance()
    fig_by_project()
    fig_tsne_embedding()
    fig_geval_dimensions()
    fig_recall_gaps()
    fig_confidence_calibration()
    fig_tsne_3s_stakeholder()
    fig_extraction_scatter_f1()
    print("\nDone. Move PNG files to thesis/figures/.")
