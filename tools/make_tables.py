"""The tables of the paper from the results of run_cell.py, fuse.py, bootstrap.py and the benchmarks.

    python tools/make_tables.py --out tables

Writes one LaTeX fragment per table and facts.txt with every number the text quotes. Gains
are AP differences to the single view r0 in percentage points; a dataset value is the mean
over its cells (detectors x families), and Mean averages the datasets with equal weight.
Tables whose inputs are missing are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asa.config import Experiments  # noqa: E402

FAMS = ["d2", "c4", "d4", "ms"]
FAM_LABEL = {"flip": "Flip", "d2": "D2", "c4": "C4", "d4": "D4", "ms": "MS"}
METHODS = [("wbf_max", "WBF, max score"), ("wbf_orig", "WBF, original score"), ("gbdt", "GBDT re-scoring"),
           ("obbstacking", "OBBStacking"), ("obbstacking_wbfbox", "OBBStacking score, WBF box"),
           ("asa_score", r"\method{}, score only"), ("asa", r"\method{}")]
ORDER = "nsmlxo"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def num(x, sign=True):
    if x is None:
        return "--"
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "$-$")


def same(v, b):
    return v is not None and b is not None and f"{v:.2f}" == f"{b:.2f}"


def fmt_row(label, vals, best, sign=True):
    cells = [r"\textbf{" + num(v, sign) + "}" if same(v, b) else num(v, sign) for v, b in zip(vals, best)]
    return label + " & " + " & ".join(cells) + r" \\"


def column_best(rows):
    return [max((v[j] for _, v in rows if v[j] is not None), default=None) for j in range(len(rows[0][1]))]


def tabular(spec, head, body):
    return "\n".join([r"\begin{tabular}{" + spec + "}", r"\toprule", *head, r"\midrule", *body,
                      r"\bottomrule", r"\end{tabular}"]) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments.json")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()
    ex = Experiments(a.config)
    R, out = Path(a.results), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    facts = []
    dsets = list(ex.datasets)
    short = {d: ex.datasets[d]["name"].split("-")[0] if d != "dior" else "DIOR-R" for d in dsets}

    def dets_of(d):
        return sorted(ex.detectors_of(d), key=lambda x: ORDER.index(ex.detectors[x]["detector"][-1]
                                                                     if ex.detectors[x]["detector"].startswith("YOLO") else "o"))

    def load(det, name):
        p = R / det / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    cells = {(d, f): load(d, f) for d in ex.detectors for f in FAMS + ["flip"]}
    r0 = {d: load(d, "r0") for d in ex.detectors}

    def gain(det, fam, m, metric="mAP50"):
        c = cells.get((det, fam))
        if not c or m not in c or not r0.get(det):
            return None
        return c[m][metric] - r0[det][metric]

    def ds_mean(d, m, metric="mAP50", fams=FAMS):
        return mean([mean([gain(x, f, m, metric) for f in fams]) for x in dets_of(d)])

    # ---- main table (Table 4)
    rows = []
    for m, lab in METHODS:
        per = [ds_mean(d, m) for d in dsets]
        rows.append((lab, per + [mean(per), mean([ds_mean(d, m, "AP75") for d in dsets]),
                                 mean([ds_mean(d, m, "mAP50_95") for d in dsets])]))
        facts.append(f"main {m}: AP50 " + " ".join(num(x) for x in per) + f" | mean {num(rows[-1][1][4])}"
                     f" AP75 {num(rows[-1][1][5])} AP50:95 {num(rows[-1][1][6])}")
    if any(v is not None for _, r in rows for v in r):
        best = column_best(rows)
        r0row = [mean([r0[x]["mAP50"] for x in dets_of(d) if r0.get(x)]) for d in dsets]
        r0row += [mean(r0row), mean([mean([r0[x]["AP75"] for x in dets_of(d) if r0.get(x)]) for d in dsets]),
                  mean([mean([r0[x]["mAP50_95"] for x in dets_of(d) if r0.get(x)]) for d in dsets])]
        head = [r"& \multicolumn{5}{c}{\ap{50}} & \ap{75} & \ap{50:95} \\",
                r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}\cmidrule(lr){8-8}",
                "Method & " + " & ".join(short[d] for d in dsets) + r" & Mean & Mean & Mean \\"]
        body = ["Single view $r_0$ (absolute) & " + " & ".join(num(x, False) for x in r0row) + r" \\", r"\midrule"]
        body += [fmt_row(l, v, best) for l, v in rows[:2]] + [r"\midrule"] + [fmt_row(l, v, best) for l, v in rows[2:]]
        (out / "tab_main.tex").write_text(tabular("lrrrrrrr", head, body), encoding="utf-8")

    # ---- bootstrap (Table 5)
    bp = R / "bootstrap_contrasts.json"
    if bp.exists():
        boot = json.loads(bp.read_text())
        head = [r"& & & \multicolumn{3}{c}{\method{} $-$ WBF, original score} "
                r"& \multicolumn{3}{c}{\method{} $-$ OBBStacking (WBF box)} \\",
                r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
                r"Dataset & Cells & Units & $\Delta$\ap{50} & 95\% CI & $+$/$-$ & $\Delta$\ap{50} & 95\% CI & $+$/$-$ \\"]
        body = []
        for d in dsets:
            if d not in boot:
                continue
            b = boot[d]
            cols = [ex.datasets[d]["name"], str(b["cells"]), f"{b['units']} {b['unit']}s"]
            for key in ("asa_vs_wbf_orig", "asa_vs_obbstacking_wbfbox"):
                s = b[key]
                cols += [num(s["mean_diff"]), "[" + num(s["ci95"][0]) + ", " + num(s["ci95"][1]) + "]",
                         f"{s['cells_ci_above_0']}/{s['cells_ci_below_0']}"]
                facts.append(f"boot {d} {key}: {num(s['mean_diff'])} [{num(s['ci95'][0])}, {num(s['ci95'][1])}] "
                             f"cells CI>0 {s['cells_ci_above_0']} <0 {s['cells_ci_below_0']} of {b['cells']}")
            body.append(" & ".join(cols) + r" \\")
        (out / "tab_ci.tex").write_text(tabular("lrlrlrrlr", head, body), encoding="utf-8")

    # ---- per detector (Table 6)
    dm = [("wbf_max", "WBF max"), ("wbf_orig", "WBF orig."), ("gbdt", "GBDT"), ("obbstacking_wbfbox", "OBBS"),
          ("asa", r"\method{}")]
    body = []
    for i, d in enumerate(dsets):
        if i:
            body.append(r"\midrule")
        for j, x in enumerate(dets_of(d)):
            vals = [mean([gain(x, f, m) for f in FAMS]) for m, _ in dm]
            if r0.get(x) is None or all(v is None for v in vals):
                continue
            b = max(v for v in vals if v is not None)
            cs = [r"\textbf{" + num(v) + "}" if same(v, b) else num(v) for v in vals]
            dev = r"$^\dagger$" if x in ex.development else ""
            body.append(" & ".join([ex.datasets[d]["name"] if j == 0 else "", ex.detectors[x]["detector"] + dev,
                                    num(r0[x]["mAP50"], False), *cs]) + r" \\")
            if vals[1] is not None and vals[-1] is not None:
                facts.append(f"detector {x}: ASA - WBF(orig) {num(vals[-1] - vals[1])}")
    head = [r"Dataset & Detector & $r_0$ & " + " & ".join(l for _, l in dm) + r" \\"]
    (out / "tab_detectors.tex").write_text(tabular("llr" + "r" * len(dm), head, body), encoding="utf-8")

    # ---- per family (Table 7)
    fm = [("wbf_max", "WBF max"), ("wbf_orig", "WBF orig."), ("obbstacking_wbfbox", "OBBS"), ("asa", r"\method{}")]
    head = [r"& \multicolumn{4}{c}{Mean over datasets} & \multicolumn{4}{c}{FAIR1M-2.0} \\",
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
            "Family & " + " & ".join(l for _, l in fm) + " & " + " & ".join(l for _, l in fm) + r" \\"]
    body = []
    for f in FAMS:
        allm = [mean([ds_mean(d, m, "mAP50", [f]) for d in dsets]) for m, _ in fm]
        fair = [ds_mean("fair", m, "mAP50", [f]) if "fair" in ex.datasets else None for m, _ in fm]
        b1, b2 = max([v for v in allm if v is not None], default=None), max([v for v in fair if v is not None], default=None)
        cs = [r"\textbf{" + num(v) + "}" if same(v, b1) else num(v) for v in allm]
        cs += [r"\textbf{" + num(v) + "}" if same(v, b2) else num(v) for v in fair]
        body.append(FAM_LABEL[f] + " & " + " & ".join(cs) + r" \\")
        facts.append(f"family {f}: mean " + " ".join(f"{m} {num(v)}" for (m, _), v in zip(fm, allm))
                     + " | FAIR " + " ".join(f"{m} {num(v)}" for (m, _), v in zip(fm, fair)))
    (out / "tab_families.tex").write_text(tabular("lrrrrrrrr", head, body), encoding="utf-8")

    # ---- C4 with NMS and Soft-NMS (Table 8)
    cm = [("nms", "Rotated NMS"), ("softnms", "Soft-NMS"), ("wbf_max", "WBF, max score"),
          ("wbf_orig", "WBF, original score"), ("obbstacking_wbfbox", "OBBStacking score, WBF box"), ("asa", r"\method{}")]
    rows = []
    for m, lab in cm:
        v = [ds_mean(d, m, "mAP50", ["c4"]) for d in dsets]
        rows.append((lab, v + [mean(v), mean([ds_mean(d, m, "AP75", ["c4"]) for d in dsets])]))
    best = column_best(rows)
    head = [r"& \multicolumn{5}{c}{\ap{50}} & \ap{75} \\", r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}",
            "Fusion & " + " & ".join(short[d] for d in dsets) + r" & Mean & Mean \\"]
    (out / "tab_c4.tex").write_text(tabular("lrrrrrr", head, [fmt_row(l, v, best) for l, v in rows]), encoding="utf-8")

    # ---- ablations on the development detectors (Table 9)
    abl = [("wbf_orig", "WBF, original score (the prior)"), ("asa", r"\method{}"),
           ("asa_score", r"\quad without learned box weights"), ("asa_no_penalty", r"\quad without the penalty on $\delta$"),
           ("asa_no_prior", r"\quad without the WBF prior"), ("asa_no_members", r"\quad without the member encoder"),
           ("wbf_orig_gaussbox", r"WBF, original score, Gaussian box average")]
    devs = {ex.dataset_of(x): x for x in ex.development}
    rows = []
    for m, lab in abl:
        per = [mean([gain(devs[d], f, m) for f in FAMS]) if d in devs else None for d in dsets]
        p75 = [mean([gain(devs[d], f, m, "AP75") for f in FAMS]) if d in devs else None for d in dsets]
        rows.append((lab, per + [mean(per), mean(p75)]))
        facts.append(f"ablation {m}: AP50 " + " ".join(num(x) for x in per) + f" mean {num(mean(per))} | AP75 mean {num(mean(p75))}")
    best = column_best(rows)
    head = [r"& \multicolumn{5}{c}{\ap{50}} & \ap{75} \\", r"\cmidrule(lr){2-6}\cmidrule(lr){7-7}",
            "Variant & " + " & ".join(short[d] for d in dsets) + r" & Mean & Mean \\"]
    body = [fmt_row(l, v, best) for l, v in rows[:1]] + [r"\midrule"] + [fmt_row(l, v, best) for l, v in rows[1:6]]
    body += [r"\midrule"] + [fmt_row(l, v, best) for l, v in rows[6:]]
    (out / "tab_ablation.tex").write_text(tabular("lrrrrrr", head, body), encoding="utf-8")

    # ---- fixed-rule TTA against r0 (Table 1)
    vp = R / "bootstrap_vs_r0.json"
    if vp.exists():
        vs = json.loads(vp.read_text())
        mf = [("flip", 2), ("d2", 4), ("c4", 4), ("d4", 8), ("ms", 3)]
        per_det = {x: v for d in vs.values() for x, v in d["detectors"].items()}
        body = []
        for f, passes in mf:
            gains = {x: v[f]["diff"] for x, v in per_det.items() if f in v}
            sig = [sum(v[f]["ci95"][0] > 0 for v in per_det.values()), 0, sum(v[f]["ci95"][1] < 0 for v in per_det.values())]
            sig[1] = len(per_det) - sig[0] - sig[2]
            bestc = sum(max((g for g, _ in mf if g in v), key=lambda g: v[g]["diff"]) == f for v in per_det.values())
            dmean = mean([mean([gains[x] for x in dets_of(d) if x in gains]) for d in dsets])
            xs = list(gains.values())
            body.append(" & ".join([FAM_LABEL[f], str(passes), num(dmean), "[" + num(min(xs)) + ", " + num(max(xs)) + "]",
                                    "/".join(map(str, sig)), str(bestc)]) + r" \\")
            facts.append(f"motivation {f}: mean {num(dmean)} range [{num(min(xs))}, {num(max(xs))}] +/0/- {sig} best for {bestc}")
        head = [r"Family & Passes & Mean & Range & $+$/0/$-$ & Best for \\"]
        (out / "tab_motivation.tex").write_text(tabular("lcrlcc", head, body), encoding="utf-8")

    # ---- cost (Tables 10 and 11)
    lat = sorted((R / "latency").glob("views_*.json")) if (R / "latency").exists() else []
    if lat:
        L = sorted((json.loads(p.read_text()) for p in lat),
                   key=lambda x: (-x["imgsz"], "nsmlx".index(Path(x["model"]).name.split("-")[0][6:])))
        head = [r"Family & Passes & " + " & ".join(f"YOLO11{Path(x['model']).name.split('-')[0][6:]}, {x['imgsz']}" for x in L) + r" \\"]
        body = [r"$r_0$ & 1 & " + " & ".join(f"{1000 * x['families']['r0']['views_s']:.1f} ms" for x in L) + r" \\", r"\midrule"]
        for f in ("d2", "c4", "d4", "ms"):
            body.append(f"{FAM_LABEL[f]} & {L[0]['families'][f]['passes']} & "
                        + " & ".join(f"{x['families'][f]['views_rel']:.2f}$\\times$" for x in L) + r" \\")
        (out / "tab_views_cost.tex").write_text(tabular("lr" + "r" * len(L), head, body), encoding="utf-8")
    agg = sorted((R / "latency").glob("aggregator_*.json")) if (R / "latency").exists() else []
    if agg:
        head = [r"Dataset & Family & Clusters & WBF & \method{} features & \method{} network & Overhead \\"]
        body = []
        rows = [json.loads(p.read_text()) for p in agg]
        rows.sort(key=lambda x: (dsets.index(ex.dataset_of(x["detector"])), FAMS.index(x["family"])))
        for x in rows:
            ms = x["ms_per_image"]
            net = ms["tensors"] + ms["network"]
            body.append(" & ".join([short[ex.dataset_of(x["detector"])], FAM_LABEL[x["family"]],
                                    f"{x['clusters'] / x['images']:.0f}", f"{ms['wbf']:.1f}", f"{ms['features']:.1f}",
                                    f"{net:.1f}", f"{ms['features'] + net - ms['wbf']:+.1f}"]) + r" \\")
        (out / "tab_asa_cost.tex").write_text(tabular("llrrrrr", head, body), encoding="utf-8")

    (out / "facts.txt").write_text("\n".join(facts) + "\n", encoding="utf-8")
    print("\n".join(facts))


if __name__ == "__main__":
    main()
