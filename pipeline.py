#!/usr/bin/env python3
"""Penn LEAD T1w preprocessing + sample construction for BrainIAC-based ADHD severity prediction.

Stages (run individually or `all`):
  manifest    build per-visit table: T1w path, age, E-SWAN score, visit index, elapsed time
  preprocess  N4 -> skull-strip -> register -> resample to 96^3 -> normalize, with QC flags
  samples     build task-specific samples (concurrent / final / future) with participant-level folds

Usage:  python pipeline.py --config config.yaml all [--limit 5]
"""
from __future__ import annotations

import argparse
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("pennlead")
T1_RE = re.compile(r"(sub-[^_]+)_(ses-[^_]+)")


def _norm(prefix: str, x) -> str:
    s = str(x)
    return s if s.startswith(prefix) else prefix + s


# --------------------------------------------------------------------------- manifest
def build_manifest(cfg: dict) -> pd.DataFrame:
    root, out = Path(cfg["bids_root"]), Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(root.glob("sub-*/ses-*/anat/*_T1w.nii*")):
        m = T1_RE.search(f.name)
        if not m:
            continue
        run = re.search(r"_run-(\d+)", f.name)
        rows.append(dict(subject=m.group(1), session=m.group(2),
                         run=int(run.group(1)) if run else 1, path=str(f)))
    if not rows:
        raise SystemExit(f"No T1w scans found under {root}")
    scans = (pd.DataFrame(rows).sort_values(["subject", "session", "run"])
             .drop_duplicates(["subject", "session"], keep=cfg["run_policy"]))

    # age at scan
    a, ages = cfg["age"], []
    for p in root.glob(a["file_glob"]):
        df = pd.read_csv(p, sep="\t")
        subj = re.search(r"(sub-[^_/]+)", p.name).group(1)
        ages.append(pd.DataFrame({"subject": subj,
                                  "session": df[a["session_col"]].map(lambda s: _norm("ses-", s)),
                                  "age": pd.to_numeric(df[a["age_col"]], errors="coerce")}))
    if not ages:
        raise SystemExit("No sessions.tsv files matched age.file_glob")
    ages = pd.concat(ages)
    if a["units"] == "months":
        ages["age"] /= 12.0

    # E-SWAN ADHD score
    ph = cfg["phenotype"]
    p = pd.read_csv(ph["file"], sep=None, engine="python")
    sess = p[ph["session_col"]].astype(str).map(lambda s: ph["session_map"].get(s, s))
    pheno = pd.DataFrame({"subject": p[ph["subject_col"]].map(lambda s: _norm("sub-", s)),
                          "session": sess.map(lambda s: _norm("ses-", s)),
                          "eswan": pd.to_numeric(p[ph["score_col"]], errors="coerce")})

    df = scans.merge(ages, on=["subject", "session"], how="inner") \
              .merge(pheno, on=["subject", "session"], how="inner") \
              .dropna(subset=["age", "eswan"])
    n0 = df.subject.nunique()
    df = df.sort_values(["subject", "age"])
    df["visit_idx"] = df.groupby("subject").cumcount()
    if cfg.get("max_visits"):
        df = df[df.visit_idx < cfg["max_visits"]]
    keep = df.groupby("subject").visit_idx.count() >= cfg["min_visits"]
    df = df[df.subject.isin(keep[keep].index)].copy()
    df["years_since_prev"] = df.groupby("subject").age.diff().fillna(0.0)
    df["years_since_baseline"] = df.age - df.groupby("subject").age.transform("first")
    df.to_csv(out / "manifest.csv", index=False)
    log.info("manifest: %d/%d subjects with >=%d usable visits (%d scans)",
             df.subject.nunique(), n0, cfg["min_visits"], len(df))
    return df


# --------------------------------------------------------------------------- preprocessing
def _skullstrip(img, row, cfg, tmp: Path):
    import ants
    sk = cfg["skullstrip"]
    if sk["mode"] == "none":
        return img
    if sk["mode"] == "mask":
        hits = sorted(Path(sk["derivatives_root"]).glob(
            sk["mask_glob"].format(subject=row.subject, session=row.session)))
        if not hits:
            raise RuntimeError("derivative brain mask not found")
        mask = ants.resample_image_to_target(ants.image_read(str(hits[0])), img,
                                             interp_type="nearestNeighbor")
        mask = ants.threshold_image(mask, 0.5, 1e9, 1, 0)
        if mask.sum() == 0:
            raise RuntimeError("mask empty after resampling (physical-space mismatch?)")
        return img * mask
    if sk["mode"] == "hdbet":
        inp, outp = tmp / "in.nii.gz", tmp / "out.nii.gz"
        ants.image_write(img, str(inp))
        subprocess.run(sk["hdbet_cmd"].format(input=inp, output=outp).split(), check=True)
        return ants.image_read(str(outp))
    raise ValueError(f"unknown skullstrip mode {sk['mode']}")


def preprocess_one(row, cfg, template, out_path: Path) -> dict:
    import ants
    img = ants.image_read(row.path)
    if cfg["n4"]:
        img = ants.n4_bias_field_correction(img)
    with tempfile.TemporaryDirectory() as t:
        img = _skullstrip(img, row, cfg, Path(t))
    if template is not None:
        reg = ants.registration(fixed=template, moving=img,
                                type_of_transform=cfg["registration"]["transform"])
        img = reg["warpedmovout"]
    brain_mm3 = float((img.numpy() > 0).sum() * np.prod(img.spacing))
    img = ants.resample_image(img, tuple(cfg["target_shape"]), use_voxels=True, interp_type=4)
    arr = img.numpy().astype(np.float32)
    brain = arr > 0
    if brain.sum() < 1000:
        raise RuntimeError("almost no brain voxels after preprocessing")
    v = arr[brain]
    arr[brain] = (v - v.mean()) / (v.std() + 1e-8) if cfg["normalize"] == "zscore" \
        else (v - v.min()) / (v.max() - v.min() + 1e-8)
    arr[~brain] = 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(img.new_image_like(arr), str(out_path))
    return dict(brain_mm3=brain_mm3, brain_frac=float(brain.mean()))


def run_preprocess(cfg: dict, limit: int | None = None) -> pd.DataFrame:
    import ants
    out = Path(cfg["out_dir"])
    man = pd.read_csv(out / "manifest.csv")
    if limit:
        man = man[man.subject.isin(man.subject.unique()[:limit])]
    tpl_path = cfg["registration"]["template"]
    template = ants.image_read(tpl_path) if tpl_path else None
    if template is None:
        log.warning("No registration template set: scans stay in native space (BrainIAC expects MNI-like input).")

    recs = []
    for row in man.itertuples():
        dst = out / "preproc" / row.subject / row.session / \
            f"{row.subject}_{row.session}_desc-preproc_T1w.nii.gz"
        rec = dict(subject=row.subject, session=row.session, preproc_path=str(dst), qc_ok=True, error="")
        try:
            if dst.exists():                      # resume support
                rec.update(brain_mm3=np.nan, brain_frac=np.nan)
            else:
                rec.update(preprocess_one(row, cfg, template, dst))
        except Exception as e:                    # keep going; failures go to the QC table
            rec.update(qc_ok=False, error=str(e)[:200], brain_mm3=np.nan, brain_frac=np.nan)
            log.warning("%s %s failed: %s", row.subject, row.session, e)
        recs.append(rec)

    qc = pd.DataFrame(recs)
    z = (qc.brain_mm3 - qc.brain_mm3.mean()) / qc.brain_mm3.std()
    qc["brain_vol_outlier"] = z.abs() > 3         # flag for manual visual QC, not auto-dropped
    qc.to_csv(out / "qc.csv", index=False)
    merged = man.merge(qc, on=["subject", "session"])
    merged.to_csv(out / "manifest_preproc.csv", index=False)
    log.info("preprocess: %d ok, %d failed, %d outlier flags",
             qc.qc_ok.sum(), (~qc.qc_ok).sum(), qc.brain_vol_outlier.sum())
    return merged


# --------------------------------------------------------------------------- samples
def build_samples(cfg: dict) -> pd.DataFrame:
    out = Path(cfg["out_dir"])
    m = pd.read_csv(out / "manifest_preproc.csv")
    m = m[m.qc_ok].sort_values(["subject", "visit_idx"])
    if cfg["require_all_visits"]:
        cnt = m.groupby("subject").visit_idx.count()
        m = m[m.subject.isin(cnt[cnt >= cfg["min_visits"]].index)]

    rng = np.random.default_rng(cfg["seed"])
    subs = np.array(sorted(m.subject.unique()))
    rng.shuffle(subs)
    fold = {s: i % cfg["n_folds"] for i, s in enumerate(subs)}   # participant-level folds

    mc, rows = cfg["min_context"], []
    for s, g in m.groupby("subject"):
        g = g.reset_index(drop=True)
        n = len(g)
        plans = {"concurrent": [(list(range(k + 1)), k) for k in range(mc - 1, n)],
                 "final": [(list(range(n)), n - 1)],
                 "future": [(list(range(k + 1)), k + 1) for k in range(mc - 1, n - 1)]}
        for task in cfg["tasks"]:
            for inp, tgt in plans[task]:
                rows.append(dict(
                    sample_id=f"{s}_{task}_{inp[-1]}to{tgt}", subject=s, task=task, fold=fold[s],
                    input_sessions=";".join(g.session[inp]), n_inputs=len(inp),
                    target_session=g.session[tgt], target_visit_idx=tgt,
                    target_eswan=g.eswan[tgt],
                    age_at_target=g.age[tgt], age_at_last_input=g.age[inp[-1]],
                    horizon_years=g.age[tgt] - g.age[inp[-1]]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "samples.csv", index=False)
    log.info("samples:\n%s", df.groupby("task").agg(n=("sample_id", "size"),
                                                    subjects=("subject", "nunique")))
    return df


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("stage", choices=["manifest", "preprocess", "samples", "all"])
    ap.add_argument("--limit", type=int, help="preprocess only the first N subjects (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cfg = yaml.safe_load(open(args.config))
    if args.stage in ("manifest", "all"):
        build_manifest(cfg)
    if args.stage in ("preprocess", "all"):
        run_preprocess(cfg, args.limit)
    if args.stage in ("samples", "all"):
        build_samples(cfg)


if __name__ == "__main__":
    main()