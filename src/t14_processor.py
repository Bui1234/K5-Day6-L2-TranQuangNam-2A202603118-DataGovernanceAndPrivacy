"""T14 core algorithm (owner: Member 2 — Core Algorithm).

Called from src/pipeline.py so the shared pipeline file stays small and conflict-free.

    process()  : map model class_id -> taxonomy name (by NAME, not by id), clip boxes to the image,
                 per-class confidence filter (small sign/motorbike boxes get a lower threshold),
                 uncertainty score for every candidate box.
    rank()     : top-B review queue (overlap / small / borderline-conf boxes first) + HTML contact sheet.
    finalize() : keep only exported boxes, re-validate (invalid-label rate must be 0).
    export_all(): YOLO txt + COCO json + round-trip QA report.

Candidate table schema (returned by process(), one row per model box with conf >= review_floor):
    box_id, image_id, class, x, y, w, h, conf, model_class_id, area, keep, threshold,
    margin, small, overlap, overlap_class, score, reason, suggested_action, path
`keep` = would be exported without human review. Boxes with keep=False are still candidates
for review ("restore?") but are dropped by finalize() unless a reviewer sets keep=True.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .common.contact_sheet import render_contact_sheet
from .common.review import make_review_queue, write_review_queue
from .metrics.detection import iou_xywh

# ----------------------------------------------------------------------------- params
# Starting priors. Freeze the final values in configs/lab.yaml `params:` (chosen on DEV only)
# BEFORE running eval; every key below can be overridden there.
DEFAULTS: dict = {
    "conf_threshold": 0.25,          # fallback for classes not listed in class_conf_threshold
    "class_conf_threshold": {"car": 0.40, "pedestrian": 0.40, "motorbike": 0.40, "sign": 0.40},
    "small_area": 150.0,             # px², same definition as the evaluator's small-object slice
    "small_classes": ["sign", "motorbike"],
    "small_conf_threshold": 0.25,    # lower bar for small boxes of small_classes (model is under-confident on them)
    "review_floor": 0.10,            # boxes below this are not even review candidates
    "margin_band": 0.20,             # |conf - threshold| at which "borderline" score reaches 0
    "overlap_iou": 0.30,             # IoU from which two boxes count as overlapping
    "score_weights": {"margin": 0.45, "overlap": 0.30, "small": 0.25},
    "min_box_px": 1.0,               # boxes thinner than this after clipping are dropped
    "budget": 30,
}

# Model label names (from model_card label_map) -> project taxonomy names.
# Mapping goes through NAMES on purpose: model id 0 = person but taxonomy id 0 = car.
MODEL_NAME_TO_TAXONOMY = {
    "person": "pedestrian",
    "car": "car",
    "motorcycle": "motorbike",
    "stop sign": "sign",
}


class LabelMapError(ValueError):
    """Model class id cannot be mapped to the project taxonomy (fail loudly, never export it)."""


def get_params(ctx) -> dict:
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in ((ctx.config.get("params") or {}) if ctx is not None else {}).items():
        if isinstance(v, dict) and isinstance(p.get(k), dict):
            p[k].update(v)
        else:
            p[k] = v
    return p


# ----------------------------------------------------------------------------- 1. label map
def load_taxonomy(path: str | Path) -> dict[int, str]:
    return {int(k): str(v) for k, v in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["classes"].items()}


def load_model_label_map(path: str | Path) -> dict[int, str]:
    return {int(k): str(v) for k, v in yaml.safe_load(Path(path).read_text(encoding="utf-8"))["label_map"].items()}


def build_class_map(model_label_map: dict[int, str], taxonomy: dict[int, str],
                    name_map: dict[str, str] = MODEL_NAME_TO_TAXONOMY) -> dict[int, str]:
    """model class_id -> taxonomy class NAME. Validates every target is in the taxonomy."""
    tax_names = set(taxonomy.values())
    out = {}
    for mid, mname in model_label_map.items():
        tname = name_map.get(mname)
        if tname is None:
            raise LabelMapError(f"model label '{mname}' (id {mid}) has no taxonomy mapping")
        if tname not in tax_names:
            raise LabelMapError(f"mapped name '{tname}' for model id {mid} not in taxonomy {sorted(tax_names)}")
        out[int(mid)] = tname
    missing = tax_names - set(out.values())
    if missing:
        # not fatal (model may simply not cover a class) but worth knowing
        print(f"[t14] warning: taxonomy classes never predicted by the model: {sorted(missing)}")
    return out


def map_classes(pred: pd.DataFrame, class_map: dict[int, str]) -> pd.DataFrame:
    """Add `class` (taxonomy name) + keep the raw `model_class_id`. Unknown ids -> LabelMapError."""
    df = pred.copy()
    df["model_class_id"] = df["class_id"].astype(int)
    unknown = sorted(set(df["model_class_id"]) - set(class_map))
    if unknown:
        raise LabelMapError(f"prediction class_id(s) {unknown} not in model label_map {sorted(class_map)}")
    df["class"] = df["model_class_id"].map(class_map)
    return df.drop(columns=["class_id"])


# ----------------------------------------------------------------------------- 1b. clip to image
def clip_boxes(df: pd.DataFrame, sizes: pd.DataFrame, min_box_px: float = 1.0) -> tuple[pd.DataFrame, int, int]:
    """Clip x,y,w,h into [0,W]x[0,H] per image. Returns (clipped_df, n_clipped, n_dropped)."""
    d = df.merge(sizes[["image_id", "width", "height"]], on="image_id", how="left", validate="many_to_one")
    if d["width"].isna().any():
        bad = sorted(d.loc[d["width"].isna(), "image_id"].unique())
        raise ValueError(f"predictions reference images not in images.csv: {bad[:5]}")
    x1 = d["x"].clip(lower=0)
    y1 = d["y"].clip(lower=0)
    x2 = (d["x"] + d["w"]).clip(upper=d["width"])
    y2 = (d["y"] + d["h"]).clip(upper=d["height"])
    nx, ny, nw, nh = x1, y1, (x2 - x1), (y2 - y1)
    changed = ~(np.isclose(nx, d["x"]) & np.isclose(ny, d["y"]) & np.isclose(nw, d["w"]) & np.isclose(nh, d["h"]))
    d["x"], d["y"], d["w"], d["h"] = nx.round(2), ny.round(2), nw.round(2), nh.round(2)
    d["was_clipped"] = changed
    ok = (d["w"] >= min_box_px) & (d["h"] >= min_box_px)
    return d[ok].drop(columns=["width", "height"]).reset_index(drop=True), int(changed.sum()), int((~ok).sum())


def invalid_mask(df: pd.DataFrame, sizes: pd.DataFrame, taxonomy_names: set[str]) -> pd.Series:
    """Same definition as src/metrics/tracks.py::eval_t14 'Invalid-label rate'."""
    s = sizes.set_index("image_id")
    W, H = df["image_id"].map(s["width"]), df["image_id"].map(s["height"])
    return ((~df["class"].astype(str).isin(taxonomy_names)) | (df.w <= 0) | (df.h <= 0) | (df.x < 0) | (df.y < 0)
            | (df.x + df.w > W + 0.5) | (df.y + df.h > H + 0.5))


# ----------------------------------------------------------------------------- 2. filter + score
def class_threshold(cls: str, area: float, p: dict) -> float:
    if area < float(p["small_area"]) and cls in set(p["small_classes"]):
        return float(p["small_conf_threshold"])
    return float(p["class_conf_threshold"].get(cls, p["conf_threshold"]))


def overlap_features(df: pd.DataFrame, iou_thr: float) -> pd.DataFrame:
    """Per box: max IoU with any other box in the same image, and whether that partner has another class."""
    max_iou = np.zeros(len(df))
    other_cls = np.zeros(len(df), dtype=bool)
    for _, g in df.groupby("image_id"):
        idx = g.index.to_list()
        boxes = g[["x", "y", "w", "h"]].to_numpy()
        cls = g["class"].to_numpy()
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                v = iou_xywh(boxes[a], boxes[b])
                for i, j in ((a, b), (b, a)):
                    if v > max_iou[idx[i]]:
                        max_iou[idx[i]] = v
                        other_cls[idx[i]] = cls[i] != cls[j]
    out = pd.DataFrame({"overlap": max_iou, "overlap_class": other_cls}, index=df.index)
    out.loc[out["overlap"] < iou_thr, "overlap_class"] = False
    return out


def score_candidates(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    """score in [0,1], higher = review first. Components:
        margin  : 1 at the class threshold, 0 when |conf - thr| >= margin_band  (borderline confidence)
        overlap : max IoU with another box (duplicate or class conflict, e.g. motorbike vs pedestrian)
        small   : 1 for area <= small_area, linearly to 0 at 3.3x small_area (small objects are often missed/mislocated)
    """
    d = df.reset_index(drop=True).copy()
    d["area"] = (d.w * d.h).round(2)
    d["threshold"] = [class_threshold(c, a, p) for c, a in zip(d["class"], d["area"])]
    d["keep"] = d["conf"] >= d["threshold"]
    band = float(p["margin_band"])
    d["margin"] = (1 - ((d["conf"] - d["threshold"]).abs() / band)).clip(0, 1)
    sa = float(p["small_area"])
    d["small"] = ((sa * 10 / 3 - d["area"]) / (sa * 10 / 3 - sa)).clip(0, 1)
    d = d.join(overlap_features(d, float(p["overlap_iou"])))
    ov = d["overlap"].where(d["overlap"] >= float(p["overlap_iou"]), 0.0)
    w = p["score_weights"]
    d["score"] = (w["margin"] * d["margin"] + w["overlap"] * ov + w["small"] * d["small"]).round(6)

    def why(r) -> str:
        tags = []
        if r.margin > 0:
            tags.append(f"borderline_conf({r.conf:.2f} vs thr {r.threshold:.2f})")
        if r.overlap >= float(p["overlap_iou"]):
            tags.append(("class_conflict" if r.overlap_class else "duplicate") + f"(iou {r.overlap:.2f})")
        if r.area < sa:
            tags.append(f"small({r.area:.0f}px2)")
        if r.was_clipped:
            tags.append("clipped_to_image")
        return "; ".join(tags) or "low_priority"

    d["reason"] = d.apply(why, axis=1)
    d["suggested_action"] = np.where(d["keep"], "approve | delete | edit class/box", "restore? (below threshold)")
    return d


def process(ctx, data: dict | None = None) -> pd.DataFrame:
    """Returns the full candidate table (see module docstring). Reads inputs itself if `data` lacks them."""
    data = data or {}
    p = get_params(ctx)
    d = ctx.input_dir
    images = data.get("images") if data.get("images") is not None else pd.read_csv(d / "images.csv")
    pred = data.get("predictions") if data.get("predictions") is not None else pd.read_csv(d / "model_predictions_fixture.csv")
    taxonomy = load_taxonomy(d / "taxonomy.yaml")
    class_map = build_class_map(load_model_label_map(d / "model_card_fixture.yaml"), taxonomy)

    df = map_classes(pred, class_map)
    df.insert(0, "box_id", [f"{i}#{n:03d}" for n, i in enumerate(df["image_id"])])
    df, n_clipped, n_dropped_tiny = clip_boxes(df, images, float(p["min_box_px"]))
    n_floor = int((df["conf"] < float(p["review_floor"])).sum())
    df = df[df["conf"] >= float(p["review_floor"])]
    df = score_candidates(df, p)
    df = df.merge(images[["image_id", "path"]], on="image_id", how="left")

    ctx.stats.update({
        "t14_class_map": {str(k): v for k, v in class_map.items()},
        "t14_n_raw_boxes": int(len(pred)), "t14_n_clipped": n_clipped, "t14_n_dropped_tiny": n_dropped_tiny,
        "t14_n_below_review_floor": n_floor, "t14_n_candidates": int(len(df)), "t14_n_keep_auto": int(df["keep"].sum()),
        "t14_params": {k: v for k, v in p.items()},
    })
    if ctx.audit:
        ctx.audit.log(actor="pipeline", action="t14_process", item_id=str(d),
                      after={k: v for k, v in ctx.stats.items() if k.startswith("t14_n_")})
    cols = ["box_id", "image_id", "class", "x", "y", "w", "h", "conf", "model_class_id", "area", "keep", "threshold",
            "margin", "small", "overlap", "overlap_class", "was_clipped", "score", "reason", "suggested_action", "path"]
    return df[cols]


# ----------------------------------------------------------------------------- 3. rank
def rank(ctx, scored: pd.DataFrame) -> pd.DataFrame:
    """Top-B review queue by score (ties: lower conf first) + contact sheet with the box drawn."""
    p = get_params(ctx)
    budget = int(p["budget"])
    s = scored.sort_values(["score", "conf"], ascending=[False, True], kind="mergesort")
    s = s.assign(slice=lambda x: x["class"] + np.where(x["area"] < float(p["small_area"]), "/small", "")
                 + np.where(x["keep"], "/kept", "/dropped"))
    q = make_review_queue(s, id_col="box_id", score_col="score", budget=budget, reason_col="reason",
                          action_col="suggested_action", slice_col="slice", evidence_col="path")
    write_review_queue(q, ctx.out("review_queue.csv"))

    by_id = scored.set_index("box_id")
    boxes = {}
    for bid in q["item_id"]:
        r = by_id.loc[bid]
        others = scored[(scored.image_id == r.image_id) & (scored.box_id != bid) & scored.keep]
        boxes[bid] = [{"x": o.x, "y": o.y, "w": o.w, "h": o.h, "color": "gray"} for o in others.itertuples()] + [
            {"x": r.x, "y": r.y, "w": r.w, "h": r.h, "label": f"{r['class']} {r.conf:.2f}", "color": "red"}]
    render_contact_sheet(q, ctx.out("contact_sheet.html"), image_root=ctx.input_dir, boxes_by_item=boxes,
                         title=f"T14 review queue — top {budget} uncertain boxes (red = box under review)")
    ctx.stats["t14_queue_kept"] = int(scored.set_index("box_id").loc[q["item_id"], "keep"].sum())
    return q


# ----------------------------------------------------------------------------- 4. finalize + export
EXPORT_COLS = ["image_id", "class", "x", "y", "w", "h", "conf"]


def finalize(ctx, final: pd.DataFrame) -> pd.DataFrame:
    """Keep exported boxes only (keep==True after review) and refuse to export invalid labels."""
    out = final[final["keep"].astype(bool)] if "keep" in final.columns else final
    images = pd.read_csv(ctx.input_dir / "images.csv")
    tax = set(load_taxonomy(ctx.input_dir / "taxonomy.yaml").values())
    out, _, _ = clip_boxes(out.drop(columns=["was_clipped"], errors="ignore"), images)   # edits may push boxes out
    bad = invalid_mask(out, images, tax)
    if bad.any():
        raise ValueError(f"export blocked: {int(bad.sum())} invalid labels, e.g. {out[bad].head(3).to_dict('records')}")
    ctx.stats["t14_n_exported"] = int(len(out))
    rest = [c for c in out.columns if c not in EXPORT_COLS]
    return out[EXPORT_COLS + rest].reset_index(drop=True)


def to_yolo_lines(df: pd.DataFrame, W: float, H: float, name_to_id: dict[str, int]) -> list[str]:
    return [f"{name_to_id[r['class']]} {(r.x + r.w / 2) / W:.6f} {(r.y + r.h / 2) / H:.6f} {r.w / W:.6f} {r.h / H:.6f}"
            for _, r in df.iterrows()]


def from_yolo_lines(lines: list[str], W: float, H: float, id_to_name: dict[int, str]) -> list[dict]:
    out = []
    for ln in lines:
        c, cx, cy, w, h = ln.split()
        w, h = float(w) * W, float(h) * H
        out.append({"class": id_to_name[int(c)], "x": float(cx) * W - w / 2, "y": float(cy) * H - h / 2, "w": w, "h": h})
    return out


def to_coco(df: pd.DataFrame, images: pd.DataFrame, taxonomy: dict[int, str]) -> dict:
    """COCO category_id = taxonomy id + 1 (COCO ids are 1-based; 0 is often treated as background)."""
    name_to_cat = {n: i + 1 for i, n in taxonomy.items()}
    img_ids = {iid: k + 1 for k, iid in enumerate(images["image_id"])}
    return {
        "info": {"description": "T14 auto-annotation export (pre-labels from SIMULATED fixture predictions)"},
        "images": [{"id": img_ids[r.image_id], "file_name": r.path, "width": int(r.width), "height": int(r.height)}
                   for r in images.itertuples()],
        "categories": [{"id": name_to_cat[n], "name": n} for _, n in sorted(taxonomy.items())],
        "annotations": [{"id": k + 1, "image_id": img_ids[r.image_id], "category_id": name_to_cat[r["class"]],
                         "bbox": [round(float(r.x), 2), round(float(r.y), 2), round(float(r.w), 2), round(float(r.h), 2)],
                         "area": round(float(r.w * r.h), 2), "iscrowd": 0, "score": float(r.get("conf", 1.0))}
                        for k, (_, r) in enumerate(df.iterrows())],
    }


def export_all(ctx, final: pd.DataFrame, subdir: str = "export") -> dict:
    """Write YOLO (labels/*.txt + classes.txt) and COCO (annotations.json) and round-trip QA them."""
    images = pd.read_csv(ctx.input_dir / "images.csv")
    taxonomy = load_taxonomy(ctx.input_dir / "taxonomy.yaml")
    name_to_id = {n: i for i, n in taxonomy.items()}
    root = ctx.out(subdir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    (root / "classes.txt").write_text("\n".join(taxonomy[i] for i in sorted(taxonomy)) + "\n", encoding="utf-8")

    sizes = images.set_index("image_id")
    max_err, n_rt = 0.0, 0
    for iid in images["image_id"]:                       # one file per image, empty file = no objects
        sub = final[final.image_id == iid]
        W, H = float(sizes.at[iid, "width"]), float(sizes.at[iid, "height"])
        lines = to_yolo_lines(sub, W, H, name_to_id)
        (root / "labels" / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        back = from_yolo_lines(lines, W, H, taxonomy)
        for (_, a), b in zip(sub.iterrows(), back):
            if a["class"] != b["class"]:
                raise ValueError(f"YOLO round-trip class mismatch on {iid}: {a['class']} != {b['class']}")
            max_err = max(max_err, *(abs(float(a[k]) - b[k]) for k in ("x", "y", "w", "h")))
            n_rt += 1

    coco = to_coco(final, images, taxonomy)
    (root / "annotations_coco.json").write_text(json.dumps(coco, indent=1), encoding="utf-8")
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    coco_classes = sorted(cat_name[a["category_id"]] for a in coco["annotations"])
    qa = {
        "n_boxes": int(len(final)), "n_images": int(len(images)),
        "n_images_with_boxes": int(final["image_id"].nunique()),
        "per_class": final["class"].value_counts().to_dict(),
        "yolo_roundtrip_boxes": n_rt, "yolo_roundtrip_max_abs_err_px": round(max_err, 4),
        "coco_n_annotations": len(coco["annotations"]),
        "coco_class_counts_match": coco_classes == sorted(final["class"].tolist()),
        "pass": n_rt == len(final) and max_err < 0.01 and len(coco["annotations"]) == len(final)
                and coco_classes == sorted(final["class"].tolist()),
    }
    (root / "qa_report.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if not qa["pass"]:
        raise ValueError(f"export QA failed: {qa}")
    ctx.stats["t14_export_qa"] = qa
    return qa
