"""Tests for src/t14_processor.py (Member 2 — Core Algorithm). No ground truth needed."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import t14_processor as t14
from src.common.context import RunContext

FX = Path("fixtures/t14_auto_annotation")


def _ctx(tmp_path, params=None):
    return RunContext(track="t14", input_dir=FX, output_dir=tmp_path, config={"params": params or {}},
                      seed=42, rng=np.random.default_rng(42))


def test_class_map_goes_through_names_not_ids():
    cm = t14.build_class_map(t14.load_model_label_map(FX / "model_card_fixture.yaml"),
                             t14.load_taxonomy(FX / "taxonomy.yaml"))
    assert cm == {0: "pedestrian", 2: "car", 3: "motorbike", 11: "sign"}
    # the trap: taxonomy id 0 is car, model id 0 is person
    assert cm[0] != t14.load_taxonomy(FX / "taxonomy.yaml")[0]


def test_unknown_class_id_fails_loudly():
    pred = pd.DataFrame({"image_id": ["aa_00"], "class_id": [7], "x": [1], "y": [1], "w": [5], "h": [5], "conf": [0.9]})
    with pytest.raises(t14.LabelMapError):
        t14.map_classes(pred, {0: "pedestrian", 2: "car"})


def test_clip_boxes_into_image_and_drop_degenerate():
    sizes = pd.DataFrame({"image_id": ["a"], "width": [320], "height": [180]})
    df = pd.DataFrame({"image_id": ["a"] * 3, "class": ["car"] * 3,
                       "x": [-0.48, 300.0, 330.0], "y": [105.6, 150.0, 10.0], "w": [12.42, 40.0, 5.0], "h": [50.09, 40.0, 5.0]})
    out, n_clipped, n_dropped = t14.clip_boxes(df, sizes)
    assert n_dropped == 1 and n_clipped == 3
    assert (out.x >= 0).all() and (out.y >= 0).all()
    assert (out.x + out.w <= 320).all() and (out.y + out.h <= 180).all()


def test_small_sign_gets_lower_threshold_but_small_car_does_not():
    p = t14.get_params(None)
    assert t14.class_threshold("sign", 100, p) == p["small_conf_threshold"]
    assert t14.class_threshold("motorbike", 100, p) == p["small_conf_threshold"]
    assert t14.class_threshold("car", 100, p) == p["class_conf_threshold"]["car"]
    assert t14.class_threshold("sign", 1000, p) == p["class_conf_threshold"]["sign"]


def test_process_rank_export_end_to_end(tmp_path):
    ctx = _ctx(tmp_path)
    cand = t14.process(ctx, {})
    assert set(cand["class"]) <= {"car", "pedestrian", "motorbike", "sign"}
    assert cand["score"].between(0, 1).all()
    q = t14.rank(ctx, cand)
    assert len(q) == 30 and q["score"].is_monotonic_decreasing
    final = t14.finalize(ctx, cand)
    assert final["keep"].all() and len(final) == cand["keep"].sum()
    images = pd.read_csv(FX / "images.csv")
    assert not t14.invalid_mask(final, images, {"car", "pedestrian", "motorbike", "sign"}).any()
    qa = t14.export_all(ctx, final)
    assert qa["pass"] and len(list((tmp_path / "export" / "labels").glob("*.txt"))) == len(images)


def test_finalize_blocks_unknown_class(tmp_path):
    ctx = _ctx(tmp_path)
    bad = pd.DataFrame({"image_id": ["aa_00"], "class": ["truck"], "x": [1.0], "y": [1.0], "w": [5.0], "h": [5.0],
                        "conf": [0.9], "keep": [True]})
    with pytest.raises(ValueError, match="export blocked"):
        t14.finalize(ctx, bad)
