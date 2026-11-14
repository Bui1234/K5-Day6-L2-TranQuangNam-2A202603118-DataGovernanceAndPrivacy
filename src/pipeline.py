"""YOUR PoC PIPELINE — this is the file your group owns.

Lab contract = 5 automated steps, run by `python -m src.main ...`:
    01 load      input + provenance check (manifest.csv)
    02 process   your algorithm / model (the research part)
    03 rank      prioritise for a fixed review budget B
    04 review    human-in-the-loop: queue -> reviewer decisions -> overrides + audit log
    05 decide    GO / HOLD / REWORK (or track equivalent) with reason_code + owner, write outputs

Required output: out/<pred_file> with the columns listed for your track in
docs/tracks/TXX.md (also in src/metrics/tracks.py::TRACKS). Then evaluate with
    python -m src.evaluate --track tXX --pred out/<pred_file>

Rules: never read fixtures/*/gt/ here (the GT guard will stop you); seed everything via ctx.rng;
no network calls; fail loudly on missing models instead of downloading.
See src/examples/random_baseline.py for a minimal end-to-end wiring example.
"""
from __future__ import annotations

import pandas as pd
import yaml

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import load_review_decisions, make_review_queue, write_review_queue
from .metrics.tracks import get_track


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict:
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        if ctx.audit is not None:
            ctx.audit.log(actor="pipeline", action="provenance_problems", item_id=str(manifest), after=problems[:20])
        if ctx.config.get("strict_provenance", True):
            raise ValueError(f"provenance check failed ({len(problems)} problems), first: {problems[:3]}")

    data: dict = {"manifest": pd.read_csv(manifest) if manifest.exists() else None}

    images_path = ctx.input_dir / "images.csv"
    if images_path.exists():
        data["images"] = pd.read_csv(images_path)
        ctx.stats["n_images"] = len(data["images"])

    if ctx.track == "t14":
        taxonomy_path = ctx.input_dir / "taxonomy.yaml"
        model_card_path = ctx.input_dir / "model_card_fixture.yaml"
        preds_path = ctx.input_dir / "model_predictions_fixture.csv"

        for req_path in (images_path, taxonomy_path, model_card_path, preds_path):
            if not req_path.exists():
                raise FileNotFoundError(f"T14 load(): missing required input file {req_path}")

        data["taxonomy"] = yaml.safe_load(taxonomy_path.read_text(encoding="utf-8")) or {}
        data["model_card"] = yaml.safe_load(model_card_path.read_text(encoding="utf-8")) or {}
        data["predictions"] = pd.read_csv(preds_path)

        ctx.stats["n_raw_predictions"] = len(data["predictions"])
        ctx.stats["model_name"] = data["model_card"].get("model_name", "unknown")
        ctx.stats["model_version"] = data["model_card"].get("version", "unknown")
        ctx.stats["taxonomy_classes"] = list((data["taxonomy"].get("classes") or {}).values())

        if ctx.audit is not None:
            ctx.audit.log(
                actor="pipeline",
                action="load_t14_inputs",
                item_id=str(ctx.input_dir),
                after={
                    "n_images": ctx.stats.get("n_images", 0),
                    "n_raw_predictions": ctx.stats["n_raw_predictions"],
                    "model_name": ctx.stats["model_name"],
                    "model_version": ctx.stats["model_version"],
                    "taxonomy_classes": ctx.stats["taxonomy_classes"],
                },
            )

    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict) -> pd.DataFrame:
    if ctx.track == "t14":
        try:
            from . import t14_processor
            return t14_processor.process(ctx, data)
        except ImportError:
            pass
    spec = get_track(ctx.track)
    raise NotImplementedError(
        f"Implement process() for {spec.track} ({spec.title}). It must produce a DataFrame with at least "
        f"{spec.pred_columns} (saved as out/{spec.pred_file}). See docs/tracks/{spec.track.upper()}.md"
    )


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Default: top-B by `score`. Adapt id/score/reason columns to your track."""
    if ctx.track == "t14":
        try:
            from . import t14_processor
            if hasattr(t14_processor, "rank"):
                return t14_processor.rank(ctx, scored)
        except ImportError:
            pass
    budget = int(ctx.param("budget", 20))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column (higher = review first), "
                         "or adapt rank() to your track")
    q = make_review_queue(scored, id_col=scored.columns[0], score_col="score", budget=budget,
                          reason_col="reason" if "reason" in scored.columns else None)
    write_review_queue(q, ctx.out("review_queue.csv"))
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """If a reviewed queue exists (out/review_queue_reviewed.csv), apply it. Otherwise pass through."""
    if ctx.track == "t14":
        try:
            from . import t14_reviewer
            if hasattr(t14_reviewer, "review"):
                return t14_reviewer.review(ctx, scored, queue)
        except ImportError:
            pass
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if reviewed_path.exists():
        reviewed = load_review_decisions(reviewed_path)
        ctx.stats["n_reviewed"] = len(reviewed)
    return scored


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    if ctx.track == "t14":
        try:
            from . import t14_reviewer
            if hasattr(t14_reviewer, "decide"):
                return t14_reviewer.decide(ctx, final)
        except ImportError:
            pass
    return Decision(status="HOLD", reason_code="POLICY_NOT_IMPLEMENTED", reason="Decision policy not implemented yet",
                    owner=ctx.config.get("owner", "TODO-owner"), scope=ctx.track)


def run(ctx: RunContext) -> None:
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    if ctx.track == "t14":
        from . import t14_processor
        final = t14_processor.finalize(ctx, final)   # keep==True only + block invalid labels
        t14_processor.export_all(ctx, final)         # out/<run>/export/: YOLO + COCO + qa_report.json
    final[spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
