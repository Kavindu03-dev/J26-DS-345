"""Evaluate existing route-based restaurant recommendation; never score reviews.

Requires the production dependencies, existing datasets and local OSRM. Run
without concurrent planner jobs because production stages use shared files.
Production ranks within each day. The pooled top 10 is an evaluation-only
ordering of existing production scores; restaurant/day rows can repeat venues.
"""

from contextlib import contextmanager
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "Data/processed"
INPUT = ROOT / "Data/input/c2_c3_input.json"
ROUTES = PROCESSED / "route_geometries"
RESULTS = Path(__file__).resolve().parent / "results"
LOG = RESULTS / "restaurant_evaluation.log"
DATASET = PROCESSED / "restaurants_osm_with_districts.csv"
FILTERED = PROCESSED / "restaurants_near_routes.csv"
DETOURS = PROCESSED / "restaurant_detour_scores.csv"
RANKED = PROCESSED / "final_ranked_restaurants.csv"
ABSA = ROOT / "ABSA/output/restaurant_aspect_scores.csv"
ASPECTS = [f"{aspect}_score" for aspect in ("food", "service", "ambience", "value", "location")]
OUTPUTS = [
    PROCESSED / "poi_preference_scores.csv",
    PROCESSED / "optimized_itinerary.csv",
    PROCESSED / "route_geometry_summary.csv", FILTERED, DETOURS, RANKED,
]
STAGES = [
    "src/optimization/02_score_pois.py",
    "src/optimization/04_optimize_multi_day.py",
    "src/routing/03_build_route_geometry.py",
    "src/restaurants/01_filter_restaurants_near_route.py",
    "src/restaurants/02_compute_restaurant_detours.py",
    "src/restaurants/06_rank_restaurants_with_absa.py",
]


def kandy_request():
    # Reuse the established control configuration, without executing its main().
    path = Path(__file__).with_name("02_evaluate_medical_support.py")
    spec = importlib.util.spec_from_file_location("medical_evaluation_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = copy.deepcopy(module.BASE_REQUEST)
    request["medical_support"] = {"required": False, "priority": "normal"}
    return request


@contextmanager
def preserved_production_files(emit):
    original_input = INPUT.read_bytes()
    paths = [*OUTPUTS, *(p for p in ROUTES.glob("*") if p.is_file())]
    backup = {p: p.read_bytes() if p.exists() else None for p in paths}
    try:
        # Prevent stale outputs and extra historical day routes entering this run.
        for path in [*OUTPUTS, *ROUTES.glob("day_*_route.geojson")]:
            path.unlink(missing_ok=True)
        yield
    finally:
        failures = []
        for path, contents in {INPUT: original_input, **backup}.items():
            try:
                if contents is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(contents)
                actual = path.read_bytes() if path.exists() else None
                if actual != contents:
                    failures.append(str(path))
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        for path in ROUTES.glob("*"):
            if path.is_file() and path not in backup:
                try:
                    path.unlink()
                except OSError as exc:
                    failures.append(f"{path}: {exc}")
        if failures:
            raise RuntimeError("Restoration failed: " + "; ".join(failures))
        emit("Input and generated production files restored: YES (byte-for-byte verified)")


def run_stage(stage, log):
    print(f"Running {stage}", flush=True)
    log.write(f"\n$ {sys.executable} {stage}\n")
    log.flush()
    result = subprocess.run(
        [sys.executable, "-u", str(ROOT / stage)], cwd=ROOT,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    log.write(result.stdout + "\n")
    log.flush()
    if result.returncode:
        raise RuntimeError(f"{stage} exited {result.returncode}:\n{result.stdout[-4000:]}")


def evaluate(dataset, filtered, detours, ranked, absa):
    if dataset.osm_id.isna().any() or dataset.osm_id.duplicated().any():
        raise ValueError("Restaurant dataset must have unique, nonmissing osm_id values")
    if ranked.empty or ranked.duplicated(["day", "osm_id"]).any():
        raise ValueError("Ranked output must have unique restaurant/day rows and be nonempty")
    for frame in (filtered, detours, ranked):
        if not set(frame.osm_id).issubset(set(dataset.osm_id)):
            raise ValueError("Output contains restaurant IDs outside the production dataset")
    if not set(zip(detours.day, detours.osm_id)).issubset(set(zip(filtered.day, filtered.osm_id))):
        raise ValueError("Detour rows do not match route candidates")
    if set(zip(ranked.day, ranked.osm_id)) != set(zip(detours.day, detours.osm_id)):
        raise ValueError("Ranked and detour candidate sets differ")
    ranked = ranked.copy()
    evidence = ranked.weighted_absa_score.notna()
    if not ranked.has_absa_data.eq(evidence).all():
        raise ValueError("Production ABSA evidence flags disagree with its weighted score")
    fallback = ranked.scoring_mode.eq("cuisine+detour")
    if not fallback.eq(~evidence).all():
        raise ValueError("Production fallback modes disagree with available review evidence")
    # Check exported aspects against existing evidence, including missing values.
    if absa.osm_id.duplicated().any():
        raise ValueError("ABSA source has duplicate restaurant IDs")
    source = absa.set_index("osm_id")
    for column in ASPECTS:
        expected = ranked.osm_id.map(source[column])
        actual = ranked[column]
        if not (actual.eq(expected) | (actual.isna() & expected.isna())).all():
            raise ValueError(f"Production aspect evidence differs from source: {column}")
    # Retain production scores and daily ranks. This is not a ranking formula.
    pooled = ranked.sort_values(
        ["final_restaurant_score", "detour_time_min", "day", "final_day_rank"],
        ascending=[False, True, True, True], kind="stable",
    ).copy()
    pooled.insert(0, "evaluation_rank", range(1, len(pooled) + 1))
    # Compare within the same daily candidate set. Equal detours share minimum rank.
    pooled["detour_only_day_rank"] = pooled.groupby("day").detour_time_min.rank(method="min")
    pooled["rank_shift_vs_detour_only"] = pooled.detour_only_day_rank - pooled.final_day_rank
    top = pooled.head(10).copy()
    valid = detours.detour_time_min[np.isfinite(detours.detour_time_min) & (detours.detour_time_min >= 0)]
    if len(valid) != len(detours) or not len(valid):
        raise ValueError("Production detour output contains missing/invalid times or is empty")
    count, inside = len(dataset), filtered.osm_id.nunique()
    reviewed = pooled[pooled.weighted_absa_score.notna()]
    fallback_rows = pooled[pooled.scoring_mode.eq("cuisine+detour")]
    summary = {
        "region": "Kandy",
        "total_restaurants_available_before_route_filter": count,
        "restaurants_inside_route_corridor": inside,
        "route_filter_reduction_count": count - inside,
        "route_filter_reduction_percent": 100 * (count - inside) / count,
        "route_filtered_restaurant_day_rows": len(filtered),
        "route_corridor_radius_m": 1000,
        "candidates_with_valid_detour": len(valid),
        "minimum_detour_min": valid.min(), "median_detour_min": valid.median(),
        "mean_detour_min": valid.mean(), "maximum_detour_min": valid.max(),
        "total_ranked_restaurants": len(pooled),
        "unique_ranked_restaurants": pooled.osm_id.nunique(),
        "count_with_absa_evidence": len(reviewed),
        "unique_restaurants_with_absa_evidence": reviewed.osm_id.nunique(),
        "count_using_cuisine_detour_fallback": len(fallback_rows),
        "percentage_with_absa_evidence": 100 * len(reviewed) / len(pooled),
        "percentage_fallback": 100 * len(fallback_rows) / len(pooled),
        "average_detour_among_top10": top.detour_time_min.mean(),
        "average_final_score_among_top10": top.final_restaurant_score.mean(),
        "best_ranked_restaurant_with_absa_evidence": reviewed.iloc[0]["name"] if len(reviewed) else None,
        "best_ranked_fallback_restaurant": fallback_rows.iloc[0]["name"] if len(fallback_rows) else None,
        # Equal-detour tie ordering alone is not evidence of an ordering effect.
        "ordering_differs_from_detour_only": any(
            group.sort_values("final_day_rank").detour_time_min.diff().lt(0).any()
            for _, group in pooled.groupby("day")
        ),
        "reviewed_restaurant_day_rows_with_rank_shift": int((reviewed.rank_shift_vs_detour_only != 0).sum()),
    }
    banana_ids = set(absa.loc[absa.restaurant_name.str.contains("banana chill", case=False, na=False), "osm_id"])
    banana_mask = pooled.osm_id.isin(banana_ids) | pooled.name.str.contains("banana chill", case=False, na=False)
    banana = pooled[banana_mask]
    summary["banana_chill_route_candidate_rows"] = int((filtered.osm_id.isin(banana_ids) | filtered.name.str.contains("banana chill", case=False, na=False)).sum())
    summary["banana_chill_ranked_rows"] = len(banana)
    summary["source_absa_restaurants"] = absa.osm_id.nunique()
    return summary, top, pooled, reviewed, banana


def print_summary(summary, top, reviewed, banana, emit):
    emit("\nROUTE FILTERING")
    for key in list(summary)[1:7]:
        emit(f"{key}: {summary[key]}")
    emit("\nDETOUR ANALYSIS (restaurant/day candidates; production detour_time_min)")
    for key in ("candidates_with_valid_detour", "minimum_detour_min", "median_detour_min", "mean_detour_min", "maximum_detour_min"):
        emit(f"{key}: {summary[key]:.4f}")
    emit("\nFINAL RANKING (pooled evaluation top 10; production daily ranks retained)")
    display = ["evaluation_rank", "day", "final_day_rank", "name", "cuisine", "final_restaurant_score", "detour_time_min", "scoring_mode"]
    emit(top[display].to_string(index=False))
    emit(f"Top-10 average detour: {summary['average_detour_among_top10']:.4f} min; average final score: {summary['average_final_score_among_top10']:.4f}")
    emit("\nABSA VS FALLBACK")
    emit(f"Ranked restaurant/day rows: {summary['total_ranked_restaurants']}; unique restaurants: {summary['unique_ranked_restaurants']}")
    emit(f"Review evidence available: {summary['count_with_absa_evidence']} ({summary['percentage_with_absa_evidence']:.2f}%); fallback: {summary['count_using_cuisine_detour_fallback']} ({summary['percentage_fallback']:.2f}%)")
    emit(f"Best pooled ABSA restaurant: {summary['best_ranked_restaurant_with_absa_evidence']}; best pooled fallback: {summary['best_ranked_fallback_restaurant']}")
    emit(f"Production ordering differs from detour-only ordering (strict reversals, excluding ties): {summary['ordering_differs_from_detour_only']}")
    if len(banana):
        emit("Banana Chill (identified by source ABSA osm_id/name):")
        emit(banana[[*display, "weighted_absa_score", *ASPECTS]].to_string(index=False))
    else:
        emit(f"Banana Chill absent from final ranking; route candidate rows: {summary['banana_chill_route_candidate_rows']}. It was not forced into the route.")
    if len(reviewed):
        emit("Reviewed-restaurant daily rank comparison (positive shift = higher final rank):")
        emit(reviewed[["day", "name", "final_day_rank", "detour_only_day_rank", "rank_shift_vs_detour_only"]].to_string(index=False))
    emit("\nLIMITATIONS")
    emit("Filtering reduction uses unique restaurant IDs; detour and ranking counts use restaurant/day rows.")
    emit("Top 10 pools existing scores across days and may repeat venues; production ranks only within each day.")
    emit("Detour-only ties share minimum rank; rank shifts describe the combined ABSA/cuisine/convenience ranking, not an isolated ABSA causal effect.")
    if summary["unique_restaurants_with_absa_evidence"] < 2:
        emit("Fewer than two unique reviewed restaurants occur on this route; reviewed-restaurant ordering effects cannot be established.")
    emit("Missing aspects remain NaN; review coverage is limited to the existing ABSA file. No reviews were generated, scraped, or retrained.")
    emit(f"Existing ABSA source contains {summary['source_absa_restaurants']} unique reviewed restaurant(s).")
    emit("Detours are modeled road insertion times, exclude meal duration, and use production-rounded minute values.")
    emit("Route-filter reduction includes production coordinate cleaning; unnamed candidates are removed by the production detour stage.")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as log:
        def emit(message):
            print(message, flush=True)
            log.write(message + "\n")
            log.flush()
        try:
            emit("route-based restaurant recommendation evaluation / ABSA-aware ranking")
            try:
                response = requests.get("http://localhost:5000/nearest/v1/driving/80.641486,7.294373", timeout=10)
                response.raise_for_status()
                if response.json().get("code") != "Ok":
                    raise RuntimeError("OSRM did not return Ok")
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                raise RuntimeError("OSRM unavailable at http://localhost:5000; evaluation stopped before modifying production files") from exc
            emit("OSRM connectivity: PASS")
            with preserved_production_files(emit):
                request = kandy_request()
                INPUT.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
                emit("Kandy request:\n" + json.dumps(request, indent=2))
                for stage in STAGES:
                    run_stage(stage, log)
                frames = [pd.read_csv(p) for p in (DATASET, FILTERED, DETOURS, RANKED, ABSA)]
                summary, top, pooled, reviewed, banana = evaluate(*frames)
                itinerary = pd.read_csv(PROCESSED / "optimized_itinerary.csv")
                selected = itinerary[itinerary.poi_id != "HOTEL"]
                summary.update(selected_pois=len(selected), preference_utility=selected.preference_score.sum())
                # Preserve actual production outputs before restoring shared paths.
                artifacts = RESULTS / "restaurant_evaluation_artifacts"
                artifacts.mkdir(exist_ok=True)
                for path in [*OUTPUTS, *ROUTES.glob("day_*_route.geojson"), ROUTES / "all_days_routes.geojson"]:
                    (artifacts / path.name).write_bytes(path.read_bytes())
                (artifacts / "kandy_request.json").write_text(json.dumps(request, indent=2), encoding="utf-8")
                (artifacts / ABSA.name).write_bytes(ABSA.read_bytes())
            pd.DataFrame([summary]).to_csv(RESULTS / "restaurant_evaluation_summary.csv", index=False)
            top.to_csv(RESULTS / "restaurant_top10_evaluation.csv", index=False)
            pooled.to_csv(RESULTS / "restaurant_ranking_comparison.csv", index=False)
            print_summary(summary, top, reviewed, banana, emit)
            emit(f"\nEvaluation outputs: {RESULTS}")
            emit("Status: SUCCESS")
        except Exception as exc:
            emit(f"Status: FAILED - {exc}")
            raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
