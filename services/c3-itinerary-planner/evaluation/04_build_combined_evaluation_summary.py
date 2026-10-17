"""Build reporting artifacts from approved regional figures and existing CSVs.

Read-only with respect to production data and prior evaluations. Regional
figures below are the user's approved results, because no regional result CSV
is present. Medical and restaurant evidence is read from the existing results.
No evaluation, optimizer, routing, ranking or model stage is executed.
"""

import hashlib
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


RESULTS = Path(__file__).resolve().parent / "results"
SOURCES = [
    RESULTS / "medical_support_evaluation.csv",
    RESULTS / "restaurant_evaluation_summary.csv",
    RESULTS / "restaurant_top10_evaluation.csv",
    RESULTS / "restaurant_ranking_comparison.csv",
]
REGIONAL_COLUMNS = [
    "region", "method", "pois_visited", "preference_utility",
    "road_distance_km", "driving_time_min",
]
# Approved valid three-region evidence supplied with the reporting request.
APPROVED_REGIONAL_ROWS = [
    ("Colombo", "greedy", 16, 80, 19.92, 35.59),
    ("Colombo", "optimizer", 16, 80, 15.92, 28.00),
    ("Kandy", "greedy", 14, 70, 140.16, 245.80),
    ("Kandy", "optimizer", 15, 75, 130.29, 245.00),
    ("Ella", "greedy", 14, 70, 98.32, 153.79),
    ("Ella", "optimizer", 15, 75, 111.70, 159.00),
]
MEDICAL_COLUMNS = {
    "scenario": "scenario", "pois_visited": "total_pois_visited",
    "preference_utility": "total_preference_utility",
    "road_distance_km": "total_road_distance_km",
    "driving_time_min": "total_driving_time_min",
    "visit_time_min": "total_visit_time_min",
    "avg_hospital_access_min": "average_hospital_access_min",
    "best_hospital_access_min": "best_hospital_access_min",
    "worst_hospital_access_min": "worst_hospital_access_min",
}
RESTAURANT_COLUMNS = {
    "total_restaurants": "total_restaurants_available_before_route_filter",
    "route_corridor_restaurants": "restaurants_inside_route_corridor",
    "route_filter_reduction": "route_filter_reduction_count",
    "route_filter_reduction_percent": "route_filter_reduction_percent",
    "restaurant_day_candidates": "route_filtered_restaurant_day_rows",
    "valid_detour_rows": "candidates_with_valid_detour",
    "min_detour_min": "minimum_detour_min", "median_detour_min": "median_detour_min",
    "mean_detour_min": "mean_detour_min", "max_detour_min": "maximum_detour_min",
    "ranked_rows": "total_ranked_restaurants",
    "unique_ranked_restaurants": "unique_ranked_restaurants",
    "absa_rows": "count_with_absa_evidence",
    "fallback_rows": "count_using_cuisine_detour_fallback",
    "absa_percent": "percentage_with_absa_evidence",
    "fallback_percent": "percentage_fallback",
}
BANANA_COLUMNS = {
    "day": "day", "production_rank": "final_day_rank",
    "detour_only_rank": "detour_only_day_rank", "final_score": "final_restaurant_score",
    "detour_min": "detour_time_min", "scoring_mode": "scoring_mode",
    "weighted_absa": "weighted_absa_score", "food": "food_score",
    "service": "service_score", "ambience": "ambience_score",
    "value": "value_score", "location": "location_score",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_columns(frame, mapping):
    return frame[list(mapping.values())].rename(columns={v: k for k, v in mapping.items()})


def close(actual, expected):
    return np.isclose(actual, expected, rtol=0, atol=1e-6)


def load_evidence():
    medical, restaurant, top, comparison = [pd.read_csv(p) for p in SOURCES]
    require(len(medical) == 2 and not medical.scenario.duplicated().any(), "Require exactly two unique medical scenarios")
    medical = medical.set_index("scenario").loc[["medical_disabled", "medical_high"]].reset_index()
    require(medical.region.eq("Kandy").all(), "Medical evidence must be Kandy")
    for index, expected in enumerate(((15, 75, 130.29, 245, 1350), (13, 65, 248.64, 362, 1170))):
        fields = list(MEDICAL_COLUMNS.values())[1:6]
        require(np.allclose(medical.loc[index, fields].to_numpy(dtype=float), expected, rtol=0, atol=1e-6), "Medical evidence differs from approved official results; refusing stale control")
    hospital_fields = list(MEDICAL_COLUMNS.values())[-3:]
    require(medical.loc[0, hospital_fields].isna().all(), "Disabled hospital-access metrics must remain missing")
    require(np.allclose(medical.loc[1, hospital_fields].to_numpy(dtype=float), [17.38, 3.47, 35.90], rtol=0, atol=0.005), "Enabled hospital-access evidence differs from approved values")
    require(len(restaurant) == 1 and restaurant.iloc[0].region == "Kandy", "Require one Kandy restaurant summary")
    restaurant = select_columns(restaurant, RESTAURANT_COLUMNS)
    r = restaurant.iloc[0]
    require(r.total_restaurants == 3239 and r.route_corridor_restaurants == 87, "Unexpected restaurant-filter evidence")
    require(r.route_filter_reduction == r.total_restaurants - r.route_corridor_restaurants, "Restaurant reduction count is inconsistent")
    require(close(r.route_filter_reduction_percent, 100 * r.route_filter_reduction / r.total_restaurants), "Restaurant reduction percentage is inconsistent")
    evidence = comparison.weighted_absa_score.notna()
    fallback = comparison.scoring_mode.eq("cuisine+detour")
    require(len(comparison) == r.ranked_rows and comparison.osm_id.nunique() == r.unique_ranked_restaurants, "Ranking counts disagree with summary")
    require(evidence.sum() == r.absa_rows and fallback.sum() == r.fallback_rows and fallback.eq(~evidence).all(), "ABSA/fallback evidence disagrees with summary")
    require(close(r.absa_percent, 100 * r.absa_rows / r.ranked_rows) and close(r.fallback_percent, 100 * r.fallback_rows / r.ranked_rows), "ABSA percentages are inconsistent")
    require(len(top) == 10, "Require ten existing top-recommendation rows")
    match = top.merge(comparison, on=["day", "osm_id"], suffixes=("_top", "_full"), validate="one_to_one")
    require(len(match) == 10 and np.allclose(match.final_restaurant_score_top, match.final_restaurant_score_full), "Top-10 evidence differs from full ranked output")
    banana_source = comparison[comparison.name.str.contains("banana chill", case=False, na=False)].copy()
    require(banana_source.osm_id.nunique() == 1 and sorted(banana_source.day.tolist()) == [1, 2, 3], "Require one Banana Chill restaurant with one row per day")
    banana = select_columns(banana_source, BANANA_COLUMNS).sort_values("day").reset_index(drop=True)
    banana.insert(3, "rank_shift", banana.detour_only_rank - banana.production_rank)
    require(banana.location.isna().all(), "Missing location evidence must remain missing")
    require(banana.scoring_mode.eq("absa+cuisine+detour").all(), "Unexpected Banana Chill scoring mode")
    require(np.array_equal(banana.production_rank, [7, 7, 5]) and np.array_equal(banana.detour_only_rank, [35, 1, 19]), "Banana Chill ranks differ from approved evidence")
    return select_columns(medical, MEDICAL_COLUMNS), restaurant, banana


def regional_improvements(regional):
    rows = []
    for region, group in regional.groupby("region", sort=False):
        methods = group.set_index("method")
        greedy, optimizer = methods.loc["greedy"], methods.loc["optimizer"]
        rows.append({
            "region": region,
            "poi_change": int(optimizer.pois_visited - greedy.pois_visited),
            "utility_change": int(optimizer.preference_utility - greedy.preference_utility),
            "distance_change_km": optimizer.road_distance_km - greedy.road_distance_km,
            "distance_change_percent": 100 * (optimizer.road_distance_km - greedy.road_distance_km) / greedy.road_distance_km,
            "driving_time_change_min": optimizer.driving_time_min - greedy.driving_time_min,
            "driving_time_change_percent": 100 * (optimizer.driving_time_min - greedy.driving_time_min) / greedy.driving_time_min,
        })
    return pd.DataFrame(rows)


def finish_chart(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_bars(ax, labels, first, second, legend, ylabel, fmt="%.0f"):
    x = np.arange(len(labels))
    for offset, values, name in ((-0.19, first, legend[0]), (0.19, second, legend[1])):
        bars = ax.bar(x + offset, values, width=0.36, label=name)
        ax.bar_label(bars, fmt=fmt, padding=4, fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(max(first), max(second)) * 1.22)
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)


def build_charts(folder, regional, medical, restaurant, banana):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 17, "axes.labelsize": 13})
    folder.mkdir()
    regions = ["Colombo", "Kandy", "Ella"]
    greedy = regional[regional.method == "greedy"].set_index("region").loc[regions]
    optimizer = regional[regional.method == "optimizer"].set_index("region").loc[regions]
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    grouped_bars(ax, regions, greedy.pois_visited, optimizer.pois_visited, ("Greedy", "Optimizer"), "POIs visited")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Three-region evaluation: POI coverage", pad=16)
    finish_chart(fig, folder / "regional_poi_utility_comparison.png")
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    grouped_bars(ax, regions, greedy.road_distance_km, optimizer.road_distance_km, ("Greedy", "Optimizer"), "Road distance (km)", "%.2f")
    ax.set_title("Three-region evaluation: road distance", pad=16)
    finish_chart(fig, folder / "regional_distance_comparison.png")
    metrics = ["pois_visited", "preference_utility", "road_distance_km", "driving_time_min"]
    baseline = medical.set_index("scenario").loc["medical_disabled", metrics].to_numpy(dtype=float)
    high = medical.set_index("scenario").loc["medical_high", metrics].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    grouped_bars(ax, ["POIs", "Preference utility", "Road distance", "Driving time"], np.full(4, 100), 100 * high / baseline, ("medical_disabled", "medical_high"), "Percent of medical_disabled baseline", "%.1f%%")
    ax.set_title("Kandy medical-access-aware trade-off\nEach disabled metric is independently normalized to 100%", pad=16)
    fig.supxlabel("Ratios compare each metric with its own baseline; hospital access has no disabled baseline.", fontsize=10)
    finish_chart(fig, folder / "medical_tradeoff.png")
    r = restaurant.iloc[0]
    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    bars = ax.bar(["Total dataset", "Inside route corridor"], [r.total_restaurants, r.route_corridor_restaurants])
    ax.bar_label(bars, fmt="%.0f", padding=5, fontsize=14)
    ax.set_ylim(0, r.total_restaurants * 1.18)
    ax.set_ylabel("Unique restaurants")
    ax.set_title(f"Route filtering: {r.route_filter_reduction_percent:.2f}% reduction", pad=16)
    ax.text(0.62, 0.77, f"{int(r.route_filter_reduction):,} restaurants excluded\n1,000 m road-route corridor", transform=ax.transAxes, ha="center", fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    finish_chart(fig, folder / "restaurant_filtering.png")
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    grouped_bars(ax, [f"Day {day}" for day in banana.day], banana.detour_only_rank, banana.production_rank, ("Detour-only rank (ties share rank)", "Production daily rank"), "Rank number (lower is better)")
    ax.set_title("Café Banana Chill: daily ranking comparison\nProduction combines ABSA + cuisine + detour", pad=16)
    fig.supxlabel("Observed shifts do not isolate ABSA from cuisine preference. Lower bars mean higher rank.", fontsize=10)
    finish_chart(fig, folder / "banana_chill_rank_comparison.png")


def summary_text(improvements, medical, restaurant, banana):
    lines = ["BASELINE VS OPTIMIZER", "Changes below are optimizer minus greedy."]
    for row in improvements.itertuples():
        lines.append(f"{row.region}: POIs {row.poi_change:+d}; utility {row.utility_change:+d}; distance {row.distance_change_km:+.2f} km ({row.distance_change_percent:+.2f}%); driving {row.driving_time_change_min:+.2f} min ({row.driving_time_change_percent:+.2f}%).")
    lines += [
        "Colombo: same POIs and utility with lower road distance and driving time.",
        "Kandy: +1 POI and +5 utility, with lower distance and similar driving time.",
        "Ella: +1 POI and +5 utility, requiring more distance and time: a utility/coverage trade-off.",
        "", "MEDICAL-ACCESS TRADE-OFF",
    ]
    d, h = medical.iloc[0], medical.iloc[1]
    lines.append(f"Kandy medical_high versus medical_disabled: POIs {h.pois_visited-d.pois_visited:+.0f}; utility {h.preference_utility-d.preference_utility:+.0f}; distance {h.road_distance_km-d.road_distance_km:+.2f} km; driving {h.driving_time_min-d.driving_time_min:+.0f} min; visit time {h.visit_time_min-d.visit_time_min:+.0f} min.")
    lines.append(f"Medical-access-aware selection incorporates modeled hospital access; high-scenario selected-POI access: average {h.avg_hospital_access_min:.2f}, best {h.best_hospital_access_min:.2f}, worst {h.worst_hospital_access_min:.2f} min.")
    lines.append("Disabled hospital-access values are missing, so no measured reduction in hospital-access time can be inferred.")
    r = restaurant.iloc[0]
    lines += ["", "RESTAURANT PIPELINE", f"Route filtering: {r.total_restaurants:.0f} to {r.route_corridor_restaurants:.0f} unique restaurants; reduction {r.route_filter_reduction:.0f} ({r.route_filter_reduction_percent:.2f}%).", f"{r.restaurant_day_candidates:.0f} restaurant/day corridor rows; {r.valid_detour_rows:.0f} valid detour rows; {r.ranked_rows:.0f} ranked rows representing {r.unique_ranked_restaurants:.0f} unique restaurants.", f"Modeled detour minutes: min {r.min_detour_min:.2f}, median {r.median_detour_min:.2f}, mean {r.mean_detour_min:.4f}, max {r.max_detour_min:.2f}.", f"ABSA evidence available: {r.absa_rows:.0f} rows ({r.absa_percent:.2f}%); transparent cuisine+detour fallback: {r.fallback_rows:.0f} rows ({r.fallback_percent:.2f}%)."]
    for row in banana.itertuples():
        lines.append(f"Banana Chill Day {row.day}: detour-only rank {row.detour_only_rank:.0f} -> production rank {row.production_rank:.0f}; shift {row.rank_shift:+.0f}; final score {row.final_score:.4f}; modeled detour {row.detour_min:.2f} min.")
    lines += ["", "KEY FINDINGS", "Optimizer results show region-dependent coverage/travel trade-offs rather than uniform distance reduction.", "Medical-access-aware weighting changes route selection and increases travel in this controlled scenario.", "Route filtering substantially narrows the candidate space; detour candidates generally have low modeled insertion times.", "Production uses ABSA-aware ranking when review evidence is available and fallback ranking when it is unavailable.", "Banana Chill ordering differs from detour-only ordering; ABSA alone cannot be credited for the rank shifts.", "", "LIMITATIONS", "The three-region evaluation is a prototype-stage sample, not full nationwide validation.", "The medical experiment is one controlled Kandy scenario. Hospital access values are modeled road travel times and do not guarantee emergency response.", "Restaurant ABSA coverage is limited: one unique Kandy restaurant on this route had ABSA evidence, appearing on three days.", "Ranking effects cannot isolate ABSA from cuisine preference. Detour-only ties share minimum rank; production ranks are day-specific.", "Missing location sentiment remains missing; no sentiment evidence was invented.", "The earlier historical OR-Tools 14-POI control discrepancy remains unexplained; current repeated runs were stable.", "Greedy driving times retain fractional minutes; optimizer driving times use production rounded-per-leg minutes, so small differences need cautious interpretation.", "Filtering counts use unique restaurants; detour/ranking counts use restaurant/day rows. Corridor reduction also includes production coordinate cleaning.", "Regional figures use the user's approved valid results embedded in this reporting script; medical and restaurant figures come from the existing source CSVs. No evaluations were rerun."]
    return "\n".join(lines) + "\n"


def main():
    snapshots = {p: p.read_bytes() for p in SOURCES}
    medical, restaurant, banana = load_evidence()
    regional = pd.DataFrame(APPROVED_REGIONAL_ROWS, columns=REGIONAL_COLUMNS)
    improvements = regional_improvements(regional)
    summary = summary_text(improvements, medical, restaurant, banana)
    tables = {
        "combined_regional_evaluation.csv": regional,
        "combined_regional_improvements.csv": improvements,
        "combined_medical_tradeoff.csv": medical,
        "combined_restaurant_summary.csv": restaurant,
        "combined_banana_chill_evidence.csv": banana,
    }
    # Generate everything before publishing; a validation/render failure leaves
    # previous reporting artifacts untouched. No production files are opened.
    with tempfile.TemporaryDirectory(prefix="combined_report_", dir=RESULTS) as temp:
        staging = Path(temp)
        for name, table in tables.items():
            table.to_csv(staging / name, index=False)
        (staging / "combined_evaluation_summary.txt").write_text(summary, encoding="utf-8")
        build_charts(staging / "charts", regional, medical, restaurant, banana)
        require(all(p.read_bytes() == data for p, data in snapshots.items()), "Source evidence changed during report generation")
        (RESULTS / "charts").mkdir(exist_ok=True)
        generated = sorted(p.relative_to(staging) for p in staging.rglob("*") if p.is_file())
        for relative in generated:
            (staging / relative).replace(RESULTS / relative)
    print("COMPONENT 3 COMBINED EVALUATION")
    for line in summary.splitlines():
        if line.startswith(("Colombo:", "Kandy:", "Ella:", "Kandy medical_high", "Route filtering:", "Banana Chill Day")):
            print(line)
    print("Interpretations, source assumptions and limitations: combined_evaluation_summary.txt")
    print("Source evidence unchanged: YES")
    for source, data in snapshots.items():
        print(f"Source SHA256 {source.name}: {hashlib.sha256(data).hexdigest()}")
    print("\nGenerated files:")
    for relative in generated:
        print(RESULTS / relative)


if __name__ == "__main__":
    main()
