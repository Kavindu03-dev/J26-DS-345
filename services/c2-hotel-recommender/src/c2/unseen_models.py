"""
The models that NEVER saw a hotel - needed to explain its fair price with SHAP.

Why: the fair price of a hotel comes from a model that had not seen that hotel (<prefix>_predictions.csv):
  - test hotels:     the final model (<prefix>_best_model.joblib) - it never saw them;
  - training hotels: one of the 3 cross-validation models of the model notebook (tune_folds).
The final model DID learn the training hotels: its price for them is close to the real price, so explaining it
would explain the real price, not the fair price. The 3 cross-validation models were not saved by the notebook,
so this script makes them again, exactly as the notebook did (same rows, same 3 groups of hotels, same settings,
same seed), and checks they give the same fair prices as <prefix>_predictions.csv.

Works for both data versions (settings.DATA_VERSIONS):
    november   model_training_booking_v5_lightgbm.ipynb -> models_v5_lgbm/booking_v5_lgbm_cv_models.joblib (LightGBM)
    september  model_training_booking.ipynb             -> models/booking_cv_models.joblib (gradient boosting)

Run once (about 3-10 minutes), from the service folder:
    python -m src.c2.unseen_models                 (the data version the service uses)
    python -m src.c2.unseen_models --version september
It saves the file next to the price model (a data file: never commit it to git).

Note: GroupKFold puts hotels with the same number of prices in an order that depends on how numpy sorts on
the computer's CPU (CPUs with AVX-512 sort ties differently). The laptop that trained the models has no AVX-512.
If the check fails, the script runs itself again once with AVX-512 sorting switched off.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings

MAX_DIFFERENCE_PCT = 0.01      # the rebuilt models must give the same fair prices
NO_AVX512 = "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
N_FOLDS = 3          # tune_folds in the model notebooks: the training hotels' fair prices come from these
SEED = 42
NOTEBOOK = {"november": "model_training_booking_v5_lightgbm.ipynb", "september": "model_training_booking.ipynb"}


def file_name(prefix):
    return f"{prefix}_cv_models.joblib"


def _make_model(cat, num, params, model_name="Gradient boosting"):
    """Same as make_model(BEST, BEST_PARAMS) in the model notebooks."""
    from sklearn.pipeline import Pipeline

    if model_name == "LightGBM":
        import lightgbm
        from sklearn.preprocessing import FunctionTransformer

        to_category = FunctionTransformer(pd.DataFrame.astype, kw_args={"dtype": {c: "category" for c in cat}})
        return Pipeline([("prepare", to_category), ("model", lightgbm.LGBMRegressor(
            subsample_freq=1, random_state=SEED, n_jobs=-1, deterministic=True, force_col_wise=True, verbose=-1,
            **params))])
    if model_name == "Gradient boosting":
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.preprocessing import OrdinalEncoder

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
        prepare = ColumnTransformer([("cat", enc, cat), ("num", "passthrough", num)])
        model = HistGradientBoostingRegressor(l2_regularization=1.0, categorical_features=list(range(len(cat))),
                                              random_state=SEED, **params)
        return Pipeline([("prepare", prepare), ("model", model)])
    raise SystemExit(f"Don't know how to rebuild a '{model_name}' model (only LightGBM and Gradient boosting).")


def build(data_dir=None, version=None, log=print):
    import joblib
    from sklearn.model_selection import GroupKFold

    data_dir = Path(data_dir or settings.data_dir())
    lay = settings.layout(data_dir, version)
    clean, models, prefix = lay["clean"], lay["models"], lay["prefix"]
    info = json.loads((models / f"{prefix}_best_model_info.json").read_text(encoding="utf-8"))
    model_name = info.get("model", "Gradient boosting")
    features, cat = info["features"], info["categorical_features"]
    num = [c for c in features if c not in cat]
    params = {k: (tuple(v) if isinstance(v, list) else v) for k, v in info["settings"].items()}
    log(f"{lay['version']} data: {model_name} with {len(features)} columns, settings {params}")

    T = pd.read_csv(clean / "model_table.csv", low_memory=False)
    W = pd.read_csv(clean / "hotel_words_facilities.csv")
    T = T[(T.is_extra_date == 0) & (T.source_ota == "booking")]
    T = T.merge(W, on="booking_hotel_id", how="left").reset_index(drop=True)
    train = T[T.split == "train"].reset_index(drop=True)
    X, y, g = train[features], train["log_price_incl_tax"].values, train.hotel_group_id.values

    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, g))
    fitted, hotel_fold, pred = [], {}, np.full(len(y), np.nan)
    t0 = time.time()
    for k, (a, b) in enumerate(folds):
        m = _make_model(cat, num, params, model_name).fit(X.iloc[a], y[a])
        pred[b] = m.predict(X.iloc[b])
        fitted.append(m)
        hotel_fold.update({h: k for h in np.unique(g[b])})
        log(f"cross-validation model {k + 1} of {N_FOLDS} made ({time.time() - t0:.0f} s)")

    # the check: the same fair prices as <prefix>_predictions.csv?
    saved = pd.read_csv(models / f"{prefix}_predictions.csv")
    saved = saved[saved.split == "train"] if "split" in saved else saved
    saved = saved[["row_id", "fair_price_lkr"]]
    both = pd.DataFrame({"row_id": train.row_id, "ours": np.exp(pred)}).merge(saved, on="row_id")
    diff = (100 * (both.ours / both.fair_price_lkr - 1)).abs()
    log(f"same fair prices as {prefix}_predictions.csv: {len(both):,} of {len(train):,} prices, "
        f"biggest difference {diff.max():.6f}%")
    if not (len(both) == len(train) and diff.max() <= MAX_DIFFERENCE_PCT):
        if "NPY_DISABLE_CPU_FEATURES" not in os.environ:
            log("Not the same: the hotels were split into different groups on this computer. "
                "Trying again with AVX-512 sorting switched off ...")
            env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NO_AVX512)
            subprocess.run([sys.executable, "-m", "src.c2.unseen_models", str(data_dir), "--version", lay["version"]],
                           env=env, check=True)
            return None
        raise SystemExit("The rebuilt models do not give the same fair prices, so nothing was saved. "
                         f"Was model_table.csv or {prefix}_predictions.csv changed after the model was trained?")

    bundle = {"models": fitted, "hotel_fold": hotel_fold, "features": features, "model": model_name,
              "data_version": lay["version"], "max_difference_pct": float(diff.max()), "rows_checked": int(len(both)),
              "made": datetime.now().strftime("%Y-%m-%d %H:%M"), "settings": info["settings"], "folds": N_FOLDS,
              "note": f"Cross-validation models of {NOTEBOOK.get(lay['version'], 'the model notebook')} (tune_folds), "
                      "made again. Each training hotel's fair price comes from the model that did not see it."}
    path = models / file_name(prefix)
    joblib.dump(bundle, path, compress=3)
    log(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return bundle


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Make the 3 cross-validation price models again (for SHAP).")
    ap.add_argument("data_dir", nargs="?", default=None, help="the hotel_preprocessing folder (default: C2_DATA_DIR)")
    ap.add_argument("--version", choices=sorted(settings.DATA_VERSIONS), default=None,
                    help="data version (default: C2_DATA_VERSION, or november when its files are there)")
    a = ap.parse_args()
    build(a.data_dir, a.version)
