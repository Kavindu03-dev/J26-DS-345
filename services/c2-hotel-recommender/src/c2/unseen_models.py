"""
The models that NEVER saw a hotel - needed to explain its fair price with SHAP.

Why: the fair price of a hotel comes from a model that had not seen that hotel (booking_predictions.csv):
  - test hotels (222):     the final model (booking_best_model.joblib) - it never saw them;
  - training hotels (843): one of the 3 cross-validation models in model_training_booking.ipynb.
The final model DID learn the training hotels: its price for them is within about 3.5% of the real price, so
explaining it would explain the real price, not the fair price. The 3 cross-validation models were not saved
by the notebook, so this script makes them again, exactly as the notebook did (same rows, same 3 groups of
hotels, same settings, same seed), and checks they give the same fair prices as booking_predictions.csv.

Run once (about 3-10 minutes), from the service folder:
    python -m src.c2.unseen_models
It saves <C2_DATA_DIR>/models/booking_cv_models.joblib (a data file: never commit it to git).

Note: GroupKFold puts hotels with the same number of prices in an order that depends on how numpy sorts on
the computer's CPU (CPUs with AVX-512 sort ties differently). The laptop that trained the model has no AVX-512.
If the check fails, the script runs itself again once with AVX-512 sorting switched off.
"""
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

FILE_NAME = "booking_cv_models.joblib"
MAX_DIFFERENCE_PCT = 0.01      # the rebuilt models must give the same fair prices
NO_AVX512 = "AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
N_FOLDS = 3          # tune_folds in model_training_booking.ipynb: the fair prices come from these
SEED = 42


def _make_model(cat, num, params):
    """Same as make_model("Gradient boosting", BEST_PARAMS) in model_training_booking.ipynb."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan)
    prepare = ColumnTransformer([("cat", enc, cat), ("num", "passthrough", num)])
    model = HistGradientBoostingRegressor(l2_regularization=1.0, categorical_features=list(range(len(cat))),
                                          random_state=SEED, **params)
    return Pipeline([("prepare", prepare), ("model", model)])


def build(data_dir=None, log=print):
    import joblib
    from sklearn.model_selection import GroupKFold

    data_dir = Path(data_dir or settings.data_dir())
    clean, models = data_dir / "output_clean", data_dir / "models"
    info = json.loads((models / "booking_best_model_info.json").read_text(encoding="utf-8"))
    features, cat = info["features"], info["categorical_features"]
    num = [c for c in features if c not in cat]
    params = info["settings"]

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
        m = _make_model(cat, num, params).fit(X.iloc[a], y[a])
        pred[b] = m.predict(X.iloc[b])
        fitted.append(m)
        hotel_fold.update({h: k for h in np.unique(g[b])})
        log(f"cross-validation model {k + 1} of {N_FOLDS} made ({time.time() - t0:.0f} s)")

    # the check: the same fair prices as booking_predictions.csv?
    saved = pd.read_csv(models / "booking_predictions.csv", usecols=["row_id", "fair_price_lkr"])
    both = pd.DataFrame({"row_id": train.row_id, "ours": np.exp(pred)}).merge(saved, on="row_id")
    diff = (100 * (both.ours / both.fair_price_lkr - 1)).abs()
    log(f"same fair prices as booking_predictions.csv: {len(both):,} prices, biggest difference {diff.max():.6f}%")
    if not diff.max() <= MAX_DIFFERENCE_PCT:
        if "NPY_DISABLE_CPU_FEATURES" not in os.environ:
            log("Not the same: the hotels were split into different groups on this computer. "
                "Trying again with AVX-512 sorting switched off ...")
            env = dict(os.environ, NPY_DISABLE_CPU_FEATURES=NO_AVX512)
            subprocess.run([sys.executable, "-m", "src.c2.unseen_models", str(data_dir)], env=env, check=True)
            return None
        raise SystemExit("The rebuilt models do not give the same fair prices, so nothing was saved. "
                         "Was model_table.csv or booking_predictions.csv changed after the model was trained?")

    bundle = {"models": fitted, "hotel_fold": hotel_fold, "features": features,
              "max_difference_pct": float(diff.max()), "rows_checked": int(len(both)),
              "made": datetime.now().strftime("%Y-%m-%d %H:%M"), "settings": params, "folds": N_FOLDS,
              "note": "Cross-validation models of model_training_booking.ipynb (tune_folds), made again. "
                      "Each training hotel's fair price comes from the model that did not see it."}
    path = models / FILE_NAME
    joblib.dump(bundle, path, compress=3)
    log(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return bundle


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else None)
