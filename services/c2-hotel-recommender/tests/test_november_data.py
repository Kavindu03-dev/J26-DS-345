"""The November data layout (output_clean_v5 + models_v5_lgbm): chosen by default, towns from the map position,
Booking.com only, links for every hotel, live hotels get the v5 place columns, LightGBM explained with SHAP."""
import numpy as np
import pandas as pd
import pytest

from src.c2 import settings
from src.c2.data import HotelData, get_data, set_data
from src.c2.explain import explain_price
from src.c2.live import search as S
from src.c2.ranker import recommend
from src.c2.unseen_models import _make_model
from tests.helpers import make_files_november, make_request
from tests.test_live_search import FakePages
from tests.test_price_explanation import CAT, NUM


@pytest.fixture(scope="module")
def nov_dirs(tmp_path_factory):
    return make_files_november(tmp_path_factory.mktemp("c2nov"))


@pytest.fixture(scope="module")
def nov(nov_dirs):
    return HotelData(*nov_dirs)


def ids(result):
    return [h["hotel_id"] for h in result["results"][0]["ranked_hotels"]]


def test_november_is_the_default_when_its_files_are_there(nov, nov_dirs, monkeypatch):
    assert nov.version == "november" and nov.prefix == "booking_v5_lgbm"
    assert nov.clean_dir.name == "output_clean_v5" and nov.models_dir.name == "models_v5_lgbm"
    assert nov.prices_collected == "20-23 November 2026"
    assert any("Booking.com only" in n for n in nov.notes)
    assert "Hikkaduwa" in nov.towns.data_towns and nov.towns.point("Galle")[2] == "town centre in our data"
    monkeypatch.setenv("C2_DATA_VERSION", "september")                 # switch back
    old = HotelData(*nov_dirs)
    assert old.version == "september" and old.clean_dir.name == "output_clean"
    assert old.prices_collected == "25-27 September 2026" and "Hikkaduwa" not in old.towns.data_towns


def test_a_far_hotel_booking_showed_for_galle_is_not_in_galle(nov):
    galle = recommend(make_request(town="Galle"), top_k=10, today="2026-11-26", data=nov)
    assert "x-hikka" not in ids(galle) and "g-beach" in ids(galle)
    assert galle["prices_collected"] == "20-23 November 2026"
    hikka = recommend(make_request(town="Hikkaduwa"), top_k=10, today="2026-11-26", data=nov)
    assert ids(hikka) == ["x-hikka"]
    h = hikka["results"][0]["ranked_hotels"][0]
    # not in hotel_urls.csv: the link is made from the Booking.com hotel id
    assert h["booking_url"].startswith("https://www.booking.com/hotel/lk/x-hikka.html?checkin=")
    assert h["price"]["best_platform"] == "Booking.com" and h["price"]["where_to_book_advice"] == "Book on Booking.com."


def test_live_hotels_get_the_november_place_columns(nov_dirs, data):
    old = data                                  # the session data (conftest)
    try:
        new = HotelData(*nov_dirs)
        set_data(new)
        assert new.live_dir.parent.name == "output_clean_v5"
        report = S.run_live_search(make_request(town="Galle"), "Galle", today="2026-11-26", pages=FakePages(),
                                   data=new)
        assert report["status"] == "ok" and report["new_hotels"]
        saved = pd.read_csv(new.live_dir / "hotels_booking_live.csv")
        # the saved hotel page is Le Grand Galle (Galle District): every new hotel gets its position
        assert set(saved.town) == {"Galle"} and set(saved.searched_towns) == {"Galle"}
        assert set(saved.district) == {"Galle"} and set(saved.province) == {"Southern"}
        assert set(saved.kind_of_place) == {"beach"} and (saved.km_from_town_centre < 3).all()
        prices = pd.read_csv(new.live_dir / "prices_live.csv")
        assert set(prices.destination_clean) == {"Galle"}                  # the price model's town
        assert S._model_town(new, 6.139, 80.101, "Galle") == "Hikkaduwa"    # nearest town centre
        fresh = get_data()                                                   # reloaded: still the november data
        assert fresh.version == "november" and fresh.is_live(saved.booking_hotel_id.iloc[0])
    finally:
        set_data(old)


def test_lightgbm_price_model_is_explained_with_shap(nov):
    pytest.importorskip("lightgbm")
    d = HotelData(nov.data_dir, nov.extra_dir, load_model=False)
    rows = d.model_rows
    model = _make_model(CAT, NUM, {"learning_rate": 0.1, "n_estimators": 60, "num_leaves": 7, "min_child_samples": 3},
                        "LightGBM")
    model.fit(rows[CAT + NUM], np.log(rows.price_incl_tax_lkr_night.values))
    d.model, d.features = model, CAT + NUM
    rid = int(rows[rows.booking_hotel_id == "g-beach"].row_id.iloc[0])          # a test hotel
    e = explain_price(d, rid)
    assert e["available"] and e["checks"]["parts_add_up_to_the_fair_price"]
    assert e["fair_price_lkr"] == pytest.approx(float(np.exp(model.predict(rows.loc[[rid], d.features]))[0]), rel=1e-6)
    assert e["start_lkr"] + sum(p["lkr"] for p in e["parts"]) == pytest.approx(e["fair_price_lkr"], rel=1e-6)


def test_bar_switch_leaves_the_barbecue_on():
    from src.c2.pricing import facility_switches
    from src.c2.explain import group_of
    off = facility_switches(["has_bar", "fac_bar", "fac_snack_bar", "fac_barbecue", "fac_room_barbecue"])["bar"]
    assert set(off) == {"has_bar", "fac_bar", "fac_snack_bar"}
    assert group_of("fac_barbecue") == "Other facilities" and group_of("fac_snack_bar") == "Bar"


def test_every_november_town_has_rainy_months():
    import src.c2.settings as s
    towns = ["Colombo", "Negombo", "Mount Lavinia", "Katunayake", "Kalutara", "Wadduwa", "Beruwala", "Bentota",
             "Hikkaduwa", "Galle", "Unawatuna", "Koggala", "Ahangama", "Weligama", "Mirissa", "Matara", "Tangalle",
             "Hambantota", "Tissamaharama", "Kandy", "Nuwara Eliya", "Hatton", "Matale", "Dambulla", "Sigiriya",
             "Ella", "Bandarawela", "Haputale", "Badulla", "Ratnapura", "Kitulgala", "Pinnawala", "Anuradhapura",
             "Habarana", "Polonnaruwa", "Kurunegala", "Chilaw", "Kalpitiya", "Trincomalee", "Nilaveli", "Batticaloa",
             "Pasikuda", "Arugam Bay", "Jaffna", "Mannar", "Vavuniya"]
    assert len(towns) == 46 and all(t in s.TOWN_AREA for t in towns)
    assert settings.prices_collected("september") == "25-27 September 2026"
