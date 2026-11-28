"""Good deal / Looks high limits per town (src/c2/margins.py): town -> province -> whole country -> fixed 30%."""
import pandas as pd

from src.c2.data import HotelData
from src.c2.margins import PriceLimits
from src.c2.ranker import recommend
from tests.helpers import make_request


def write_margins(path, kandy_high=1.50, kandy_low=0.56):
    pd.DataFrame([
        {"built_on": "2027-01-10", "town": "Kandy", "province": "Central", "good_deal_ratio": kandy_low,
         "looks_high_ratio": kandy_high},
        {"built_on": "2027-01-10", "town": "(province) Central", "province": "Central", "good_deal_ratio": 0.57,
         "looks_high_ratio": 1.74},
        {"built_on": "2027-01-10", "town": "(whole country)", "province": "", "good_deal_ratio": 0.56,
         "looks_high_ratio": 1.77},
    ]).to_csv(path, index=False)
    return path


def test_town_then_province_then_country_then_fixed(tmp_path):
    limits = PriceLimits(write_margins(tmp_path / "price_margins.csv"))
    kandy = limits.for_town("Kandy")
    assert (kandy["good_deal_below_pct"], kandy["looks_high_above_pct"], kandy["place"]) == (44.0, 50.0, "Kandy")
    assert limits.for_town("Hatton", province="Central")["place"] == "Central province"
    country = limits.for_town("Hatton")
    assert (country["looks_high_above_pct"], country["place"]) == (77.0, "Sri Lanka")
    fixed = PriceLimits(tmp_path / "missing.csv").for_town("Kandy")
    assert (fixed["good_deal_below_pct"], fixed["looks_high_above_pct"], fixed["place"]) == (30.0, 30.0, None)


def test_a_broken_file_falls_back_to_the_fixed_rule(tmp_path):
    bad = tmp_path / "price_margins.csv"
    bad.write_text("this is not the file\n1,2\n")
    limits = PriceLimits(bad)
    assert not limits.loaded and "could not be read" in limits.note
    assert limits.for_town("Kandy")["looks_high_above_pct"] == 30.0


def test_the_towns_own_limit_decides_overpriced(fixture_dirs, tmp_path):
    """k-lux is about 67% above its fair price: overpriced with Kandy's limit +50%, normal with +80%."""
    def status(kandy_high):
        data = HotelData(*fixture_dirs, load_model=False,
                         margins_file=write_margins(tmp_path / f"m{kandy_high}.csv", kandy_high=kandy_high))
        hotels = recommend(make_request(), top_k=10, data=data)["results"][0]["ranked_hotels"]
        return {h["hotel_id"]: h for h in hotels}["k-lux"]

    strict = status(1.50)
    assert strict["price"]["price_status"] == "Potentially Overpriced"
    assert strict["price"]["price_limits"]["place"] == "Kandy"
    assert "In Kandy, the dearest 1 in 10 prices are 50% or more above the fair price." in strict["explanation"]
    assert status(1.80)["price"]["price_status"] == "Within Expected Price Range"