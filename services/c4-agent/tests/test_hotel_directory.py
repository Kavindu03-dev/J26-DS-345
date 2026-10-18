from src.hotel_directory import HotelDirectory

directory = HotelDirectory()


def test_loads_c1_hotels():
    assert len(directory) > 1000


def test_exact_name_match_scores_one():
    matches = directory.find("Orchard House Galle")
    assert matches[0].hotel.hotel_id == "HTL_GAL_ORCHARDH_2FE6"
    assert matches[0].score == 1.0


def test_partial_and_misspelt_names_still_match():
    assert directory.find("orchard house")[0].hotel.hotel_id == "HTL_GAL_ORCHARDH_2FE6"
    assert directory.find("Orchad Hous Galle")[0].hotel.hotel_id == "HTL_GAL_ORCHARDH_2FE6"


def test_unknown_name_returns_nothing():
    assert directory.find("Hotel That Does Not Exist Anywhere Xyz") == []


def test_destination_by_city_and_district():
    assert all(h.city == "Ella" for h in directory.in_destination("ella"))
    assert len(directory.in_destination("Nuwara Eliya")) > 0
    assert directory.in_destination("Atlantis") == []
