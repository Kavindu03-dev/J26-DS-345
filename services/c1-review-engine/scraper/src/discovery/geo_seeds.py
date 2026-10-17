"""
Sri Lanka Geographic Seeds & Stratification Data (25 Administrative Districts, 9 Provinces)
and Top 50 Tourism Destination Landmarks.
"""

# 1. 25 Administrative Districts grouped by Province with key search queries
SRI_LANKA_DISTRICTS = {
    # Southern Province
    "Galle": {
        "province": "Southern",
        "search_terms": ["Galle Fort, Sri Lanka", "Unawatuna, Sri Lanka", "Hikkaduwa, Sri Lanka", "Ahangama, Sri Lanka"]
    },
    "Matara": {
        "province": "Southern",
        "search_terms": ["Mirissa, Sri Lanka", "Weligama, Sri Lanka", "Matara, Sri Lanka", "Dikwella, Sri Lanka"]
    },
    "Hambantota": {
        "province": "Southern",
        "search_terms": ["Tangalle, Sri Lanka", "Yala, Sri Lanka", "Tissamaharama, Sri Lanka", "Hambantota, Sri Lanka"]
    },

    # Western Province
    "Colombo": {
        "province": "Western",
        "search_terms": ["Colombo, Sri Lanka", "Mount Lavinia, Sri Lanka", "Galle Face, Colombo", "Kollupitiya, Colombo"]
    },
    "Gampaha": {
        "province": "Western",
        "search_terms": ["Negombo, Sri Lanka", "Gampaha, Sri Lanka", "Kelaniya, Sri Lanka"]
    },
    "Kalutara": {
        "province": "Western",
        "search_terms": ["Bentota, Sri Lanka", "Kalutara, Sri Lanka", "Wadduwa, Sri Lanka", "Beruwala, Sri Lanka"]
    },

    # Central Province
    "Kandy": {
        "province": "Central",
        "search_terms": ["Kandy, Sri Lanka", "Peradeniya, Sri Lanka", "Katugastota, Sri Lanka", "Digana, Sri Lanka"]
    },
    "Nuwara Eliya": {
        "province": "Central",
        "search_terms": ["Nuwara Eliya, Sri Lanka", "Hatton, Sri Lanka", "Nanu Oya, Sri Lanka", "Maskeliya, Sri Lanka"]
    },
    "Matale": {
        "province": "Central",
        "search_terms": ["Sigiriya, Sri Lanka", "Dambulla, Sri Lanka", "Matale, Sri Lanka", "Knuckles, Sri Lanka"]
    },

    # North Central Province
    "Anuradhapura": {
        "province": "North Central",
        "search_terms": ["Anuradhapura, Sri Lanka", "Mihintale, Sri Lanka", "Wilpattu, Sri Lanka"]
    },
    "Polonnaruwa": {
        "province": "North Central",
        "search_terms": ["Polonnaruwa, Sri Lanka", "Giritale, Sri Lanka", "Habarana, Sri Lanka", "Minneriya, Sri Lanka"]
    },

    # Eastern Province
    "Trincomalee": {
        "province": "Eastern",
        "search_terms": ["Trincomalee, Sri Lanka", "Nilaveli, Sri Lanka", "Uppuveli, Sri Lanka"]
    },
    "Batticaloa": {
        "province": "Eastern",
        "search_terms": ["Pasikuda, Sri Lanka", "Batticaloa, Sri Lanka", "Kalkudah, Sri Lanka"]
    },
    "Ampara": {
        "province": "Eastern",
        "search_terms": ["Arugam Bay, Sri Lanka", "Pottuvil, Sri Lanka", "Ampara, Sri Lanka"]
    },

    # Northern Province
    "Jaffna": {
        "province": "Northern",
        "search_terms": ["Jaffna, Sri Lanka", "Nallur, Sri Lanka", "Point Pedro, Sri Lanka", "Chavakachcheri, Sri Lanka", "Karainagar, Sri Lanka"]
    },
    "Mannar": {
        "province": "Northern",
        "search_terms": ["Mannar, Sri Lanka", "Talaimannar, Sri Lanka", "Pesalai, Sri Lanka", "Murunkan, Sri Lanka"]
    },
    "Vavuniya": {
        "province": "Northern",
        "search_terms": ["Vavuniya, Sri Lanka", "Cheddikulam, Sri Lanka", "Nedunkeni, Sri Lanka"]
    },
    "Kilinochchi": {
        "province": "Northern",
        "search_terms": ["Kilinochchi, Sri Lanka", "Paranthan, Sri Lanka", "Pooneryn, Sri Lanka"]
    },
    "Mullaitivu": {
        "province": "Northern",
        "search_terms": ["Mullaitivu, Sri Lanka", "Puthukkudiyiruppu, Sri Lanka", "Alampil, Sri Lanka"]
    },

    # Uva Province
    "Badulla": {
        "province": "Uva",
        "search_terms": ["Ella, Sri Lanka", "Haputale, Sri Lanka", "Badulla, Sri Lanka", "Bandarawela, Sri Lanka"]
    },
    "Monaragala": {
        "province": "Uva",
        "search_terms": ["Kataragama, Sri Lanka", "Wellawaya, Sri Lanka", "Buttala, Sri Lanka"]
    },

    # Sabaragamuwa Province
    "Ratnapura": {
        "province": "Sabaragamuwa",
        "search_terms": ["Ratnapura, Sri Lanka", "Udawalawe, Sri Lanka", "Belihuloya, Sri Lanka", "Sinharaja, Sri Lanka"]
    },
    "Kegalle": {
        "province": "Sabaragamuwa",
        "search_terms": ["Kitulgala, Sri Lanka", "Pinnawala, Sri Lanka", "Kegalle, Sri Lanka"]
    },

    # North Western Province
    "Puttalam": {
        "province": "North Western",
        "search_terms": ["Kalpitiya, Sri Lanka", "Puttalam, Sri Lanka", "Chilaw, Sri Lanka", "Marawila, Sri Lanka"]
    },
    "Kurunegala": {
        "province": "North Western",
        "search_terms": ["Kurunegala, Sri Lanka", "Ibbagamuwa, Sri Lanka", "Yapahuwa, Sri Lanka"]
    }
}

# 2. Top 50 Sri Lankan Travel Destinations & Landmarks
TOP_SRI_LANKA_DESTINATIONS = [
    # Cultural & UNESCO Heritage
    {
        "id": "DST_001",
        "name": "Sigiriya Ancient Rock Fortress",
        "category": "UNESCO World Heritage / Historical Landmark",
        "city": "Sigiriya",
        "district": "Matale",
        "province": "Central",
        "latitude": 7.9570,
        "longitude": 80.7603,
        "search_query": "Sigiriya Rock Fortress Sri Lanka"
    },
    {
        "id": "DST_002",
        "name": "Temple of the Sacred Tooth Relic (Sri Dalada Maligawa)",
        "category": "Religious / Cultural Landmark",
        "city": "Kandy",
        "district": "Kandy",
        "province": "Central",
        "latitude": 7.2936,
        "longitude": 80.6413,
        "search_query": "Temple of the Sacred Tooth Relic Kandy"
    },
    {
        "id": "DST_003",
        "name": "Galle Dutch Fort",
        "category": "UNESCO World Heritage / Colonial Heritage",
        "city": "Galle",
        "district": "Galle",
        "province": "Southern",
        "latitude": 6.0270,
        "longitude": 80.2170,
        "search_query": "Galle Dutch Fort Sri Lanka"
    },
    {
        "id": "DST_004",
        "name": "Dambulla Royal Cave Temple",
        "category": "UNESCO World Heritage / Cave Temple",
        "city": "Dambulla",
        "district": "Matale",
        "province": "Central",
        "latitude": 7.8567,
        "longitude": 80.6483,
        "search_query": "Dambulla Royal Cave Temple"
    },
    {
        "id": "DST_005",
        "name": "Ruwanwelisaya Sacred Stupa",
        "category": "Ancient Monument / Sacred City",
        "city": "Anuradhapura",
        "district": "Anuradhapura",
        "province": "North Central",
        "latitude": 8.3500,
        "longitude": 80.3964,
        "search_query": "Ruwanwelisaya Anuradhapura"
    },
    {
        "id": "DST_006",
        "name": "Jaya Sri Maha Bodhi",
        "category": "Sacred Tree / Ancient Pilgrimage",
        "city": "Anuradhapura",
        "district": "Anuradhapura",
        "province": "North Central",
        "latitude": 8.3448,
        "longitude": 80.3970,
        "search_query": "Jaya Sri Maha Bodhi Anuradhapura"
    },
    {
        "id": "DST_007",
        "name": "Polonnaruwa Ancient City Ruins & Gal Vihara",
        "category": "UNESCO World Heritage / Archaeological Site",
        "city": "Polonnaruwa",
        "district": "Polonnaruwa",
        "province": "North Central",
        "latitude": 7.9653,
        "longitude": 81.0022,
        "search_query": "Gal Vihara Polonnaruwa"
    },
    {
        "id": "DST_008",
        "name": "Mihintale Sacred Mountain Peak",
        "category": "Cradle of Buddhism / Historical Landmark",
        "city": "Mihintale",
        "district": "Anuradhapura",
        "province": "North Central",
        "latitude": 8.3508,
        "longitude": 80.5097,
        "search_query": "Mihintale Sacred Mountain"
    },

    # Nature, Mountains & Scenic Vistas
    {
        "id": "DST_009",
        "name": "Nine Arch Bridge",
        "category": "Colonial Railway Architecture / Scenic Spot",
        "city": "Ella",
        "district": "Badulla",
        "province": "Uva",
        "latitude": 6.8768,
        "longitude": 81.0608,
        "search_query": "Nine Arch Bridge Ella Sri Lanka"
    },
    {
        "id": "DST_010",
        "name": "Little Adam's Peak (Punchi Sri Pada)",
        "category": "Hiking / Scenic Viewpoint",
        "city": "Ella",
        "district": "Badulla",
        "province": "Uva",
        "latitude": 6.8647,
        "longitude": 81.0573,
        "search_query": "Little Adams Peak Ella"
    },
    {
        "id": "DST_011",
        "name": "Horton Plains National Park & World's End",
        "category": "National Park / Cloud Forest Trek",
        "city": "Ohiya",
        "district": "Nuwara Eliya",
        "province": "Central",
        "latitude": 6.8096,
        "longitude": 80.8017,
        "search_query": "Horton Plains Worlds End Sri Lanka"
    },
    {
        "id": "DST_012",
        "name": "Adam's Peak (Sri Pada)",
        "category": "Sacred Mountain / Sunrise Pilgrimage",
        "city": "Nallathanniya",
        "district": "Nuwara Eliya",
        "province": "Central",
        "latitude": 6.8096,
        "longitude": 80.4994,
        "search_query": "Adams Peak Sri Pada Sri Lanka"
    },
    {
        "id": "DST_013",
        "name": "Lipton's Seat",
        "category": "Tea Plantation / Panoramic Viewpoint",
        "city": "Haputale",
        "district": "Badulla",
        "province": "Uva",
        "latitude": 6.7828,
        "longitude": 81.0139,
        "search_query": "Liptons Seat Haputale"
    },
    {
        "id": "DST_014",
        "name": "Pidurangala Rock",
        "category": "Hiking / Sunrise Panoramic Rock",
        "city": "Sigiriya",
        "district": "Matale",
        "province": "Central",
        "latitude": 7.9664,
        "longitude": 80.7631,
        "search_query": "Pidurangala Rock Sigiriya"
    },
    {
        "id": "DST_015",
        "name": "Knuckles Mountain Range",
        "category": "UNESCO Wilderness / Adventure Trekking",
        "city": "Matale",
        "district": "Matale",
        "province": "Central",
        "latitude": 7.4589,
        "longitude": 80.7892,
        "search_query": "Knuckles Mountain Range Sri Lanka"
    },
    {
        "id": "DST_016",
        "name": "Sinharaja Virgin Rain Forest Reserve",
        "category": "UNESCO Biosphere / Tropical Rainforest",
        "city": "Deniyaya",
        "district": "Ratnapura",
        "province": "Sabaragamuwa",
        "latitude": 6.4167,
        "longitude": 80.4667,
        "search_query": "Sinharaja Forest Reserve"
    },

    # Wildlife & Safari Parks
    {
        "id": "DST_017",
        "name": "Yala National Park Safari (Ruhuna)",
        "category": "Wildlife Safari / Leopard Sanctuary",
        "city": "Yala",
        "district": "Hambantota",
        "province": "Southern",
        "latitude": 6.3725,
        "longitude": 81.5165,
        "search_query": "Yala National Park Sri Lanka"
    },
    {
        "id": "DST_018",
        "name": "Udawalawe National Park",
        "category": "Wildlife Safari / Wild Asian Elephant Sanctuary",
        "city": "Udawalawe",
        "district": "Ratnapura",
        "province": "Sabaragamuwa",
        "latitude": 6.4744,
        "longitude": 80.8986,
        "search_query": "Udawalawe National Park"
    },
    {
        "id": "DST_019",
        "name": "Minneriya National Park (The Elephant Gathering)",
        "category": "Wildlife Safari / Elephant Gathering",
        "city": "Habarana",
        "district": "Polonnaruwa",
        "province": "North Central",
        "latitude": 8.0333,
        "longitude": 80.8333,
        "search_query": "Minneriya National Park"
    },
    {
        "id": "DST_020",
        "name": "Wilpattu National Park",
        "category": "Wildlife Safari / Natural Lakes & Leopards",
        "city": "Nochchiyagama",
        "district": "Anuradhapura",
        "province": "North Central",
        "latitude": 8.4419,
        "longitude": 80.0039,
        "search_query": "Wilpattu National Park"
    },
    {
        "id": "DST_021",
        "name": "Pigeon Island National Marine Park",
        "category": "Marine Sanctuary / Snorkeling & Coral Reef",
        "city": "Nilaveli",
        "district": "Trincomalee",
        "province": "Eastern",
        "latitude": 8.7203,
        "longitude": 81.2036,
        "search_query": "Pigeon Island National Park Nilaveli"
    },

    # Coastal, Beaches & Marine Tourism
    {
        "id": "DST_022",
        "name": "Mirissa Beach & Whale Watching Harbour",
        "category": "Beach / Blue Whale & Dolphin Watching",
        "city": "Mirissa",
        "district": "Matara",
        "province": "Southern",
        "latitude": 5.9483,
        "longitude": 80.4578,
        "search_query": "Mirissa Beach Whale Watching"
    },
    {
        "id": "DST_023",
        "name": "Weligama Bay Surf Point",
        "category": "Surfing / Coastal Bay",
        "city": "Weligama",
        "district": "Matara",
        "province": "Southern",
        "latitude": 5.9722,
        "longitude": 80.4286,
        "search_query": "Weligama Bay Surfing"
    },
    {
        "id": "DST_024",
        "name": "Arugam Bay Surf Point",
        "category": "World-Class Surfing Point / Coastal",
        "city": "Arugam Bay",
        "district": "Ampara",
        "province": "Eastern",
        "latitude": 6.8428,
        "longitude": 81.8344,
        "search_query": "Arugam Bay Surf Point"
    },
    {
        "id": "DST_025",
        "name": "Pasikuda & Kalkudah Coral Bay",
        "category": "Shallow Coral Beach / Watersports",
        "city": "Pasikuda",
        "district": "Batticaloa",
        "province": "Eastern",
        "latitude": 7.9258,
        "longitude": 81.5644,
        "search_query": "Pasikuda Beach Sri Lanka"
    },
    {
        "id": "DST_026",
        "name": "Hikkaduwa Coral Reef & Marine Sanctuary",
        "category": "Coral Reef / Sea Turtle Hatchery",
        "city": "Hikkaduwa",
        "district": "Galle",
        "province": "Southern",
        "latitude": 6.1395,
        "longitude": 80.1008,
        "search_query": "Hikkaduwa Coral Reef"
    },
    {
        "id": "DST_027",
        "name": "Bentota Beach & Madu River Boat Safari",
        "category": "River Estuary / Watersports / Mangrove Safari",
        "city": "Bentota",
        "district": "Kalutara",
        "province": "Western",
        "latitude": 6.4219,
        "longitude": 79.9989,
        "search_query": "Bentota Beach Madu River Safari"
    },
    {
        "id": "DST_028",
        "name": "Kalpitiya Peninsula & Dolphin Safari",
        "category": "Kitesurfing / Spinner Dolphin Lagoon",
        "city": "Kalpitiya",
        "district": "Puttalam",
        "province": "North Western",
        "latitude": 8.2325,
        "longitude": 79.7656,
        "search_query": "Kalpitiya Kitesurfing Dolphin Safari"
    },
    {
        "id": "DST_029",
        "name": "Nilaveli Beach & Marble Beach",
        "category": "Pristine White Sand Beach",
        "city": "Trincomalee",
        "district": "Trincomalee",
        "province": "Eastern",
        "latitude": 8.6833,
        "longitude": 81.1833,
        "search_query": "Nilaveli Beach Trincomalee"
    },

    # Northern & Historical Landmarks
    {
        "id": "DST_030",
        "name": "Nallur Kandaswamy Kovil",
        "category": "Historic Hindu Dravidian Temple",
        "city": "Jaffna",
        "district": "Jaffna",
        "province": "Northern",
        "latitude": 9.6744,
        "longitude": 80.0294,
        "search_query": "Nallur Kandaswamy Temple Jaffna"
    },
    {
        "id": "DST_031",
        "name": "Jaffna Dutch Fort & Nagadeepa Island",
        "category": "Colonial Fortification / Island Pilgrimage",
        "city": "Jaffna",
        "district": "Jaffna",
        "province": "Northern",
        "latitude": 9.6603,
        "longitude": 80.0097,
        "search_query": "Jaffna Fort Sri Lanka"
    },
    {
        "id": "DST_032",
        "name": "Koneswaram Temple (Temple of the Thousand Pillars)",
        "category": "Cliffside Historic Kovil / Swami Rock",
        "city": "Trincomalee",
        "district": "Trincomalee",
        "province": "Eastern",
        "latitude": 8.5878,
        "longitude": 81.2422,
        "search_query": "Koneswaram Temple Trincomalee"
    },

    # Adventure, Waterfalls & Activities
    {
        "id": "DST_033",
        "name": "Kitulgala White Water Rafting & Rain Forest",
        "category": "Adventure Sports / White Water Rafting",
        "city": "Kitulgala",
        "district": "Kegalle",
        "province": "Sabaragamuwa",
        "latitude": 6.9917,
        "longitude": 80.4139,
        "search_query": "Kitulgala White Water Rafting"
    },
    {
        "id": "DST_034",
        "name": "Diyaluma Falls & Natural Infinity Pools",
        "category": "Waterfall / Natural Rock Pools",
        "city": "Koslanda",
        "district": "Badulla",
        "province": "Uva",
        "latitude": 6.7328,
        "longitude": 81.0319,
        "search_query": "Diyaluma Falls Sri Lanka"
    },
    {
        "id": "DST_035",
        "name": "Ravana Falls & Ravana Cave",
        "category": "Scenic Waterfall / Mythological Cave",
        "city": "Ella",
        "district": "Badulla",
        "province": "Uva",
        "latitude": 6.8408,
        "longitude": 81.0536,
        "search_query": "Ravana Falls Ella"
    },
    {
        "id": "DST_036",
        "name": "Pinnawala Elephant Orphanage",
        "category": "Wildlife Conservation / Elephant River Bath",
        "city": "Pinnawala",
        "district": "Kegalle",
        "province": "Sabaragamuwa",
        "latitude": 7.3014,
        "longitude": 80.3867,
        "search_query": "Pinnawala Elephant Orphanage"
    },
    {
        "id": "DST_037",
        "name": "Royal Botanic Gardens Peradeniya",
        "category": "Botanical Gardens / Orchid House",
        "city": "Peradeniya",
        "district": "Kandy",
        "province": "Central",
        "latitude": 7.2683,
        "longitude": 80.5969,
        "search_query": "Peradeniya Botanical Gardens Kandy"
    },
    {
        "id": "DST_038",
        "name": "Gangaramaya Buddhist Temple",
        "category": "Iconic Urban Buddhist Temple & Museum",
        "city": "Colombo",
        "district": "Colombo",
        "province": "Western",
        "latitude": 6.9167,
        "longitude": 79.8569,
        "search_query": "Gangaramaya Temple Colombo"
    },
    {
        "id": "DST_039",
        "name": "Colombo National Museum & Galle Face Green",
        "category": "National Heritage / Urban Promenade",
        "city": "Colombo",
        "district": "Colombo",
        "province": "Western",
        "latitude": 6.9044,
        "longitude": 79.8614,
        "search_query": "National Museum Colombo Galle Face"
    },
    {
        "id": "DST_040",
        "name": "Lotus Tower Colombo (Nelum Kuluna)",
        "category": "Modern Architecture / City Observation Deck",
        "city": "Colombo",
        "district": "Colombo",
        "province": "Western",
        "latitude": 6.9298,
        "longitude": 79.8580,
        "search_query": "Lotus Tower Colombo"
    }
]

def get_province_for_district(district_name: str) -> str:
    """Lookup province for a given administrative district."""
    for dist, data in SRI_LANKA_DISTRICTS.items():
        if dist.lower() == district_name.lower():
            return data["province"]
    return "Sri Lanka"

def get_district_and_province(location_str: str):
    """
    Heuristically extract district and province from a location string (e.g. 'Mirissa, Sri Lanka').
    """
    if not location_str:
        return "Unknown", "Unknown"
    
    loc_lower = location_str.lower()
    for district, data in SRI_LANKA_DISTRICTS.items():
        if district.lower() in loc_lower:
            return district, data["province"]
        for term in data["search_terms"]:
            term_key = term.split(",")[0].strip().lower()
            if term_key in loc_lower:
                return district, data["province"]
                
    return "Other", "Sri Lanka"

DISTRICT_CENTROIDS = {
    # 25 Administrative District Capitals & Centroids
    "galle": (6.0535, 80.2210),
    "matara": (5.9549, 80.5550),
    "hambantota": (6.1246, 81.1185),
    "colombo": (6.9271, 79.8612),
    "gampaha": (7.0840, 79.9939),
    "kalutara": (6.5854, 79.9607),
    "kandy": (7.2906, 80.6337),
    "nuwara eliya": (6.9497, 80.7891),
    "matale": (7.4675, 80.6234),
    "anuradhapura": (8.3114, 80.4037),
    "polonnaruwa": (7.9403, 81.0188),
    "trincomalee": (8.5874, 81.2152),
    "batticaloa": (7.7310, 81.6747),
    "ampara": (7.2912, 81.6724),
    "jaffna": (9.6615, 80.0255),
    "mannar": (8.9810, 79.9044),
    "vavuniya": (8.7542, 80.4982),
    "kilinochchi": (9.3803, 80.3770),
    "mullaitivu": (9.2671, 80.8142),
    "badulla": (6.9934, 81.0550),
    "monaragala": (6.8728, 81.3507),
    "ratnapura": (6.6828, 80.3992),
    "kegalle": (7.2513, 80.3464),
    "puttalam": (8.0362, 79.8283),
    "kurunegala": (7.4863, 80.3623),
    # Key Tourism Hotspots & Sub-regions
    "galle fort": (6.0270, 80.2170),
    "unawatuna": (6.0104, 80.2494),
    "hikkaduwa": (6.1408, 80.1009),
    "ahangama": (5.9723, 80.3644),
    "mirissa": (5.9482, 80.4570),
    "weligama": (5.9723, 80.4285),
    "dikwella": (5.9622, 80.6975),
    "tangalle": (6.0244, 80.7941),
    "yala": (6.3683, 81.5208),
    "tissamaharama": (6.2803, 81.2882),
    "negombo": (7.2088, 79.8358),
    "bentota": (6.4258, 79.9959),
    "beruwala": (6.4786, 79.9828),
    "sigiriya": (7.9570, 80.7603),
    "dambulla": (7.8567, 80.6483),
    "ella": (6.8667, 81.0466),
    "haputale": (6.7681, 80.9506),
    "bandarawela": (6.8288, 80.9858),
    "hatton": (6.8916, 80.5966),
    "arugam bay": (6.8415, 81.8340),
    "pasikuda": (7.9234, 81.5644),
    "nilaveli": (8.6874, 81.1895),
    "uppuveli": (8.6167, 81.2167),
    "kataragama": (6.4136, 81.3325),
    "kalpitiya": (8.2294, 79.7688),
    "kitulgala": (6.9906, 80.4144)
}

def get_fallback_coordinates(city: str, district: str):
    """Return realistic GPS coordinates for a town or district in Sri Lanka."""
    city_clean = city.strip().lower() if city else ""
    dist_clean = district.strip().lower() if district else ""
    
    if city_clean in DISTRICT_CENTROIDS:
        return DISTRICT_CENTROIDS[city_clean]
    
    for k, coords in DISTRICT_CENTROIDS.items():
        if k in city_clean or city_clean in k:
            return coords
            
    if dist_clean in DISTRICT_CENTROIDS:
        return DISTRICT_CENTROIDS[dist_clean]
        
    for k, coords in DISTRICT_CENTROIDS.items():
        if k in dist_clean or dist_clean in k:
            return coords
            
    # Default to Colombo (Sri Lanka commercial capital)
    return (6.9271, 79.8612)

