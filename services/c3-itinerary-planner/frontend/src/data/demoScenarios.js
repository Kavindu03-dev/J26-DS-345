// Hotel identity/location values copied from C1 data/hotel_aspect_profiles.csv.
// Predefined conversation examples, not live C2/C4 results or computed rankings.
function hotel(rank, hotel_id, name, town, stay_in, latitude, longitude) {
  return {
    rank, hotel_id, name, town, stay_in, latitude, longitude, nights: 3,
    price_per_night_lkr: null, total_hotel_cost_lkr: null, budget_left_lkr: null,
    price_label: null, book_on: null, booking_url: null,
    nearest_hospital_km: null, photo_url: null,
  };
}

function scenario(id, label, conversation, interests, hotels) {
  return {
    id, label, conversation,
    for_c3: {
      trip: {
        stay_in: [label], want_to_visit: [], check_in: '2026-12-20',
        check_out: '2026-12-23', nights: 3, dates_fixed: false,
        adults: 2, children: 1, travel_style: 'family', budget_total_lkr: null,
      },
      interests, health: { consent_given: false }, recommended_hotels: hotels,
    },
  };
}

export const DEMO_SCENARIOS = [
  scenario('ella', 'Ella',
    'I am planning a three-day family trip to Ella. We enjoy nature, wildlife and hiking.',
    ['nature', 'wildlife', 'hiking'], [
      hotel(1, 'HTL_ELL_WINWOODE_0DAF', 'Winwood Ella', 'Ella', 'Ella', 6.863345, 81.049326),
      hotel(2, 'HTL_ELL_SATHKARA_2DE8', 'Sathkara Resort Ella', 'Ella', 'Ella', 6.864756, 81.048041),
    ]),
  scenario('kandy', 'Kandy',
    'I want a three-day trip to Kandy with my family. We like cultural, religious and nature places.',
    ['culture', 'religious', 'nature'], [
      hotel(1, 'HTL_KAN_GRANDSER_A9EE', 'Grand Serendib Hotel', 'Kandy', 'Kandy', 7.288638, 80.636835),
      hotel(2, 'HTL_KAN_SWPECOLO_3F4C', 'SWP Eco Lodge, Kandy', 'Kandy', 'Kandy', 7.291011, 80.637118),
    ]),
  scenario('colombo', 'Colombo',
    'Our family has three days in Colombo. We would like city experiences, food and cultural places.',
    ['nightlife', 'food', 'culture'], [
      hotel(1, 'HTL_COL_C1COLOMB_0FA9', 'C 1 Colombo Fort', 'Colombo', 'Colombo', 6.924474, 79.863626),
      hotel(2, 'HTL_COL_EZSTAYGU_C01C', 'Ez Stay Guesthouse', 'Colombo', 'Colombo', 6.925041, 79.857592),
    ]),
  scenario('galle', 'Galle',
    'We want a three-day family trip to Galle, exploring the coast and local history.',
    ['beach', 'culture'], [
      hotel(1, 'HTL_GAL_ORCHARDH_2FE6', 'Orchard House Galle', 'Galle Fort', 'Galle', 6.030118, 80.217461),
      hotel(2, 'HTL_GAL_BRIXIACA_EE10', 'Brixia Cafe & Guest', 'Galle Fort', 'Galle', 6.028325, 80.217354),
    ]),
];

export function createScenarioRequest(scenario, selectedHotelId) {
  if (!scenario.for_c3.recommended_hotels.some((item) => item.hotel_id === selectedHotelId)) {
    throw new Error('Select a recommended hotel before generating an itinerary.');
  }
  return { for_c3: scenario.for_c3, selected_hotel_id: selectedHotelId };
}
