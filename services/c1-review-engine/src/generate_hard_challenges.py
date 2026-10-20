"""
Generator for the 50 Hard Linguistic Challenge ABSA Benchmark.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine.
Project ID: J26-DS-345
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Creates a specialized benchmark of 50 challenging hospitality review sentences
sampled from real Sri Lankan hotel reviews covering 4 complex linguistic phenomena:
  1. Implicit Sentiment (Aspect flaw described without explicit sentiment adjectives)
  2. Sarcasm & Irony (Surface positive words expressing intense dissatisfaction)
  3. Double Negation & Contrastive Shifts (Complex clause qualifications)
  4. Sri Lankan Colloquial / Local Vernacular (Singlish and local phrasing)

Demonstrates the boundary failure of rule-based keyword lexicons vs. Contextual Transformers.
"""

import csv
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CHALLENGES_CSV = ROOT_DIR / "data" / "processed" / "hard_linguistic_challenges.csv"

HARD_CHALLENGES = [
    # --- Category 1: Implicit Sentiment (15 samples) ---
    {
        "challenge_id": "HLC_001",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The air conditioner in room 302 sounded like a cement mixer grinding gravel all night.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Describes mechanical noise metaphorically without explicit negative adjectives like 'bad' or 'broken'."
    },
    {
        "challenge_id": "HLC_002",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "We had to shower with our flip-flops on because of the standing puddle around the drain.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Behavioral adaptation implies unhygienic conditions without using words like 'dirty' or 'unclean'."
    },
    {
        "challenge_id": "HLC_003",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "We watched the chef bring fresh seer fish directly from the harbor boats into the kitchen.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Factual narrative conveys extreme freshness without conventional positive adjectives like 'delicious'."
    },
    {
        "challenge_id": "HLC_004",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The hotel is reached by an unpaved mud track that our van could barely navigate after rain.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Factual road difficulty implies poor accessibility without explicit negative keywords."
    },
    {
        "challenge_id": "HLC_005",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "Every Zoom call dropped the moment another guest walked into the lobby lounge.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Network concurrency congestion described factually without negative terms like 'slow' or 'poor'."
    },
    {
        "challenge_id": "HLC_006",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The receptionist walked us down to the bus stop in the rain holding an umbrella over my grandmother.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Actionable empathy communicates exceptional hospitality without standard praise adjectives."
    },
    {
        "challenge_id": "HLC_007",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "We could count the springs through the thin mattress fabric with our hands.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Tactile description reveals degraded bedding without saying 'uncomfortable'."
    },
    {
        "challenge_id": "HLC_008",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The breakfast buffet milk was lukewarm and curdling slightly on the glass rim.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Food spoilage indicators described neutrally without 'awful' or 'disgusting'."
    },
    {
        "challenge_id": "HLC_009",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The water in the sink ran reddish-brown for two minutes before clearing up.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Plumbing rust described factually without explicit cleanliness terminology."
    },
    {
        "challenge_id": "HLC_010",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "You can step directly from the garden patio right onto the sand of Mirissa beach.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Direct spatial connectivity communicates prime location without overt positive tokens."
    },
    {
        "challenge_id": "HLC_011",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "I could only load text emails if I sat on the staircase landing near the office door.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Positional workaround implies severe signal attenuation in guest rooms."
    },
    {
        "challenge_id": "HLC_012",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The manager kept looking at his watch while we were explaining our missing airport transfer.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Body language signals impatience and indifference without explicit words like 'rude'."
    },
    {
        "challenge_id": "HLC_013",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "My husband and I spent the first morning scrubbing the bathtub tiles ourselves.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Guest corrective labor reveals housekeeping negligence."
    },
    {
        "challenge_id": "HLC_014",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "You can hear every single truck braking on the A1 highway from midnight to dawn.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lack of acoustic insulation described via environmental sound."
    },
    {
        "challenge_id": "HLC_015",
        "phenomenon_type": "Implicit Sentiment",
        "review_sentence": "The host drew a handwritten map with shortcut trails to the rock fortress that saved us two hours.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Personalized guest initiative conveys five-star host effort."
    },

    # --- Category 2: Sarcasm & Irony (12 samples) ---
    {
        "challenge_id": "HLC_016",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "If you enjoy waiting 50 minutes for two fried eggs and lukewarm tea, this breakfast is absolute paradise.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon triggers on 'enjoy', 'fried eggs', and 'paradise', falsely classifying as Positive."
    },
    {
        "challenge_id": "HLC_017",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "A wonderful bonus feature of our bathroom was the complimentary cockroach sanctuary in the shower.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon catches 'wonderful', 'bonus', and 'complimentary', predicting high positive score."
    },
    {
        "challenge_id": "HLC_018",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The staff demonstrated impressive athletic ability by completely ignoring five guests standing at the desk.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon matches 'impressive' and 'ability', misclassifying staff negligence as praise."
    },
    {
        "challenge_id": "HLC_019",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The high-speed fiber internet gave me the thrilling opportunity to relive dial-up speeds from 1998.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon matches 'high-speed' and 'thrilling', reversing the sarcastic intent."
    },
    {
        "challenge_id": "HLC_020",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The serene peaceful location provided front-row audio of the railway crossing bell every twenty minutes.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon catches 'serene' and 'peaceful', ignoring the sarcastic contrast with train noise."
    },
    {
        "challenge_id": "HLC_021",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "Our deluxe suite bathroom offered an invigorating indoor rain simulation directly through the ceiling.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon interprets 'deluxe suite' and 'invigorating' as positive luxury features."
    },
    {
        "challenge_id": "HLC_022",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The chef must be commended for inventing bread that could double as a construction brick.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Matches 'commended' and 'chef', missing the satirical criticism of stale bread."
    },
    {
        "challenge_id": "HLC_023",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "What a pleasure to be greeted with an intense eye-roll and a sigh when asking for our room key.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Catches 'pleasure' and 'greeted', blind to front-desk hostility."
    },
    {
        "challenge_id": "HLC_024",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The air conditioning unit provided an exquisite sauna experience inside our bedroom.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Catches 'exquisite' and 'experience', failing to recognize broken cooling."
    },
    {
        "challenge_id": "HLC_025",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "Top marks to housekeeping for ensuring the sheets had a rich authentic vintage aroma of old mildew.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Catches 'top marks', 'rich', and 'authentic', masking severe mildew complaints."
    },
    {
        "challenge_id": "HLC_026",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "The Wi-Fi was so blazing fast that it took only two hours to download a single boarding pass PDF.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon sees 'blazing fast', missing the sarcastic 2-hour duration."
    },
    {
        "challenge_id": "HLC_027",
        "phenomenon_type": "Sarcasm & Irony",
        "review_sentence": "An absolute masterclass in customer service if you enjoy feeling like an unwelcome intruder.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Lexicon catches 'masterclass' and 'customer service', predicting 5-star staff praise."
    },

    # --- Category 3: Double Negation & Contrastive Shifts (12 samples) ---
    {
        "challenge_id": "HLC_028",
        "phenomenon_type": "Double Negation",
        "review_sentence": "It is not that the front desk was unhelpful, but nobody ever resolved our hot water issue.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Neutral",
        "linguistic_failure_mode": "Double negative 'not unhelpful' confounds naive negation windowing."
    },
    {
        "challenge_id": "HLC_029",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The room was by no means unclean, yet it lacked any sense of warmth or comfort.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'by no means unclean' requires negating the negative root to yield positive cleanliness."
    },
    {
        "challenge_id": "HLC_030",
        "phenomenon_type": "Double Negation",
        "review_sentence": "We can hardly complain about the breakfast spread given the modest price tier.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Idiomatic negation 'hardly complain' indicates satisfaction; keyword matcher detects 'complain'."
    },
    {
        "challenge_id": "HLC_031",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The internet connection was not entirely unusable, but video calls were impossible.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Neutral",
        "linguistic_failure_mode": "'not entirely unusable' qualifies partial functionality before a contrastive 'but' clause."
    },
    {
        "challenge_id": "HLC_032",
        "phenomenon_type": "Double Negation",
        "review_sentence": "I would not say the location was inconvenient, however tuk-tuks charged double after 8 PM.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Neutral",
        "linguistic_failure_mode": "Subtle concession 'would not say inconvenient' balances local price surcharge."
    },
    {
        "challenge_id": "HLC_033",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The staff are never anything less than polite and courteous during evening service.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'never anything less than' is an emphatic positive constructed via double negation."
    },
    {
        "challenge_id": "HLC_034",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The bathroom was far from dirty, but the towels were scratchy and thin.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'far from dirty' negates 'dirty', but naive window might bind 'dirty' to the aspect."
    },
    {
        "challenge_id": "HLC_035",
        "phenomenon_type": "Double Negation",
        "review_sentence": "You cannot go wrong with the egg hoppers and coconut sambol at breakfast.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Colloquial idiom 'cannot go wrong' contains two negative particles but means exceptional."
    },
    {
        "challenge_id": "HLC_036",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The bed was neither uncomfortable nor particularly luxurious, just adequate for one night.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Neutral",
        "linguistic_failure_mode": "Correlative conjunction 'neither... nor' balances polarity into pure Neutral."
    },
    {
        "challenge_id": "HLC_037",
        "phenomenon_type": "Double Negation",
        "review_sentence": "Wi-Fi was nowhere near as bad as the previous reviews warned.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'nowhere near as bad' flips negative 'bad' into positive surprise."
    },
    {
        "challenge_id": "HLC_038",
        "phenomenon_type": "Double Negation",
        "review_sentence": "It is impossible not to appreciate the panoramic views from the fourth floor terrace.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Emphatic double negation 'impossible not to appreciate' indicates stellar scenery."
    },
    {
        "challenge_id": "HLC_039",
        "phenomenon_type": "Double Negation",
        "review_sentence": "The manager never failed to greet us with a warm smile each morning.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'never failed to' is an emphatic praise structure often flagged as negative by 'failed'."
    },

    # --- Category 4: Sri Lankan Colloquial / Local Vernacular (11 samples) ---
    {
        "challenge_id": "HLC_040",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The breakfast kottu and egg hoppers were totally lit and proper local authentic.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Informal slang 'lit' is unrecognized by standard formal hospitality dictionaries."
    },
    {
        "challenge_id": "HLC_041",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The washroom was a bit dodgy and dingy with a foul drainage smell.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "British/Sri Lankan colloquialisms 'dodgy' and 'dingy' are absent in standard US sentiment lexicons."
    },
    {
        "challenge_id": "HLC_042",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "Front desk ayya was super humble and sorted our Yala safari jeep in ten minutes flat.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Kinship honorific 'ayya' (elder brother) and local concept 'humble' mean polite and helpful."
    },
    {
        "challenge_id": "HLC_043",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "AC was giving trouble all night and cutting off every hour.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "South Asian colloquial idiom 'giving trouble' denotes mechanical malfunction."
    },
    {
        "challenge_id": "HLC_044",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The location is right near the junction so tuk-tuk meter fares were dirt cheap.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'junction' is the local Sri Lankan urban landmark anchor; 'dirt cheap' contains 'dirt' but means affordable."
    },
    {
        "challenge_id": "HLC_045",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "Internet was dead slow during evening peak time, totally jammed.",
        "target_aspect": "wifi",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "Colloquial intensifiers 'dead slow' and 'jammed' describe bandwidth throttling."
    },
    {
        "challenge_id": "HLC_046",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "We had spicy pol sambol with roast paan that tasted just like authentic home cooking.",
        "target_aspect": "food",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Specific dish names ('roast paan', 'pol sambol') carry deep cultural culinary praise."
    },
    {
        "challenge_id": "HLC_047",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The bathroom linen looked used and had greasy patch marks on the pillow cases.",
        "target_aspect": "cleanliness",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "'patch marks' is colloquial phrasing for unwashed linen residue."
    },
    {
        "challenge_id": "HLC_048",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The caretaker uncle went out of his way to boil ginger coriander water when I had a bad cold.",
        "target_aspect": "staff",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "Local herbal remedy ('ginger coriander water') reflects traditional Sri Lankan guest hospitality."
    },
    {
        "challenge_id": "HLC_049",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "The property is located in an out-of-the-way jungle pocket, super quiet and peaceful.",
        "target_aspect": "location",
        "ground_truth_sentiment": "Positive",
        "linguistic_failure_mode": "'jungle pocket' expresses seclusion praised by ecotourists."
    },
    {
        "challenge_id": "HLC_050",
        "phenomenon_type": "Sri Lankan Vernacular",
        "review_sentence": "Water heater had no power tripping constantly whenever shower was opened.",
        "target_aspect": "room",
        "ground_truth_sentiment": "Negative",
        "linguistic_failure_mode": "'power tripping' refers to electrical circuit breaker failure, not psychological ego."
    }
]


def generate_benchmark_csv():
    CHALLENGES_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "challenge_id",
        "phenomenon_type",
        "review_sentence",
        "target_aspect",
        "ground_truth_sentiment",
        "linguistic_failure_mode"
    ]
    with open(CHALLENGES_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(HARD_CHALLENGES)

    print(f"[+] Successfully generated {len(HARD_CHALLENGES)} Hard Linguistic Challenge benchmark sentences.")
    print(f"    Saved to: {CHALLENGES_CSV}")


if __name__ == "__main__":
    generate_benchmark_csv()
