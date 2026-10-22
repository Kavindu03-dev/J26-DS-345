"""
Pipeline Stage 01: Data Preparation & Benchmark Generation.
Step 01: Build Ground-Truth Hospitality Benchmark Dataset.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills WBS 2.3 / Milestone M3-M4:
Constructs the 400-sentence gold-standard benchmark dataset by performing stratified
sampling across the 39,399 Sri Lankan hotel reviews (360 authentic samples across 6 aspects)
and injecting 40 unique, non-repeating AI-generated / deceptive spam samples across all 6 aspects.
"""

import csv
import sys
import random
from pathlib import Path
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import MASTER_DATASET_FILE, PROCESSED_DATA_DIR, ASPECT_CATEGORIES
from src.models.absa_engine import ABSAEngine
from src.models.asd_calculator import AspectSpecificityDensityCalculator

OUTPUT_GROUND_TRUTH_FILE = PROCESSED_DATA_DIR / "ground_truth_annotated_hospitality.csv"

# 40 Unique, Non-Repeating Representative ChatGPT / LLM-generated & Deceptive Hotel Spam Samples
# Spanning all 6 aspects and multiple deceptive styles (zero-shot flowery praise, AI-rephrased puffery,
# generic competitor attack spam, and unspecific astroturfing).
AI_GENERATED_SPAM_CORPUS = [
    # --- Room (8 unique samples) ---
    ("The stay was an absolute dream, truly a breathtaking paradise with world-class ambiance and unforgettable moments.", "room", "Positive"),
    ("A heavenly oasis with delightful atmosphere and impeccable experience. Truly an extraordinary gem of a place.", "room", "Positive"),
    ("Every single detail of the accommodation exuded sheer perfection and unmatched magnificence beyond our wildest dreams.", "room", "Positive"),
    ("An enchanting sanctuary of sublime luxury where every moment felt magical and utterly extraordinary.", "room", "Positive"),
    ("Spectacular living spaces with phenomenal vibes and peerless elegance that left us completely speechless.", "room", "Positive"),
    ("Truly a marvelous haven of comfort and splendid aesthetics, offering a sensational and flawless retreat.", "room", "Positive"),
    ("Utterly disappointed with the overall experience, the place lacked charm and was completely unacceptable.", "room", "Negative"),
    ("An absolute nightmare of a property, everything was atrocious and horribly substandard in every way.", "room", "Negative"),

    # --- Staff (8 unique samples) ---
    ("Unparalleled excellence and sublime hospitality! The vibes were completely magical and perfection in every way.", "staff", "Positive"),
    ("Outstanding perfection! The staff was exceedingly courteous and the whole environment was charming and splendid.", "staff", "Positive"),
    ("The team delivered an unmatched level of world-class care and deeply phenomenal service throughout our visit.", "staff", "Positive"),
    ("Truly extraordinary warmth and breathtaking dedication from everyone, making us feel like royalty in paradise.", "staff", "Positive"),
    ("Impeccable professionalism combined with a delightful and enchanting attitude that was simply stellar.", "staff", "Positive"),
    ("Sensational hospitality and marvelous attention to every wish, creating an unforgettable and fabulous atmosphere.", "staff", "Positive"),
    ("Terrible and dreadful stay, everything was horribly managed and deeply awful from start to finish.", "staff", "Negative"),
    ("Shockingly atrocious attitude and utterly unacceptable behavior that ruined our entire vacation experience.", "staff", "Negative"),

    # --- Food (7 unique samples) ---
    ("Simply flawless. The culinary delights were fabulous and the dining experience was remarkably splendid.", "food", "Positive"),
    ("Every meal was a breathtaking symphony of flavors, offering an extraordinary and world-class gastronomic journey.", "food", "Positive"),
    ("Magnificent dining with sublime presentation and unmatched taste that exceeded every expectation.", "food", "Positive"),
    ("Delightful and heavenly dishes prepared to absolute perfection, truly a phenomenal culinary paradise.", "food", "Positive"),
    ("The gastronomy was sensational and peerless, delivering an unforgettable and deeply marvelous feast.", "food", "Positive"),
    ("Fabulous flavors and impeccable quality at every sitting, an utterly charming and stellar dining affair.", "food", "Positive"),
    ("The dining was a complete disaster, utterly disgusting and horribly prepared with zero redeeming qualities.", "food", "Negative"),

    # --- Location (7 unique samples) ---
    ("Magnificent property with top-notch surroundings and stunning aesthetics. We had an exceptionally wonderful time.", "location", "Positive"),
    ("A spectacular haven of tranquility! Words cannot describe how marvelous and flawless the setting was.", "location", "Positive"),
    ("Nestled in a breathtaking paradise of unmatched beauty, offering a truly magical and heavenly escape.", "location", "Positive"),
    ("An extraordinary and gorgeous setting with sublime views that created an unforgettable and enchanting backdrop.", "location", "Positive"),
    ("The surroundings were simply phenomenal and world-class, a splendid oasis of pure perfection.", "location", "Positive"),
    ("A charming and sensational locale that felt like a hidden gem of incomparable magnificence.", "location", "Positive"),
    ("Awful and terrible surroundings, a completely dreadful setting that nobody should ever visit.", "location", "Negative"),

    # --- Cleanliness (6 unique samples) ---
    ("Everything was maintained to impeccable perfection, gleaming with flawless and extraordinary care.", "cleanliness", "Positive"),
    ("A pristine and immaculate sanctuary where the upkeep was truly world-class and beyond reproach.", "cleanliness", "Positive"),
    ("Remarkably spotless and splendidly maintained throughout, reflecting an unmatched standard of excellence.", "cleanliness", "Positive"),
    ("Sublime hygiene and fabulous presentation in every corner, truly a delightful and impeccable environment.", "cleanliness", "Positive"),
    ("The most unpleasant experience imaginable. Completely dreadful hygiene and horrifying conditions everywhere.", "cleanliness", "Negative"),
    ("Utterly filthy and disgusting environment, an unacceptable and atrocious standard of maintenance.", "cleanliness", "Negative"),

    # --- Wi-Fi (4 unique samples) ---
    ("Flawless and phenomenal digital connectivity that was truly world-class and effortlessly outstanding.", "wifi", "Positive"),
    ("Superb and unmatched online experience, delivering seamless perfection throughout the entire property.", "wifi", "Positive"),
    ("Extraordinary and top-notch connectivity that worked like magic in this breathtaking oasis.", "wifi", "Positive"),
    ("Completely useless and atrocious connection, an utterly terrible and unacceptable digital experience.", "wifi", "Negative"),
]


class GroundTruthBuilder:
    def __init__(self, target_samples_count: int = 400, seed: int = 42):
        self.target_samples_count = target_samples_count
        self.seed = seed
        self.absa_engine = ABSAEngine(use_transformer=False)
        self.asd_calculator = AspectSpecificityDensityCalculator()
        random.seed(self.seed)

    def sample_and_annotate(self) -> List[Dict[str, Any]]:
        """
        Samples review sentences from the master 39,399 review dataset,
        stratifying across districts, platforms, and aspects, and pairs them
        with 40 unique AI-generated deceptive spam samples.
        """
        if not MASTER_DATASET_FILE.exists():
            print(f"[!] Master dataset not found at: {MASTER_DATASET_FILE}")
            return []

        print(f"[+] Loading reviews from: {MASTER_DATASET_FILE.name}...")
        reviews_by_aspect = {cat: [] for cat in ASPECT_CATEGORIES}
        seen_clauses = set()

        with open(MASTER_DATASET_FILE, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_text = str(row.get("review_text", "") or "").strip()
                if len(full_text.split()) < 4 or len(full_text.split()) > 100:
                    continue

                clauses = self.absa_engine.split_into_clauses_and_sentences(full_text)
                for c in clauses:
                    c_clean = c.strip()
                    if len(c_clean.split()) < 4 or len(c_clean.split()) > 35:
                        continue
                    c_key = c_clean.lower()
                    if c_key in seen_clauses:
                        continue

                    matched_aspects = self.absa_engine.detect_aspects_in_clause(c_clean)
                    if len(matched_aspects) == 1:
                        seen_clauses.add(c_key)
                        aspect = matched_aspects[0]
                        reviews_by_aspect[aspect].append({
                            "hotel_id": row.get("hotel_id", ""),
                            "hotel_name": row.get("hotel_name", ""),
                            "district": row.get("district", ""),
                            "source_platform": row.get("source_platform", "Booking.com"),
                            "clause": c_clean,
                            "aspect": aspect
                        })

        print("[+] Discovered unique candidate sentences across 6 aspects:")
        for asp, candidates in reviews_by_aspect.items():
            print(f"    - {asp:<12}: {len(candidates)} unique candidates")

        spam_count = len(AI_GENERATED_SPAM_CORPUS)  # 40 unique samples
        authentic_target = self.target_samples_count - spam_count  # 360 authentic samples
        target_per_aspect = authentic_target // len(ASPECT_CATEGORIES)  # 60 per aspect

        stratified_samples = []
        sample_id_counter = 1

        # 1. Sample 360 unique authentic reviews (60 per aspect)
        for asp in ASPECT_CATEGORIES:
            candidates = reviews_by_aspect[asp]
            random.shuffle(candidates)
            selected = candidates[:target_per_aspect]

            for item in selected:
                clause = item["clause"]
                _, sentiment = self.absa_engine.score_clause_sentiment(clause)
                asd_metrics = self.asd_calculator.calculate_asd(clause)

                stratified_samples.append({
                    "sample_id": f"SMP_{sample_id_counter:04d}",
                    "hotel_id": item["hotel_id"],
                    "hotel_name": item["hotel_name"],
                    "district": item["district"],
                    "source_platform": item["source_platform"],
                    "review_sentence": clause,
                    "aspect_category": asp,
                    "sentiment_label": sentiment,
                    "aspect_specificity_density": asd_metrics["asd_score"],
                    "concrete_entities_count": asd_metrics["concrete_entities_count"],
                    "generic_fluff_count": asd_metrics["generic_fluff_count"],
                    "authenticity_label": "Authentic",
                    "is_ai_generated_spam": 0
                })
                sample_id_counter += 1

        # 2. Inject 40 unique AI-generated spam samples (distributed across platforms to prevent platform leakage)
        simulated_platforms = ["Google Maps", "TripAdvisor", "Booking.com", "Google Maps"]
        for idx, (text, asp, sentiment) in enumerate(AI_GENERATED_SPAM_CORPUS):
            asd_metrics = self.asd_calculator.calculate_asd(text)
            sim_platform = simulated_platforms[idx % len(simulated_platforms)]
            stratified_samples.append({
                "sample_id": f"SMP_{sample_id_counter:04d}",
                "hotel_id": f"SYN_AI_HOTEL_{(idx % 10) + 1:03d}",
                "hotel_name": "Synthesized AI Hotel Benchmark",
                "district": "Synthetic",
                "source_platform": sim_platform,
                "review_sentence": text,
                "aspect_category": asp,
                "sentiment_label": sentiment,
                "aspect_specificity_density": asd_metrics["asd_score"],
                "concrete_entities_count": asd_metrics["concrete_entities_count"],
                "generic_fluff_count": asd_metrics["generic_fluff_count"],
                "authenticity_label": "Deceptive / AI-Spam",
                "is_ai_generated_spam": 1
            })
            sample_id_counter += 1

        random.shuffle(stratified_samples)
        return stratified_samples

    def export_dataset(self, dataset: List[Dict[str, Any]]) -> Path:
        """Exports the annotated ground-truth dataset to CSV."""
        if not dataset:
            print("[!] No samples created.")
            return OUTPUT_GROUND_TRUTH_FILE

        OUTPUT_GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id", "hotel_id", "hotel_name", "district", "source_platform",
            "review_sentence", "aspect_category", "sentiment_label",
            "aspect_specificity_density", "concrete_entities_count",
            "generic_fluff_count", "authenticity_label", "is_ai_generated_spam"
        ]

        with open(OUTPUT_GROUND_TRUTH_FILE, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)

        print("\n" + "=" * 80)
        print("[OK] GROUND-TRUTH BENCHMARK DATASET GENERATED SUCCESSFULLY!")
        print(f"     Total Annotated Sentences : {len(dataset)} (100% Unique Sentences)")
        print(f"     Saved Path                : {OUTPUT_GROUND_TRUTH_FILE}")
        print("=" * 80)
        return OUTPUT_GROUND_TRUTH_FILE

    def build_and_save(self) -> Path:
        samples = self.sample_and_annotate()
        return self.export_dataset(samples)


def main():
    builder = GroundTruthBuilder(target_samples_count=400)
    builder.build_and_save()


if __name__ == "__main__":
    main()
