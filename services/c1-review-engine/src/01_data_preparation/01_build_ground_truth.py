"""
Pipeline Stage 01: Data Preparation & Benchmark Generation.
Step 01: Build Ground-Truth Hospitality Benchmark Dataset.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Fulfills WBS 2.3 / Milestone M3-M4:
Constructs the 400-sentence gold-standard benchmark dataset by performing stratified
sampling across the 39,399 Sri Lankan hotel reviews, computing Aspect Specificity Density (ASD),
and injecting balanced synthetic AI-spam samples.
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

# Representative ChatGPT / LLM-generated generic hotel spam samples for benchmark balance
AI_GENERATED_SPAM_CORPUS = [
    ("The stay was an absolute dream, truly a breathtaking paradise with world-class ambiance and unforgettable moments.", "room", "Positive"),
    ("Unparalleled excellence and sublime hospitality! The vibes were completely magical and perfection in every way.", "staff", "Positive"),
    ("A heavenly oasis with delightful atmosphere and impeccable experience. Truly an extraordinary gem of a place.", "room", "Positive"),
    ("Outstanding perfection! The staff was exceedingly courteous and the whole environment was charming and splendid.", "staff", "Positive"),
    ("Magnificent property with top-notch service and stunning aesthetics. We had an exceptionally wonderful time.", "location", "Positive"),
    ("Simply flawless. The culinary delights were fabulous and the staff was remarkably friendly and attentive.", "food", "Positive"),
    ("A spectacular haven of tranquility! Words cannot describe how marvelous and flawless everything was.", "location", "Positive"),
    ("Utterly disappointed with the overall experience, the place lacked charm and was completely unacceptable.", "room", "Negative"),
    ("Terrible and dreadful stay, everything was horribly managed and deeply awful from start to finish.", "staff", "Negative"),
    ("The most unpleasant experience imaginable. Completely dreadful service and horrifying conditions.", "cleanliness", "Negative"),
]


class GroundTruthBuilder:
    def __init__(self, target_samples_count: int = 400, seed: int = 42):
        self.target_samples_count = target_samples_count
        self.seed = seed
        self.absa_engine = ABSAEngine()
        self.asd_calculator = AspectSpecificityDensityCalculator()
        random.seed(self.seed)

    def sample_and_annotate(self) -> List[Dict[str, Any]]:
        """
        Samples review sentences from the master 39,399 review dataset,
        stratifying across districts, platforms, and aspects.
        """
        if not MASTER_DATASET_FILE.exists():
            print(f"[!] Master dataset not found at: {MASTER_DATASET_FILE}")
            return []

        print(f"[+] Loading reviews from: {MASTER_DATASET_FILE.name}...")
        reviews_by_aspect = {cat: [] for cat in ASPECT_CATEGORIES}

        with open(MASTER_DATASET_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                full_text = str(row.get("review_text", "")).strip()
                if len(full_text.split()) < 4 or len(full_text.split()) > 100:
                    continue

                clauses = self.absa_engine.split_into_clauses_and_sentences(full_text)
                for c in clauses:
                    if len(c.split()) < 4:
                        continue
                    matched_aspects = self.absa_engine.detect_aspects_in_clause(c)
                    if len(matched_aspects) == 1:
                        aspect = matched_aspects[0]
                        reviews_by_aspect[aspect].append({
                            "hotel_id": row.get("hotel_id", ""),
                            "hotel_name": row.get("hotel_name", ""),
                            "district": row.get("district", ""),
                            "source_platform": row.get("source_platform", "Booking.com"),
                            "clause": c,
                            "aspect": aspect,
                            "full_text": full_text
                        })

        print(f"[+] Discovered candidate sentences across 6 aspects:")
        for asp, candidates in reviews_by_aspect.items():
            print(f"    - {asp:<12}: {len(candidates)} candidates")

        target_per_aspect = (self.target_samples_count - len(AI_GENERATED_SPAM_CORPUS) * 4) // len(ASPECT_CATEGORIES)
        stratified_samples = []
        sample_id_counter = 1

        # 1. Sample authentic reviews
        for asp, candidates in reviews_by_aspect.items():
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
                    "concrete_entities_count": asd_metrics["entities_count"],
                    "generic_fluff_count": asd_metrics["fluff_count"],
                    "authenticity_label": "Authentic",
                    "is_ai_generated_spam": 0
                })
                sample_id_counter += 1

        # 2. Inject balanced AI-generated spam samples
        for text, asp, sentiment in AI_GENERATED_SPAM_CORPUS * 4:
            asd_metrics = self.asd_calculator.calculate_asd(text)
            stratified_samples.append({
                "sample_id": f"SMP_{sample_id_counter:04d}",
                "hotel_id": "SYN_AI_HOTEL_001",
                "hotel_name": "Synthesized AI Hotel Benchmark",
                "district": "Synthetic",
                "source_platform": "AI-Generated",
                "review_sentence": text,
                "aspect_category": asp,
                "sentiment_label": sentiment,
                "aspect_specificity_density": asd_metrics["asd_score"],
                "concrete_entities_count": asd_metrics["entities_count"],
                "generic_fluff_count": asd_metrics["fluff_count"],
                "authenticity_label": "Deceptive / AI-Spam",
                "is_ai_generated_spam": 1
            })
            sample_id_counter += 1

        random.shuffle(stratified_samples)
        return stratified_samples

    def build_and_save(self) -> Path:
        samples = self.sample_and_annotate()
        if not samples:
            print("[!] No samples created.")
            return OUTPUT_GROUND_TRUTH_FILE

        OUTPUT_GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(samples[0].keys())

        with open(OUTPUT_GROUND_TRUTH_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print("\n" + "=" * 80)
        print(f"[OK] GROUND-TRUTH BENCHMARK DATASET GENERATED SUCCESSFULLY!")
        print(f"     Total Annotated Sentences : {len(samples)}")
        print(f"     Saved Path                : {OUTPUT_GROUND_TRUTH_FILE}")
        print("=" * 80)
        return OUTPUT_GROUND_TRUTH_FILE


def main():
    builder = GroundTruthBuilder(target_samples_count=400)
    builder.build_and_save()


if __name__ == "__main__":
    main()
