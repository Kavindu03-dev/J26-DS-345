"""
Ground-Truth Dataset Builder & Stratified Sampler for Research Component 1.
Project ID: J26-DS-345
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)

Constructs the gold-standard benchmark dataset (WBS 2.3 / Milestone M3-M4)
by performing stratified sampling across the 39,399 Sri Lankan hotel reviews,
incorporating Aspect Specificity Density (ASD) scores and authentic vs. deceptive spam tags.
"""

import csv
import sys
import random
from pathlib import Path
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
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

        with open(MASTER_DATASET_FILE, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = (row.get("review_text") or "").strip()
                if not text or len(text.split()) < 4:
                    continue

                clauses = self.absa_engine.split_into_clauses_and_sentences(text)
                for clause in clauses:
                    if len(clause.split()) < 4 or len(clause.split()) > 35:
                        continue

                    aspects = self.absa_engine.detect_aspects_in_clause(clause)
                    for asp in aspects:
                        reviews_by_aspect[asp].append({
                            "hotel_id": row.get("hotel_id", ""),
                            "hotel_name": row.get("hotel_name", ""),
                            "district": row.get("district", ""),
                            "platform": row.get("source_platform", "Booking.com"),
                            "review_id": row.get("review_id", ""),
                            "rating_score": row.get("rating_score", "8.0"),
                            "clause_text": clause,
                            "aspect": asp
                        })

        print(f"[+] Aspect Clause Candidates Gathered:")
        for asp, clauses in reviews_by_aspect.items():
            print(f"    - {asp:12}: {len(clauses)} candidate clauses")

        # Stratified sampling: ~60 samples per aspect category = 360 authentic samples
        samples_per_aspect = self.target_samples_count // len(ASPECT_CATEGORIES)
        final_dataset = []
        sample_id = 1

        for asp in ASPECT_CATEGORIES:
            candidates = reviews_by_aspect[asp]
            if len(candidates) > samples_per_aspect:
                chosen = random.sample(candidates, samples_per_aspect)
            else:
                chosen = candidates

            for c in chosen:
                score, sentiment = self.absa_engine.score_clause_sentiment(c["clause_text"])
                asd_res = self.asd_calculator.calculate_asd(c["clause_text"])

                final_dataset.append({
                    "sample_id": f"SMP_{sample_id:04d}",
                    "hotel_id": c["hotel_id"],
                    "hotel_name": c["hotel_name"],
                    "district": c["district"],
                    "source_platform": c["platform"],
                    "review_sentence": c["clause_text"],
                    "aspect_category": asp,
                    "sentiment_label": sentiment,
                    "aspect_specificity_density": asd_res["asd_score"],
                    "concrete_entities_count": asd_res["concrete_entities_count"],
                    "generic_fluff_count": asd_res["generic_fluff_count"],
                    "authenticity_label": "Authentic",
                    "is_ai_generated_spam": 0
                })
                sample_id += 1

        # Add balanced AI-Generated / Astroturfing Spam samples
        print(f"[+] Injecting balanced AI-generated spam samples...")
        for spam_text, spam_asp, spam_sent in AI_GENERATED_SPAM_CORPUS * 4:  # 40 spam samples
            asd_res = self.asd_calculator.calculate_asd(spam_text)
            final_dataset.append({
                "sample_id": f"SMP_{sample_id:04d}",
                "hotel_id": "SYN_AI_HOTEL_001",
                "hotel_name": "Synthesized AI Hotel Benchmark",
                "district": "Synthetic",
                "source_platform": "AI-Generated",
                "review_sentence": spam_text,
                "aspect_category": spam_asp,
                "sentiment_label": spam_sent,
                "aspect_specificity_density": asd_res["asd_score"],
                "concrete_entities_count": asd_res["concrete_entities_count"],
                "generic_fluff_count": asd_res["generic_fluff_count"],
                "authenticity_label": "Deceptive / AI-Spam",
                "is_ai_generated_spam": 1
            })
            sample_id += 1

        # Shuffle to mix authentic and synthetic
        random.shuffle(final_dataset)
        return final_dataset

    def export_dataset(self, dataset: List[Dict[str, Any]]):
        """Exports the annotated ground-truth dataset to CSV."""
        if not dataset:
            print("[!] No records to export.")
            return

        OUTPUT_GROUND_TRUTH_FILE.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "sample_id", "hotel_id", "hotel_name", "district", "source_platform",
            "review_sentence", "aspect_category", "sentiment_label",
            "aspect_specificity_density", "concrete_entities_count",
            "generic_fluff_count", "authenticity_label", "is_ai_generated_spam"
        ]

        with open(OUTPUT_GROUND_TRUTH_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(dataset)

        print("\n" + "=" * 75)
        print(f"[OK] GROUND-TRUTH BENCHMARK DATASET GENERATED SUCCESSFULLY!")
        print(f"   Total Annotated Sentences : {len(dataset)}")
        print(f"   Output File Path          : {OUTPUT_GROUND_TRUTH_FILE}")
        print("=" * 75)


def main():
    builder = GroundTruthBuilder(target_samples_count=360)
    data = builder.sample_and_annotate()
    builder.export_dataset(data)


if __name__ == "__main__":
    main()
