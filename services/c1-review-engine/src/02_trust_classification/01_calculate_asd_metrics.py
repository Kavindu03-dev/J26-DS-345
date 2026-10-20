"""
Pipeline Stage 02: Trust Classification & Authenticity Engine.
Step 01: Calculate Aspect Specificity Density (ASD) Metrics.
Research Component 1: Trust-Aware Review Analysis & Aspect Sentiment Engine (J26-DS-345)
Candidate: DISSANAYAKE L.G.D.P.M. (IT23252554)
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.asd_calculator import AspectSpecificityDensityCalculator

def demo_asd():
    calc = AspectSpecificityDensityCalculator()
    print("=" * 80)
    print("   STAGE 02: ASPECT SPECIFICITY DENSITY (ASD) CALCULATION DEMO")
    print("=" * 80)

    samples = [
        "The egg hoppers at breakfast were warm and the split AC in room 204 was ice cold.",
        "A heavenly oasis with delightful ambiance, truly an unforgettable paradise of sheer perfection."
    ]

    for s in samples:
        res = calc.calculate_asd(s)
        print(f"\nReview Text : \"{s}\"")
        print(f"ASD Score   : {res['asd_score']:.4f}")
        print(f"Entities    : {res['concrete_entities_count']} -> {res['detected_entities']}")
        print(f"Fluff Count : {res['generic_fluff_count']} -> {res['detected_fluff']}")
        print(f"Spam Flag   : {'DECEPTIVE / AI-SPAM' if res['is_likely_ai_spam'] else 'AUTHENTIC'}")
    print("=" * 80)

if __name__ == "__main__":
    demo_asd()
