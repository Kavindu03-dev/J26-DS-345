"""
Gap 1: the tourist's sentence -> structured request (Output 1).

The fine-tuned model (hotel-nlp-v2 in Ollama) copies what the tourist said into JSON; the rule check in
run_stage1.py fixes dates, money, health and spelling (same code as the tests: 98.7% on the practice
sentences, 98.5% on the hard sentences).

One extra rule here: things the tourist says they DON'T need ("we don't need breakfast", "no need for a pool").
The model's JSON has no place for these, so the code finds them. They are used for the personal fair price
("about Rs 4,500 of this price is for the pool you don't need").
"""
import re
import time
from datetime import date
from pathlib import Path

from .run_stage1 import RATES_FIXED, apply_rules, call_ollama, clean, live_rates, parse_json

PROMPT_FILE = Path(__file__).with_name("ft_prompt.txt")

NOT_NEEDED_WORDS = {
    "pool": r"(?:swimming\s+)?pools?",
    "bar": r"bars?",
    "breakfast": r"breakfasts?|meals?",
    "beachfront": r"beach\s*front|beach",
}
NEGATION = (r"(?:don'?t|do not|doesn'?t|does not|won'?t|will not)\s+(?:really\s+)?(?:need|want|care about|use)"
            r"|no need (?:for|of)|not interested in|without|skip(?:ping)?|not (?:need|needed|necessary)")


def find_not_needed(sentence):
    """Facilities the tourist says they don't need: ['breakfast', 'pool', ...].
    Looks at the words after "don't need / no need for / without ..." up to the next comma, full stop or "but"."""
    found = []
    s = sentence.lower()
    if re.search(r"\broom[\s-]only\b", s):
        found.append("breakfast")
    any_word = "|".join(NOT_NEEDED_WORDS.values())
    for m in re.finditer(NEGATION, s):
        clause = re.split(r"[.;!?]|\bbut\b|\bhowever\b|\bthough\b", s[m.end():])[0][:80]
        parts = clause.split(",")
        kept = [parts[0]]
        for part in parts[1:]:          # "a pool, a bar or breakfast": keep going while the list names facilities
            if re.search(rf"\b(?:{any_word})\b", kept[-1]) and re.search(rf"\b(?:{any_word})\b", part):
                kept.append(part)
            else:
                break
        clause = ",".join(kept)
        for facility, words in NOT_NEEDED_WORDS.items():
            if facility not in found and re.search(rf"\b(?:{words})\b", clause):
                found.append(facility)
    return found


def understand(sentence, model="hotel-nlp-v2", url="http://localhost:11434", today=None, rates=None,
               live=True):
    """Sentence -> (request, notes, seconds). request = Output 1 + "not_needed"."""
    today = today or date.today().isoformat()
    rate_info = "fixed test rates"
    if rates is None:
        if live:
            rates, rate_info = live_rates()
        else:
            rates = RATES_FIXED
    start = time.time()
    prompt = PROMPT_FILE.read_text(encoding="utf-8").replace("{today}", today)
    raw = call_ollama(url, model, prompt, sentence)
    pred, _removed = clean(parse_json(raw))
    final, notes = apply_rules(pred, sentence, today, rates)
    not_needed = find_not_needed(sentence)
    if not_needed:
        for f in not_needed:
            wish = "beach" if f == "beachfront" else f
            if wish in final["hotel"]["wishes"]:
                final["hotel"]["wishes"].pop(wish)
                notes.append(f"wish '{wish}' removed (tourist said they don't need it)")
        notes.append("not needed: " + ", ".join(not_needed))
    final["not_needed"] = not_needed
    return final, notes, round(time.time() - start, 1), rate_info
