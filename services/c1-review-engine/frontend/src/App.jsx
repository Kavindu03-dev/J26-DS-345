import React, { useState, useEffect, useMemo } from "react";

const API_BASE = "http://localhost:8001";

// Fallback data in case FastAPI server isn't started yet
const DEFAULT_OVERVIEW = {
  funnel: {
    raw_scraped_reviews: 59889,
    discovered_properties: 1090,
    duplicates_removed: 13298,
    after_deduplication: 46591,
    non_english_removed: 7192,
    final_verified_reviews: 39399,
    final_verified_hotels: 1025,
    districts_covered: 25,
    average_word_count: 64.6,
    platform_distribution: {
      "Booking.com": 32382,
      "Google Maps": 6592,
      "TripAdvisor": 425,
    },
    district_distribution: {
      Galle: 2632, Matara: 2359, Trincomalee: 2352, Badulla: 2245,
      Hambantota: 2224, Kalutara: 2224, Colombo: 2171, Matale: 2162,
      "Nuwara Eliya": 2141, Batticaloa: 1906, Anuradhapura: 1879, Ratnapura: 1783,
      Polonnaruwa: 1650, Puttalam: 1546, Kandy: 1535, Kegalle: 1529,
      Gampaha: 1517, Monaragala: 1283, Jaffna: 1250, Ampara: 1178,
      Kurunegala: 705, Mannar: 505, Kilinochchi: 319, Vavuniya: 250, Mullaitivu: 54,
    },
  },
  cleaning_examples: [
    {
      step: "1. Cross-Platform Deduplication",
      issue: "Exact & near-duplicate review posted on both Booking.com and Google Maps for the same stay.",
      raw_input: "hotel_id: HTL_GAL_012 | author: 'John D.' | text: 'Great location near Galle Fort, breakfast hoppers were tasty.' (x2 rows)",
      cleaned_output: "13,298 duplicate records purged via deterministic hash (hotel_id + normalized_text_prefix + stay_month).",
      status: "Purged (-13,298)",
    },
    {
      step: "2. Non-English Language Filtering",
      issue: "Scraped reviews in Dutch, Italian, German, Russian, or emoji-only strings with < 3 English words.",
      raw_input: "'Fijne accommodatie waar we enorm gastvrij werden ontvangen... De kamer was enorm groot.' (Dutch)",
      cleaned_output: "7,192 non-English / empty rows removed using ASCII ratio + English hospitality stopword validation.",
      status: "Purged (-7,192)",
    },
    {
      step: "3. Ethical PII Anonymization (SHA-256)",
      issue: "Raw reviewer names and profile URLs scraped from public OTA pages expose personal identity.",
      raw_input: "reviewer_name: 'Kasun Perera (Colombo)' -> profile_id: '10492819283'",
      cleaned_output: "author_id: 'USR_317BA56ACF' (Irreversible SHA-256 cryptographic hash; preserves reviewer history count).",
      status: "Anonymized (39,399)",
    },
    {
      step: "4. Multi-Platform Rating & Text Unification",
      issue: "Booking.com uses 1–10 scale with split positive/negative fields; Google Maps & TripAdvisor use 1–5 scale with single text.",
      raw_input: "Booking: rating=9.2, pos='Spacious room, fast wifi', neg='Slow check-in' | Google: rating=4.5/5.0",
      cleaned_output: "Unified 35-field schema: normalized score_5 (1.0–5.0), preserved positive/negative clause priors for ABSA.",
      status: "Standardized (1,025 Hotels)",
    },
  ],
};

const DEFAULT_BENCHMARKS = {
  kappa_agreement: {
    sentiment_kappa: 0.6812,
    sentiment_interpretation: "Substantial Agreement (0.61 - 0.80)",
    sentiment_accuracy_pct: 83.0,
    aspect_kappa: 0.9159,
    aspect_interpretation: "Almost Perfect Agreement (0.81 - 1.00)",
    aspect_accuracy_pct: 93.0,
  },
  absa_comparison: {
    standard_400: [
      { model: "VADER Baseline", accuracy: 74.75, macro_f1: 63.62, latency_ms: 8.2 },
      { model: "Domain Lexical Engine (Tier 1)", accuracy: 96.25, macro_f1: 96.43, latency_ms: 1.2 },
      { model: "RoBERTa-base", accuracy: 99.25, macro_f1: 98.88, latency_ms: 66.4 },
      { model: "DeBERTa-v3-small (Proposed Tier 2)", accuracy: 100.0, macro_f1: 100.0, latency_ms: 128.7 },
    ],
    hard_challenges: [
      { model: "VADER Baseline", accuracy: 32.0, macro_f1: 27.93, sarcasm: 0.0, negation: 50.0, vernacular: 63.6, implicit: 20.0 },
      { model: "Domain Lexical Engine", accuracy: 14.0, macro_f1: 15.68, sarcasm: 0.0, negation: 25.0, vernacular: 27.3, implicit: 6.7 },
      { model: "RoBERTa-base", accuracy: 76.0, macro_f1: 70.91, sarcasm: 66.7, negation: 91.7, vernacular: 72.7, implicit: 73.3 },
      { model: "DeBERTa-v3-small (Proposed)", accuracy: 86.0, macro_f1: 80.21, sarcasm: 83.3, negation: 91.7, vernacular: 90.9, implicit: 80.0 },
    ],
    sample_challenges: [
      {
        challenge_id: "HLC_016",
        phenomenon_type: "Sarcasm & Irony",
        review_sentence: "If you enjoy waiting 50 minutes for two fried eggs and lukewarm tea, this breakfast is absolute paradise.",
        target_aspect: "food",
        ground_truth_sentiment: "Negative",
        linguistic_failure_mode: "Lexicon triggers on 'enjoy' and 'paradise', falsely classifying as Positive.",
      },
      {
        challenge_id: "HLC_017",
        phenomenon_type: "Sarcasm & Irony",
        review_sentence: "A wonderful bonus feature of our bathroom was the complimentary cockroach sanctuary in the shower.",
        target_aspect: "cleanliness",
        ground_truth_sentiment: "Negative",
        linguistic_failure_mode: "Lexicon catches 'wonderful', 'bonus', and 'complimentary', predicting high positive score.",
      },
      {
        challenge_id: "HLC_028",
        phenomenon_type: "Double Negation",
        review_sentence: "It is not that the front desk was unhelpful, but nobody ever resolved our hot water issue.",
        target_aspect: "staff",
        ground_truth_sentiment: "Neutral",
        linguistic_failure_mode: "Double negative 'not unhelpful' confounds naive negation windowing.",
      },
      {
        challenge_id: "HLC_040",
        phenomenon_type: "Sri Lankan Vernacular",
        review_sentence: "Machan the kottu and hoppers at the restaurant were shape, no complaints at all.",
        target_aspect: "food",
        ground_truth_sentiment: "Positive",
        linguistic_failure_mode: "Sri Lankan English slang 'shape' (meaning good/fine) is unknown to standard VADER.",
      },
    ],
  },
  trust_ablation: [
    { config: "Naive Baseline (Word Count Only)", features: 2, roc_auc: 93.0, macro_f1: 34.16, accuracy: 91.0, top_feature: "text_length_words" },
    { config: "ASD & Content-Only (True Novelty)", features: 5, roc_auc: 98.91, macro_f1: 77.31, accuracy: 96.5, top_feature: "generic_fluff_count + ASD" },
    { config: "De-Leaked Hybrid LightGBM (Proposed)", features: 8, roc_auc: 99.9, macro_f1: 91.7, accuracy: 98.5, top_feature: "generic_fluff_count (58.9%) + ASD (7.3%)" },
  ],
  trust_models: {
    lightgbm: { roc_auc: 100.0, accuracy: 98.5, precision: 93.33, recall: 92.5, macro_f1: 92.28 },
    xgboost: { roc_auc: 99.93, accuracy: 98.5, precision: 95.56, recall: 90.0, macro_f1: 91.7 },
  },
  ranking_ablation: {
    total_hotels: 1025,
    total_reviews: 39399,
    spearman_rho: 0.9449,
    kendall_tau: 0.8274,
    mars_positions: 61.26,
    hotels_shifted_ge_10_pct: 67.22,
    hotels_shifted_ge_25_pct: 56.1,
    hotels_shifted_ge_50_pct: 40.59,
    hotels_shifted_ge_100_pct: 21.27,
  },
};

export default function App() {
  const [activeStep, setActiveStep] = useState(1);
  const [overview, setOverview] = useState(DEFAULT_OVERVIEW);
  const [benchmarks, setBenchmarks] = useState(DEFAULT_BENCHMARKS);
  const [apiConnected, setApiConnected] = useState(false);

  useEffect(() => {
    fetch(`${API_BASE}/api/v1/pipeline-overview`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.funnel) {
          setOverview(data);
          setApiConnected(true);
        }
      })
      .catch(() => setApiConnected(false));

    fetch(`${API_BASE}/api/v1/benchmarks`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setBenchmarks(data);
      })
      .catch(() => {});
  }, []);

  const steps = [
    { id: 1, title: "1. Data Scraping & Cleaning", subtitle: "59,889 → 39,399 Reviews" },
    { id: 2, title: "2. Trust & Spam Filter", subtitle: "ASD + LightGBM / XGBoost" },
    { id: 3, title: "3. 6-Aspect Sentiment", subtitle: "Lexical vs. DeBERTa-v3" },
    { id: 4, title: "4. Hotel Rankings & Output", subtitle: "Before vs. After Rank (1,025 Hotels)" },
    { id: 5, title: "5. 80% Viva Progress", subtitle: "Completed vs. Final 20%" },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Top Academic Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40 shadow-sm">
        <div className="w-full px-4 sm:px-6 py-3 flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-slate-900 text-base sm:text-lg">
                Trust-Aware Review & Aspect Sentiment Engine
              </h1>
              <span className="text-xs font-semibold bg-blue-50 text-blue-700 border border-blue-200 px-2.5 py-0.5 rounded-full">
                J26-DS-345
              </span>
            </div>
            <p className="text-xs text-slate-500">
              Candidate: Dissanayake L.G.D.P.M. (IT23252554) • Demonstration UI
            </p>
          </div>
        </div>

        {/* Step-by-Step Pipeline Navigation Bar */}
        <div className="bg-slate-100/80 border-t border-slate-200">
          <div className="w-full px-4 sm:px-6">
            <nav className="grid grid-cols-2 sm:grid-cols-5 gap-1.5 py-2">
              {steps.map((s) => {
                const active = activeStep === s.id;
                return (
                  <button
                    key={s.id}
                    onClick={() => setActiveStep(s.id)}
                    className={`text-left px-3 py-2 rounded-lg border transition-all ${
                      active
                        ? "bg-white border-blue-600 shadow-sm ring-1 ring-blue-600"
                        : "bg-transparent border-transparent hover:bg-white/60 text-slate-600"
                    }`}
                  >
                    <div
                      className={`text-xs font-bold ${
                        active ? "text-blue-700" : "text-slate-800"
                      }`}
                    >
                      {s.title}
                    </div>
                    <div className="text-[11px] text-slate-500 truncate">{s.subtitle}</div>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 w-full px-6 sm:px-8 lg:px-10 py-6">
        {activeStep === 1 && <Step1DataFunnel overview={overview} onNext={() => setActiveStep(2)} />}
        {activeStep === 2 && (
          <Step2TrustAndASD
            benchmarks={benchmarks}
            onPrev={() => setActiveStep(1)}
            onNext={() => setActiveStep(3)}
          />
        )}
        {activeStep === 3 && (
          <Step3AspectSentiment
            benchmarks={benchmarks}
            onPrev={() => setActiveStep(2)}
            onNext={() => setActiveStep(4)}
          />
        )}
        {activeStep === 4 && (
          <Step4RankingsAndProfiles
            benchmarks={benchmarks}
            onPrev={() => setActiveStep(3)}
            onNext={() => setActiveStep(5)}
          />
        )}
        {activeStep === 5 && <Step5VivaProgress onPrev={() => setActiveStep(4)} />}
      </main>

      {/* Simple Academic Footer */}
      <footer className="bg-white border-t border-slate-200 py-4 text-center text-xs text-slate-500">
        SLIIT B.Sc. (Hons) in Information Technology Specialising in Data Science • Project J26-DS-345 • Component 1 Walkthrough UI
      </footer>
    </div>
  );
}

/* ============================================================================
   STEP 1: DATA COLLECTION & CLEANING FUNNEL (01_data_preparation)
   ============================================================================ */
function Step1DataFunnel({ overview, onNext }) {
  const f = overview.funnel;
  const districts = Object.entries(f.district_distribution || {}).sort((a, b) => b[1] - a[1]);

  return (
    <div className="space-y-6">
      {/* Intro Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-1 rounded">
            Stage 01 • Data Collection & Preprocessing
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-2">
            From 59,889 Raw Scraped Reviews to 39,399 Verified English Reviews
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            We scraped hotel reviews from three major platforms across Sri Lanka, removed cross-platform duplicates and non-English noise, anonymized personal user names, and standardized ratings into a unified 35-field dataset.
          </p>
        </div>
        <button
          onClick={onNext}
          className="self-start md:self-center whitespace-nowrap bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-sm"
        >
          Next: Step 2 (Trust & ASD) →
        </button>
      </div>

      {/* Visual Step-by-Step Cleaning Funnel */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-4">
          1. Data Scraping & Filtering Funnel (What We Scraped vs. What We Kept)
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-stretch">
          {/* Box 1: Raw Scraped */}
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-500 uppercase">Step 1A: Raw Scraped</span>
                <span className="text-xs bg-slate-200 text-slate-700 font-semibold px-2 py-0.5 rounded">100%</span>
              </div>
              <div className="text-3xl font-extrabold text-slate-900 mt-2">
                {f.raw_scraped_reviews.toLocaleString()}
              </div>
              <p className="text-xs text-slate-600 mt-1">
                Raw reviews scraped across <strong>{f.discovered_properties.toLocaleString()}</strong> properties
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-200 text-xs text-slate-500">
              Sources: Booking.com, Google Maps, TripAdvisor
            </div>
          </div>

          {/* Box 2: Deduplication */}
          <div className="bg-rose-50/60 border border-rose-200 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-rose-700 uppercase">Step 1B: Deduplication</span>
                <span className="text-xs bg-rose-100 text-rose-800 font-bold px-2 py-0.5 rounded">
                  -22.2% Removed
                </span>
              </div>
              <div className="text-3xl font-extrabold text-rose-600 mt-2">
                -{f.duplicates_removed.toLocaleString()}
              </div>
              <p className="text-xs text-slate-700 mt-1">
                Exact & near-duplicate reviews removed across platforms
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-rose-200 text-xs font-semibold text-slate-700">
              Remaining: {f.after_deduplication.toLocaleString()} reviews
            </div>
          </div>

          {/* Box 3: Language Filter */}
          <div className="bg-amber-50/60 border border-amber-200 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-amber-800 uppercase">Step 1C: Language Filter</span>
                <span className="text-xs bg-amber-100 text-amber-900 font-bold px-2 py-0.5 rounded">
                  -12.0% Removed
                </span>
              </div>
              <div className="text-3xl font-extrabold text-amber-600 mt-2">
                -{f.non_english_removed.toLocaleString()}
              </div>
              <p className="text-xs text-slate-700 mt-1">
                Non-English (Dutch, German, Russian, Italian) & empty rows removed
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-amber-200 text-xs font-semibold text-slate-700">
              65 poorly-reviewed properties dropped (1,090 → 1,025)
            </div>
          </div>

          {/* Box 4: Final Clean Dataset */}
          <div className="bg-emerald-50/70 border-2 border-emerald-500 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-emerald-800 uppercase">Step 1D: Final Clean Corpus</span>
                <span className="text-xs bg-emerald-600 text-white font-bold px-2 py-0.5 rounded">
                  65.8% Retained
                </span>
              </div>
              <div className="text-3xl font-extrabold text-emerald-700 mt-2">
                {f.final_verified_reviews.toLocaleString()}
              </div>
              <p className="text-xs text-slate-700 mt-1">
                Verified English reviews across <strong>{f.final_verified_hotels.toLocaleString()} hotels</strong>
              </p>
            </div>
            <div className="mt-4 pt-3 border-t border-emerald-200 text-xs font-bold text-emerald-800">
              ✓ All 25 Districts • Avg {f.average_word_count} words/review
            </div>
          </div>
        </div>
      </div>

      {/* Platform Split & 25 Districts Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Platform Breakdown */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              2. Multi-Source Platform Breakdown
            </h3>
            <p className="text-xs text-slate-500 mt-1">
              Distribution of the 39,399 clean English reviews by source OTA platform.
            </p>

            <div className="space-y-4 mt-5">
              {[
                { name: "Booking.com", count: f.platform_distribution["Booking.com"] || 32382, pct: "82.19%", color: "bg-blue-600", note: "Verified stays, split positive & negative review fields" },
                { name: "Google Maps", count: f.platform_distribution["Google Maps"] || 6592, pct: "16.73%", color: "bg-emerald-600", note: "Local Guide badges, open ratings (1–5 stars)" },
                { name: "TripAdvisor", count: f.platform_distribution["TripAdvisor"] || 425, pct: "1.08%", color: "bg-amber-500", note: "Long-form narrative resort & boutique hotel reviews" },
              ].map((p) => (
                <div key={p.name} className="bg-slate-50 p-3.5 rounded-lg border border-slate-200">
                  <div className="flex justify-between items-center text-sm font-bold text-slate-800">
                    <span>{p.name}</span>
                    <span>
                      {p.count.toLocaleString()} <span className="text-xs text-slate-500 font-normal">({p.pct})</span>
                    </span>
                  </div>
                  <div className="w-full bg-slate-200 h-2 rounded-full mt-2 overflow-hidden">
                    <div className={`${p.color} h-full`} style={{ width: p.pct }} />
                  </div>
                  <p className="text-xs text-slate-500 mt-1.5">{p.note}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 bg-blue-50 border border-blue-200 rounded-lg p-3 text-xs text-blue-900">
            <strong>Why 3 Platforms?</strong> Combining Booking.com (verified stays) with Google Maps & TripAdvisor (open unverified platforms) allows our Trust Engine to detect unverified rating manipulation.
          </div>
        </div>

        {/* Right: All 25 Districts Coverage */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                3. Island-Wide Coverage Across All 25 Sri Lankan Districts
              </h3>
              <p className="text-xs text-slate-500 mt-1">
                Stratified across coastal, hill-country, cultural triangle, and northern/eastern districts.
              </p>
            </div>
            <span className="text-xs font-bold bg-slate-100 text-slate-700 px-2.5 py-1 rounded border border-slate-200">
              25 / 25 Districts
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5 mt-4">
            {districts.map(([dist, cnt]) => (
              <div
                key={dist}
                className="bg-slate-50 border border-slate-200 rounded-lg p-2.5 flex flex-col justify-between"
              >
                <span className="text-xs font-semibold text-slate-800 truncate">{dist}</span>
                <span className="text-sm font-extrabold text-blue-600 mt-1">
                  {cnt.toLocaleString()} <span className="text-[10px] font-normal text-slate-500">revs</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* What We Did in Preprocessing: Before vs After Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-1">
          4. What We Did to Clean & Standardize the Data (Step-by-Step Operations)
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Concrete examples of how raw scraped HTML/JSON records were cleaned before entering the ML pipeline.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                <th className="p-3 font-bold">Preprocessing Step</th>
                <th className="p-3 font-bold">Problem in Raw Scraped Data</th>
                <th className="p-3 font-bold">Raw Scraped Input Example</th>
                <th className="p-3 font-bold">Cleaned Pipeline Output</th>
                <th className="p-3 font-bold">Impact</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {(overview.cleaning_examples || []).map((ex, i) => (
                <tr key={i} className="hover:bg-slate-50">
                  <td className="p-3 font-bold text-slate-900 whitespace-nowrap">{ex.step}</td>
                  <td className="p-3 text-slate-600">{ex.issue}</td>
                  <td className="p-3 font-mono text-[11px] text-rose-700 bg-rose-50/40">{ex.raw_input}</td>
                  <td className="p-3 font-mono text-[11px] text-emerald-800 bg-emerald-50/40">{ex.cleaned_output}</td>
                  <td className="p-3 font-bold text-slate-800 whitespace-nowrap">{ex.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   STEP 2: TRUST & FAKE REVIEW FILTERING (02_trust_classification)
   ============================================================================ */
function Step2TrustAndASD({ benchmarks, onPrev, onNext }) {
  const [reviewText, setReviewText] = useState(
    "We stayed in room 204 for 3 nights. The egg hoppers and pol sambol at breakfast were delicious, and the hot water shower had great pressure. Only 10 mins walk from Galle Fort."
  );
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  const presets = [
    {
      label: "Authentic Sri Lankan Review (High ASD)",
      text: "We stayed in room 204 for 3 nights. The egg hoppers and pol sambol at breakfast were delicious, and the hot water shower had great pressure. Only 10 mins walk from Galle Fort.",
    },
    {
      label: "AI-Generated Promotional Spam (Low ASD, High Fluff)",
      text: "An absolute heavenly paradise on earth! Breathtaking, immaculate, and beyond extraordinary. Highly recommended to everyone, a truly magical dream vacation!",
    },
    {
      label: "Mixed Review with Negation ('not working')",
      text: "The location near Mirissa beach was convenient, but the air conditioning was not working and the wifi connection was very slow.",
    },
  ];

  const runLiveTest = (txt) => {
    const targetText = txt !== undefined ? txt : reviewText;
    setLoading(true);
    fetch(`${API_BASE}/api/v1/analyze-review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_text: targetText, hotel_baseline_rating: 8.0 }),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setAnalysis(data);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    runLiveTest(reviewText);
  }, []);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-1 rounded">
            Stage 02 • Aspect Specificity Density (ASD) & ML Trust Classification
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-2">
            Detecting AI-Generated & Promotional Spam Reviews
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            Instead of relying only on word count, we formulated <strong>Aspect Specificity Density (ASD)</strong>—measuring concrete Sri Lankan hospitality entities (e.g., <em>egg hoppers, pol sambol, split AC, room 204</em>) vs. generic promotional superlatives (<em>heavenly, paradise, breathtaking</em>)—and trained <strong>LightGBM & XGBoost</strong> classifiers.
          </p>
        </div>
        <div className="flex space-x-2 self-start md:self-center">
          <button
            onClick={onPrev}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold px-3 py-2.5 rounded-lg"
          >
            ← Step 1
          </button>
          <button
            onClick={onNext}
            className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-sm whitespace-nowrap"
          >
            Next: Step 3 (6-Aspect ABSA) →
          </button>
        </div>
      </div>

      {/* Live Interactive ASD & Trust Inspector */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              1. Interactive Live Review Trust & ASD Inspector
            </h3>
            <p className="text-xs text-slate-500">
              Click a preset or type any review to see how our engine extracts concrete entities, penalizes fluff, and computes Trust Score.
            </p>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {presets.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setReviewText(p.text);
                  runLiveTest(p.text);
                }}
                className="text-xs bg-slate-100 hover:bg-blue-50 hover:text-blue-700 hover:border-blue-200 text-slate-700 font-medium px-2.5 py-1.5 rounded-lg border border-slate-200 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-3">
          <textarea
            rows={2}
            value={reviewText}
            onChange={(e) => setReviewText(e.target.value)}
            className="flex-1 bg-slate-50 border border-slate-300 rounded-lg p-3 text-sm text-slate-900 focus:outline-none focus:border-blue-600 focus:bg-white"
            placeholder="Enter a hotel review sentence to test..."
          />
          <button
            onClick={() => runLiveTest(reviewText)}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs px-5 py-3 rounded-lg shadow-sm whitespace-nowrap self-end sm:self-stretch"
          >
            {loading ? "Analyzing..." : "Run Trust & ASD Filter"}
          </button>
        </div>

        {analysis && (
          <div className="grid grid-cols-1 md:grid-cols-12 gap-4 pt-2">
            {/* Score Summary Card */}
            <div
              className={`md:col-span-4 rounded-xl p-5 border-2 flex flex-col justify-between ${
                analysis.is_suspected_ai_spam
                  ? "bg-rose-50/70 border-rose-400"
                  : "bg-emerald-50/70 border-emerald-500"
              }`}
            >
              <div>
                <span className="text-xs font-bold uppercase text-slate-600">Authenticity Verdict</span>
                <div className="mt-2 flex items-baseline space-x-2">
                  <span
                    className={`text-3xl font-extrabold ${
                      analysis.is_suspected_ai_spam ? "text-rose-700" : "text-emerald-700"
                    }`}
                  >
                    {analysis.trust_score.toFixed(1)}
                  </span>
                  <span className="text-xs font-semibold text-slate-500">/ 10.0 Trust Score</span>
                </div>
                <div className="mt-2">
                  <span
                    className={`inline-block text-xs font-bold px-2.5 py-1 rounded ${
                      analysis.is_suspected_ai_spam
                        ? "bg-rose-600 text-white"
                        : "bg-emerald-600 text-white"
                    }`}
                  >
                    {analysis.authenticity_label}
                  </span>
                </div>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-200/80 space-y-1 text-xs text-slate-700">
                <div className="flex justify-between">
                  <span>ASD Score (0.02–1.00):</span>
                  <strong className="font-mono">{analysis.asd_score.toFixed(4)}</strong>
                </div>
                <div className="flex justify-between">
                  <span>Fluff-to-Entity Ratio:</span>
                  <strong className="font-mono">{analysis.fluff_to_entity_ratio}</strong>
                </div>
                <div className="flex justify-between">
                  <span>Word Count:</span>
                  <strong>{analysis.word_count} words</strong>
                </div>
              </div>
            </div>

            {/* Extracted Tokens */}
            <div className="md:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                <div className="text-xs font-bold text-emerald-800 uppercase">
                  ✓ Concrete Hospitality Entities ({analysis.concrete_entities_count}) & Measurements ({(analysis.detected_measurements || []).length})
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Verifiable dishes, room fixtures, landmarks, and numbers that prove a real stay:
                </p>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {[...(analysis.detected_entities || []), ...(analysis.detected_measurements || [])].length > 0 ? (
                    [...(analysis.detected_entities || []), ...(analysis.detected_measurements || [])].map((e, i) => (
                      <span
                        key={i}
                        className="bg-emerald-100 text-emerald-900 border border-emerald-300 text-xs font-semibold px-2.5 py-0.5 rounded-md"
                      >
                        {e}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-400 italic">Zero concrete entities found (vague text)</span>
                  )}
                </div>
              </div>

              <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
                <div className="text-xs font-bold text-rose-800 uppercase">
                  ✕ Generic Promotional Fluff ({analysis.generic_fluff_count})
                </div>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Hyperbolic superlatives over-used by zero-shot LLM / promotional bots:
                </p>
                <div className="flex flex-wrap gap-1.5 mt-3">
                  {(analysis.detected_fluff || []).length > 0 ? (
                    analysis.detected_fluff.map((fl, i) => (
                      <span
                        key={i}
                        className="bg-rose-100 text-rose-900 border border-rose-300 text-xs font-semibold px-2.5 py-0.5 rounded-md"
                      >
                        {fl}
                      </span>
                    ))
                  ) : (
                    <span className="text-xs text-slate-400 italic">Zero promotional fluff detected</span>
                  )}
                </div>
              </div>

              {/* Extracted Aspects Preview */}
              <div className="sm:col-span-2 bg-slate-50 border border-slate-200 rounded-xl p-3.5 flex flex-wrap items-center justify-between gap-2">
                <span className="text-xs font-bold text-slate-700 uppercase">
                  Detected Aspect Sentiments:
                </span>
                <div className="flex flex-wrap gap-2">
                  {(analysis.detected_aspects || []).map((a, i) => (
                    <span
                      key={i}
                      className={`text-xs font-bold px-2.5 py-1 rounded border ${
                        a.sentiment === "Positive"
                          ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                          : a.sentiment === "Negative"
                          ? "bg-rose-50 text-rose-700 border-rose-200"
                          : "bg-slate-100 text-slate-700 border-slate-300"
                      }`}
                    >
                      {a.aspect.toUpperCase()}: {a.sentiment} ({a.score_5}/5.0)
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Trained Classifier Results & Feature Ablation */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
            2. Trained Trust Classifiers (5-Fold Cross-Validation)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Trained on our 400-sentence benchmark (360 authentic Sri Lankan reviews + 40 unique AI-spam reviews across all 6 aspects).
          </p>

          <div className="grid grid-cols-2 gap-3 mt-4">
            <div className="bg-blue-50/60 border border-blue-200 rounded-xl p-4">
              <div className="text-xs font-bold text-blue-900">LightGBM (Primary)</div>
              <div className="text-2xl font-extrabold text-blue-700 mt-1">92.28%</div>
              <div className="text-[11px] text-slate-600">Macro-F1 Score</div>
              <div className="mt-2 pt-2 border-t border-blue-200 text-xs space-y-0.5 text-slate-700">
                <div>Accuracy: <strong>98.50%</strong></div>
                <div>ROC-AUC: <strong>100.00%</strong></div>
                <div>Precision: <strong>93.33%</strong></div>
              </div>
            </div>

            <div className="bg-slate-50 border border-slate-200 rounded-xl p-4">
              <div className="text-xs font-bold text-slate-800">XGBoost (Comparator)</div>
              <div className="text-2xl font-extrabold text-slate-900 mt-1">91.70%</div>
              <div className="text-[11px] text-slate-600">Macro-F1 Score</div>
              <div className="mt-2 pt-2 border-t border-slate-200 text-xs space-y-0.5 text-slate-700">
                <div>Accuracy: <strong>98.50%</strong></div>
                <div>ROC-AUC: <strong>99.93%</strong></div>
                <div>Precision: <strong>95.56%</strong></div>
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
            3. Feature Ablation Study (Proving Our ASD Feature Works)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Comparing a naive word-count baseline against our ASD content features and the full 8-feature hybrid model.
          </p>

          <div className="overflow-x-auto mt-4">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                  <th className="p-2.5 font-bold">Model Configuration</th>
                  <th className="p-2.5 font-bold">Features</th>
                  <th className="p-2.5 font-bold">ROC-AUC</th>
                  <th className="p-2.5 font-bold">Accuracy</th>
                  <th className="p-2.5 font-bold">Macro-F1</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {(benchmarks.trust_ablation || []).map((row, i) => (
                  <tr key={i} className={i === 2 ? "bg-blue-50/50 font-semibold" : ""}>
                    <td className="p-2.5 text-slate-900">{row.config}</td>
                    <td className="p-2.5 text-slate-600">{row.features}</td>
                    <td className="p-2.5 font-mono">{row.roc_auc}%</td>
                    <td className="p-2.5 font-mono">{row.accuracy}%</td>
                    <td className="p-2.5 font-mono font-bold text-blue-700">{row.macro_f1}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-emerald-700 font-semibold mt-3">
            ✓ Key Finding: Adding Aspect Specificity Density (ASD) & fluff features lifts Macro-F1 by +57.54 percentage points over word count alone (34.16% → 91.70%).
          </p>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   STEP 3: 6-ASPECT SENTIMENT ANALYSIS (03_aspect_sentiment)
   ============================================================================ */
function Step3AspectSentiment({ benchmarks, onPrev, onNext }) {
  const stdList = benchmarks.absa_comparison?.standard_400 || [];
  const hardList = benchmarks.absa_comparison?.hard_challenges || [];
  const samples = benchmarks.absa_comparison?.sample_challenges || [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-1 rounded">
            Stage 03 • 6-Aspect Sentiment Analysis (ABSA)
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-2">
            Extracting 6 Hotel Aspects & Handling Sarcasm / Sri Lankan Vernacular
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            A single star rating hides whether a hotel has great food but broken WiFi. We split every review into clauses and score <strong>6 hospitality aspects</strong> (<code>food</code>, <code>staff</code>, <code>room</code>, <code>cleanliness</code>, <code>location</code>, <code>wifi</code>) using a fast Lexical Tier-1 engine escalated to <strong>DeBERTa-v3-small</strong> for complex clauses.
          </p>
        </div>
        <div className="flex space-x-2 self-start md:self-center">
          <button
            onClick={onPrev}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold px-3 py-2.5 rounded-lg"
          >
            ← Step 2
          </button>
          <button
            onClick={onNext}
            className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-sm whitespace-nowrap"
          >
            Next: Step 4 (Hotel Rankings) →
          </button>
        </div>
      </div>

      {/* Inter-Annotator Agreement & Benchmark Setup */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Gold-Standard Dataset</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">450 Sentences</div>
          <p className="text-xs text-slate-600 mt-1">
            400 stratified hotel sentences + 50 adversarial linguistic challenges (Sarcasm, Double Negation, Vernacular, Implicit).
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Aspect Annotation Agreement</span>
          <div className="text-2xl font-extrabold text-blue-600 mt-1">κ = 0.9159</div>
          <p className="text-xs text-slate-600 mt-1">
            Cohen’s Kappa ("Almost Perfect Agreement", 93.0% accuracy) across dual annotators.
          </p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Sentiment Polarity Agreement</span>
          <div className="text-2xl font-extrabold text-emerald-600 mt-1">κ = 0.6812</div>
          <p className="text-xs text-slate-600 mt-1">
            Cohen’s Kappa ("Substantial Agreement", 83.0% accuracy) on 3-class polarity.
          </p>
        </div>
      </div>

      {/* Comparison Tables: Standard 400 vs Hard 50 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Table A: Standard 400 */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
            1. Standard Hospitality Benchmark (N = 400 Sentences)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            On direct reviews, our fast Tier-1 Lexical Engine reaches 96.25% accuracy in 1.2ms, while DeBERTa-v3 reaches 100%.
          </p>
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                  <th className="p-2.5 font-bold">Model</th>
                  <th className="p-2.5 font-bold">Accuracy</th>
                  <th className="p-2.5 font-bold">Macro-F1</th>
                  <th className="p-2.5 font-bold">Latency / 100</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {stdList.map((m, i) => (
                  <tr key={i} className={i === 3 ? "bg-blue-50/50 font-semibold" : ""}>
                    <td className="p-2.5 text-slate-900">{m.model}</td>
                    <td className="p-2.5 font-mono">{m.accuracy}%</td>
                    <td className="p-2.5 font-mono font-bold text-blue-700">{m.macro_f1}%</td>
                    <td className="p-2.5 font-mono text-slate-500">{m.latency_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Table B: Hard 50 Challenges */}
        <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
            2. 50 Hard Linguistic Challenges (Why DeBERTa-v3 is Essential)
          </h3>
          <p className="text-xs text-slate-500 mt-1">
            Rule-based lexicons collapse to 14%–32% when guests use sarcasm or implicit complaints. DeBERTa-v3 achieves 86.0%.
          </p>
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                  <th className="p-2.5 font-bold">Model</th>
                  <th className="p-2.5 font-bold">Hard Acc</th>
                  <th className="p-2.5 font-bold">Sarcasm</th>
                  <th className="p-2.5 font-bold">Negation</th>
                  <th className="p-2.5 font-bold">Vernacular</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200">
                {hardList.map((m, i) => (
                  <tr key={i} className={i === 3 ? "bg-emerald-50/60 font-semibold" : ""}>
                    <td className="p-2.5 text-slate-900">{m.model}</td>
                    <td className="p-2.5 font-mono font-bold text-emerald-700">{m.accuracy}%</td>
                    <td className="p-2.5 font-mono">{m.sarcasm}%</td>
                    <td className="p-2.5 font-mono">{m.negation}%</td>
                    <td className="p-2.5 font-mono">{m.vernacular || 70.0}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Real Adversarial Examples Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider mb-1">
          3. Real Sentences from Our 50 Hard Challenges Dataset
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Examples showing why VADER / keyword dictionaries fail and why our Cascaded Hybrid Escalation routes these clauses to DeBERTa-v3.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                <th className="p-2.5 font-bold">Phenomenon</th>
                <th className="p-2.5 font-bold">Aspect</th>
                <th className="p-2.5 font-bold">Review Sentence</th>
                <th className="p-2.5 font-bold">True Sentiment</th>
                <th className="p-2.5 font-bold">Why Lexicon / VADER Fails</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {samples.slice(0, 8).map((s, idx) => (
                <tr key={idx} className="hover:bg-slate-50">
                  <td className="p-2.5 font-semibold text-blue-700 whitespace-nowrap">{s.phenomenon_type}</td>
                  <td className="p-2.5 font-bold uppercase text-slate-700">{s.target_aspect}</td>
                  <td className="p-2.5 text-slate-900 italic">"{s.review_sentence}"</td>
                  <td className="p-2.5">
                    <span
                      className={`px-2 py-0.5 rounded font-bold ${
                        s.ground_truth_sentiment === "Positive"
                          ? "bg-emerald-100 text-emerald-800"
                          : s.ground_truth_sentiment === "Negative"
                          ? "bg-rose-100 text-rose-800"
                          : "bg-slate-200 text-slate-800"
                      }`}
                    >
                      {s.ground_truth_sentiment}
                    </span>
                  </td>
                  <td className="p-2.5 text-slate-600">{s.linguistic_failure_mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ============================================================================
   STEP 4: HOTEL RANKINGS (BEFORE vs AFTER) & ASPECT PROFILES (1,025 HOTELS)
   ============================================================================ */
function Step4RankingsAndProfiles({ benchmarks, onPrev, onNext }) {
  const [districtFilter, setDistrictFilter] = useState("All");
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState("rank_after");
  const [rankingData, setRankingData] = useState({
    total_matching: 0,
    total_national_hotels: 1025,
    selected_district: "All",
    districts: [],
    hotels: [],
  });
  const [loadingRanks, setLoadingRanks] = useState(false);
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  // Selected hotel for detailed 6-aspect & contract inspection
  const [selectedHotelId, setSelectedHotelId] = useState("HTL_KEG_BOGALAVI_C759");
  const [selectedProfile, setSelectedProfile] = useState(null);
  const [selectedAlerts, setSelectedAlerts] = useState(null);

  // Load rankings whenever district or sort_by changes
  useEffect(() => {
    setLoadingRanks(true);
    const params = new URLSearchParams();
    if (districtFilter && districtFilter !== "All") params.set("district", districtFilter);
    if (sortBy) params.set("sort_by", sortBy);

    fetch(`${API_BASE}/api/v1/rankings?${params.toString()}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data && data.hotels) {
          setRankingData(data);
          setPage(1);
          if (data.hotels.length > 0 && !data.hotels.some((h) => h.hotel_id === selectedHotelId)) {
            setSelectedHotelId(data.hotels[0].hotel_id);
          }
        }
      })
      .catch(() => {})
      .finally(() => setLoadingRanks(false));
  }, [districtFilter, sortBy]);

  // Filter client-side by search query for instant responsiveness across all 1,025 hotels
  const filteredHotels = useMemo(() => {
    if (!searchQuery.trim()) return rankingData.hotels;
    const q = searchQuery.toLowerCase();
    return rankingData.hotels.filter(
      (h) =>
        h.hotel_name.toLowerCase().includes(q) ||
        h.city.toLowerCase().includes(q) ||
        h.district.toLowerCase().includes(q) ||
        h.hotel_id.toLowerCase().includes(q)
    );
  }, [rankingData.hotels, searchQuery]);

  const totalPages = Math.max(1, Math.ceil(filteredHotels.length / PAGE_SIZE));
  const visibleHotels = filteredHotels.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  // Fetch detailed aspect profile & alerts when a hotel row is clicked
  useEffect(() => {
    if (!selectedHotelId) return;
    fetch(`${API_BASE}/hotels/${selectedHotelId}/aspect-scores`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setSelectedProfile(data);
      })
      .catch(() => {});

    fetch(`${API_BASE}/hotels/${selectedHotelId}/alerts`)
      .then((r) => (r.ok ? r.json() : null))
      .then((data) => {
        if (data) setSelectedAlerts(data);
      })
      .catch(() => {});
  }, [selectedHotelId]);

  const selectedRankRow = useMemo(
    () => rankingData.hotels.find((h) => h.hotel_id === selectedHotelId) || visibleHotels[0],
    [rankingData.hotels, selectedHotelId, visibleHotels]
  );

  const isDistrictScoped = districtFilter && districtFilter !== "All";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-5 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-1 rounded">
            Stage 04 • Time-Decay Aggregation, Hotel Rankings & Contract Output
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-2">
            Before vs. After Hotel Rankings (All 1,025 Hotels & 25 Districts)
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            By weighting reviews with <strong>Trust Score $T(r)$</strong> and <strong>365-Day Exponential Time-Decay $w(t) = e^&#123;-\lambda \Delta t&#125;$</strong>, hotels that recently improved rise in rank while properties coasting on 4-year-old reviews or promotional spam drop.
          </p>
        </div>
        <div className="flex space-x-2 self-start md:self-center">
          <button
            onClick={onPrev}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold px-3 py-2.5 rounded-lg"
          >
            ← Step 3
          </button>
          <button
            onClick={onNext}
            className="bg-blue-600 hover:bg-blue-700 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-sm whitespace-nowrap"
          >
            Next: Step 5 (80% Viva Status) →
          </button>
        </div>
      </div>

      {/* Ablation Summary KPI Strip */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Total Ranked Hotels</span>
          <div className="text-2xl font-extrabold text-slate-900 mt-1">1,025 Hotels</div>
          <p className="text-xs text-slate-500 mt-0.5">Across all 25 districts (39,399 reviews)</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Mean Rank Shift (MARS)</span>
          <div className="text-2xl font-extrabold text-blue-600 mt-1">61.26 Spots</div>
          <p className="text-xs text-slate-500 mt-0.5">Average position change (Before vs After)</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Shifted ≥ 10 Positions</span>
          <div className="text-2xl font-extrabold text-emerald-600 mt-1">67.22%</div>
          <p className="text-xs text-slate-500 mt-0.5">689 of 1,025 hotels re-ordered by ≥ 10 spots</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-sm">
          <span className="text-xs font-bold text-slate-500 uppercase">Spearman Rank (ρ)</span>
          <div className="text-2xl font-extrabold text-amber-600 mt-1">0.9449</div>
          <p className="text-xs text-slate-500 mt-0.5">Kendall’s τ = 0.8274 (p &lt; 0.001)</p>
        </div>
      </div>

      {/* Filter Bar + Interactive Before/After Leaderboard Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3 border-b border-slate-200 pb-4">
          <div>
            <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
              1. Interactive Hotel Ranking Table: Before Rank (Raw OTA) vs. After Rank (Trust + Time-Decay)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Filter across all 1,025 hotels nationally or select any district to see District-specific Before & After rankings. Click any hotel row to inspect its 6 aspects below.
            </p>
          </div>

          {/* Controls */}
          <div className="flex flex-wrap items-center gap-2">
            {/* District Selector */}
            <select
              value={districtFilter}
              onChange={(e) => setDistrictFilter(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-900 text-xs font-semibold rounded-lg px-3 py-2 focus:outline-none focus:border-blue-600"
            >
              <option value="All">All 25 Districts (Full 1,025 Hotels)</option>
              {(rankingData.districts || []).map((d) => (
                <option key={d.district} value={d.district}>
                  {d.district} District ({d.hotel_count} hotels)
                </option>
              ))}
            </select>

            {/* Sort By Selector */}
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-slate-50 border border-slate-300 text-slate-900 text-xs font-semibold rounded-lg px-3 py-2 focus:outline-none focus:border-blue-600"
            >
              <option value="rank_after">Sort: After Rank (#1 → #{isDistrictScoped ? filteredHotels.length : 1025})</option>
              <option value="rank_before">Sort: Before Rank (Raw OTA #1 →)</option>
              <option value="biggest_gain">Sort: Biggest Rank Gainers (Recovering SMEs ↑)</option>
              <option value="biggest_drop">Sort: Biggest Rank Drops (Spam / Stale ↓)</option>
              <option value="reviews">Sort: Most Reviews</option>
            </select>

            {/* Search Input */}
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setPage(1);
              }}
              placeholder="Search hotel name or city..."
              className="bg-slate-50 border border-slate-300 text-slate-900 text-xs rounded-lg px-3 py-2 w-48 focus:outline-none focus:border-blue-600"
            />
          </div>
        </div>

        {/* Ranking Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100 text-slate-700 border-b border-slate-200 uppercase">
                <th className="p-2.5 font-bold">Hotel Name</th>
                <th className="p-2.5 font-bold">District / City</th>
                <th className="p-2.5 font-bold text-center">Reviews</th>
                <th className="p-2.5 font-bold text-right">Raw Score (Before)</th>
                <th className="p-2.5 font-bold text-right">Proposed Score (After)</th>
                <th className="p-2.5 font-bold text-center bg-slate-200/60">
                  {isDistrictScoped ? `${districtFilter} Rank Before` : "National Rank Before"}
                </th>
                <th className="p-2.5 font-bold text-center bg-blue-50 text-blue-900">
                  {isDistrictScoped ? `${districtFilter} Rank After` : "National Rank After"}
                </th>
                <th className="p-2.5 font-bold text-center">Rank Shift (Before → After)</th>
                <th className="p-2.5 font-bold text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200">
              {loadingRanks ? (
                <tr>
                  <td colSpan={9} className="p-6 text-center text-slate-500">
                    Loading 1,025 hotel rankings...
                  </td>
                </tr>
              ) : visibleHotels.length === 0 ? (
                <tr>
                  <td colSpan={9} className="p-6 text-center text-slate-500">
                    No hotels match your filter. Make sure the FastAPI backend is running on port 8001.
                  </td>
                </tr>
              ) : (
                visibleHotels.map((h) => {
                  const rBefore = isDistrictScoped ? h.district_rank_before : h.rank_before;
                  const rAfter = isDistrictScoped ? h.district_rank_after : h.rank_after;
                  const rShift = isDistrictScoped ? h.district_rank_shift : h.rank_shift;
                  const isSelected = h.hotel_id === selectedHotelId;

                  return (
                    <tr
                      key={h.hotel_id}
                      onClick={() => setSelectedHotelId(h.hotel_id)}
                      className={`cursor-pointer transition-colors ${
                        isSelected ? "bg-blue-50/80 font-semibold" : "hover:bg-slate-50"
                      }`}
                    >
                      <td className="p-2.5 text-slate-900 font-semibold">{h.hotel_name}</td>
                      <td className="p-2.5 text-slate-600">
                        {h.district} <span className="text-slate-400">({h.city})</span>
                      </td>
                      <td className="p-2.5 text-center font-mono">{h.review_count}</td>
                      <td className="p-2.5 text-right font-mono text-slate-600">
                        {h.score_before.toFixed(2)} / 5.0
                      </td>
                      <td className="p-2.5 text-right font-mono font-bold text-blue-700">
                        {h.score_after.toFixed(2)} / 5.0
                      </td>
                      <td className="p-2.5 text-center font-mono bg-slate-100/70 text-slate-700">
                        #{rBefore}
                      </td>
                      <td className="p-2.5 text-center font-mono font-extrabold bg-blue-50/70 text-blue-800">
                        #{rAfter}
                      </td>
                      <td className="p-2.5 text-center">
                        {rShift > 0 ? (
                          <span className="inline-block bg-emerald-100 text-emerald-800 border border-emerald-300 font-bold px-2 py-0.5 rounded font-mono">
                            ↑ +{rShift} spots
                          </span>
                        ) : rShift < 0 ? (
                          <span className="inline-block bg-rose-100 text-rose-800 border border-rose-300 font-bold px-2 py-0.5 rounded font-mono">
                            ↓ {rShift} spots
                          </span>
                        ) : (
                          <span className="inline-block bg-slate-100 text-slate-600 px-2 py-0.5 rounded font-mono">
                            0 (Unchanged)
                          </span>
                        )}
                      </td>
                      <td className="p-2.5 text-center">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedHotelId(h.hotel_id);
                          }}
                          className={`text-[11px] font-bold px-2.5 py-1 rounded ${
                            isSelected
                              ? "bg-blue-600 text-white"
                              : "bg-slate-100 hover:bg-blue-100 text-slate-700"
                          }`}
                        >
                          {isSelected ? "Viewing" : "Inspect"}
                        </button>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 pt-2 text-xs text-slate-600">
          <div>
            Showing <strong>{(page - 1) * PAGE_SIZE + (visibleHotels.length ? 1 : 0)}</strong> –{" "}
            <strong>{(page - 1) * PAGE_SIZE + visibleHotels.length}</strong> of{" "}
            <strong>{filteredHotels.length}</strong> hotels{" "}
            {isDistrictScoped ? `in ${districtFilter} District` : "(Full National Dataset)"}
          </div>
          <div className="flex items-center space-x-2">
            <button
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              className="px-3 py-1.5 rounded border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-40 font-semibold"
            >
              Previous
            </button>
            <span className="font-semibold">
              Page {page} of {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              className="px-3 py-1.5 rounded border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-40 font-semibold"
            >
              Next
            </button>
          </div>
        </div>
      </div>

      {/* Selected Hotel Detailed 6-Aspect Profile & Contract Output */}
      {selectedRankRow && (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left: 6-Aspect Breakdown */}
          <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 pb-3 gap-2">
              <div>
                <span className="text-[11px] font-bold uppercase text-blue-600">
                  Selected Hotel 6-Aspect Profile
                </span>
                <h3 className="text-base font-bold text-slate-900">
                  {selectedRankRow.hotel_name} ({selectedRankRow.district})
                </h3>
              </div>
              <div className="flex items-center space-x-2 text-xs">
                <span className="bg-slate-100 text-slate-700 font-mono px-2.5 py-1 rounded border border-slate-200">
                  Before: #{selectedRankRow.rank_before} ({selectedRankRow.score_before.toFixed(2)})
                </span>
                <span className="bg-blue-600 text-white font-mono font-bold px-2.5 py-1 rounded">
                  After: #{selectedRankRow.rank_after} ({selectedRankRow.score_after.toFixed(2)})
                </span>
              </div>
            </div>

            {/* 6 Aspect Bars */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              {[
                { key: "food", label: "Food & Dining", icon: "🍳" },
                { key: "staff", label: "Staff & Hospitality", icon: "🤝" },
                { key: "room", label: "Room Comfort & AC", icon: "🛏️" },
                { key: "cleanliness", label: "Cleanliness & Hygiene", icon: "✨" },
                { key: "location", label: "Location & Access", icon: "📍" },
                { key: "wifi", label: "WiFi & Connectivity", icon: "📶" },
              ].map((asp) => {
                const scoresObj =
                  selectedProfile?.aspect_scores || selectedRankRow?.aspect_scores || {};
                const val = Number(scoresObj[asp.key] ?? 4.0);
                const barColor =
                  val >= 4.2 ? "bg-emerald-600" : val >= 3.2 ? "bg-blue-600" : "bg-amber-500";
                return (
                  <div
                    key={asp.key}
                    className="bg-slate-50 border border-slate-200 rounded-lg p-3"
                  >
                    <div className="flex justify-between items-center text-xs font-bold text-slate-800">
                      <span>
                        {asp.icon} {asp.label}
                      </span>
                      <span className="font-mono text-sm">{val.toFixed(2)} / 5.0</span>
                    </div>
                    <div className="w-full bg-slate-200 h-2 rounded-full mt-2 overflow-hidden">
                      <div
                        className={`${barColor} h-full transition-all duration-300`}
                        style={{ width: `${Math.min(100, (val / 5.0) * 100)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Verified Facts for Component 4 RAG */}
            <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-bold text-slate-800 uppercase">
                Extracted Verified Facts (Sent to Component 4 RAG Chatbot)
              </div>
              <ul className="mt-2 space-y-1.5 text-xs text-slate-700 list-disc list-inside">
                {(selectedProfile?.verified_aspect_facts || []).length > 0 ? (
                  selectedProfile.verified_aspect_facts.map((fact, idx) => (
                    <li key={idx}>{fact}</li>
                  ))
                ) : (
                  <li>Verified aspect scores aggregated from {selectedRankRow.review_count} guest reviews.</li>
                )}
              </ul>
            </div>
          </div>

          {/* Right: Weak-Aspect Alerts & Shared JSON Contract */}
          <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col justify-between space-y-4">
            <div>
              <h3 className="text-sm font-bold text-slate-800 uppercase tracking-wider">
                Operational Alerts & Contract Payload
              </h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Rolling 60-day monitoring for SME hotel owners + JSON contract served to Component 2 & Component 4.
              </p>

              {/* Alerts List */}
              <div className="mt-3 space-y-2">
                {selectedAlerts?.alerts && selectedAlerts.alerts.length > 0 ? (
                  selectedAlerts.alerts.slice(0, 2).map((alt, i) => (
                    <div
                      key={i}
                      className={`p-3 rounded-lg border text-xs ${
                        alt.severity === "CRITICAL"
                          ? "bg-rose-50 border-rose-200 text-rose-900"
                          : alt.severity === "WARNING"
                          ? "bg-amber-50 border-amber-200 text-amber-900"
                          : "bg-emerald-50 border-emerald-200 text-emerald-900"
                      }`}
                    >
                      <div className="font-bold">
                        [{alt.severity}] {alt.alert_type} (60d Score: {alt.recent_score_60d})
                      </div>
                      <p className="mt-1 text-[11px]">{alt.actionable_recommendation}</p>
                    </div>
                  ))
                ) : (
                  <div className="p-3 rounded-lg border bg-emerald-50 border-emerald-200 text-xs text-emerald-900 font-medium">
                    ✓ No critical aspect degradation detected in the rolling 60-day window.
                  </div>
                )}
              </div>
            </div>

            {/* JSON Contract Box */}
            <div>
              <div className="text-xs font-bold text-slate-700 mb-1.5 flex items-center justify-between">
                <span>GET /hotels/{selectedRankRow.hotel_id}/aspect-scores</span>
                <span className="text-[10px] font-mono bg-slate-100 px-2 py-0.5 rounded border border-slate-200">
                  contracts/c1_aspect_scores.json
                </span>
              </div>
              <pre className="bg-slate-900 text-slate-100 text-[11px] p-3.5 rounded-lg overflow-x-auto font-mono max-h-52">
                {JSON.stringify(
                  selectedProfile || {
                    hotel_id: selectedRankRow.hotel_id,
                    aspect_scores: selectedRankRow.aspect_scores,
                    overall_trust_score: selectedRankRow.avg_trust_score_10,
                    time_decay_rating: selectedRankRow.score_after,
                  },
                  null,
                  2
                )}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ============================================================================
   STEP 5: 80% VIVA PROGRESS MILESTONE & REMAINING 20% ROADMAP
   ============================================================================ */
function Step5VivaProgress({ onPrev }) {
  const completedItems = [
    {
      stage: "Stage 01: Multi-Source Scraping & Cleaning Pipeline",
      pct: "100%",
      detail: "Scraped 59,889 raw reviews across 1,090 properties; removed 13,298 duplicates and 7,192 non-English rows; SHA-256 anonymized 39,399 verified English reviews across 1,025 hotels in all 25 districts.",
    },
    {
      stage: "Stage 02: Aspect Specificity Density (ASD) & Trust Classifiers",
      pct: "90%",
      detail: "Formulated ASD entity-to-fluff density metric; trained LightGBM (92.28% Macro-F1, 98.5% Acc) and XGBoost (91.70% Macro-F1) classifiers; completed 5-fold feature ablation study.",
    },
    {
      stage: "Stage 03: 6-Aspect Sentiment Engine (Lexical + DeBERTa-v3)",
      pct: "85%",
      detail: "Built 6-aspect clause splitter and benchmarked VADER, Lexical, RoBERTa-base, and DeBERTa-v3-small across 400 standard sentences (κ=0.9159) and 50 hard linguistic challenges (86.0% accuracy).",
    },
    {
      stage: "Stage 04: Exponential Time-Decay & Contract Microservice",
      pct: "80%",
      detail: "Implemented 365-day half-life weighting w(t), Before/After ranking ablation across all 1,025 hotels (MARS = 61.26 spots), Weak-Aspect Alert rules, and FastAPI Port 8001 contract endpoints.",
    },
  ];

  const remainingItems = [
    {
      title: "1. Live End-to-End Integration with Group Components (C2, C3, C4)",
      desc: "Connect Component 1's live FastAPI aspect profiles and verified facts directly into Component 2 (Hotel Recommender), Component 3 (Itinerary Planner), and Component 4 (Conversational RAG Agent).",
      target: "Final Defense (Remaining 20%)",
    },
    {
      title: "2. SME Hotel Owner Field Validation of Weak-Aspect Alerts",
      desc: "Conduct human usability evaluation with Sri Lankan SME hotel managers on the rolling 60-day CRITICAL/WARNING operational alerts and calibrate threshold sensitivity.",
      target: "Final Defense (Remaining 20%)",
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <span className="text-xs font-bold uppercase tracking-wider text-amber-800 bg-amber-100 px-2.5 py-1 rounded">
            Progress Presentation Milestone • 80% Viva Status
          </span>
          <h2 className="text-xl font-bold text-slate-900 mt-2">
            Component 1 Progress Summary: 80% Completed vs. 20% Remaining for Final Viva
          </h2>
          <p className="text-sm text-slate-600 mt-1">
            Clear breakdown of completed research deliverables for the current 80% Progress Viva and the remaining 20% work scheduled for the Final Examination.
          </p>
        </div>
        <button
          onClick={onPrev}
          className="self-start md:self-center bg-slate-100 hover:bg-slate-200 text-slate-700 text-xs font-semibold px-4 py-2.5 rounded-lg whitespace-nowrap"
        >
          ← Back to Step 4 (Rankings)
        </button>
      </div>

      {/* Overall Progress Bar Card */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-bold text-slate-900">
            Overall Component 1 Completion Status (Current Viva Target: 80%)
          </span>
          <span className="text-sm font-extrabold text-blue-600">80% Completed / 20% In Progress</span>
        </div>
        <div className="w-full h-4 bg-slate-200 rounded-full overflow-hidden flex">
          <div
            className="bg-blue-600 h-full flex items-center justify-center text-[10px] font-bold text-white"
            style={{ width: "80%" }}
          >
            80% Completed (Data, ASD, Trust ML, 6-Aspect ABSA, Time-Decay Rankings, FastAPI)
          </div>
          <div
            className="bg-amber-400 h-full flex items-center justify-center text-[10px] font-bold text-slate-900"
            style={{ width: "20%" }}
          >
            20% Final Phase
          </div>
        </div>
      </div>

      {/* Two Columns: Completed 80% vs Remaining 20% */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: 80% Completed */}
        <div className="lg:col-span-7 bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-emerald-800 uppercase tracking-wider">
            ✓ Completed for 80% Progress Viva
          </h3>
          <div className="space-y-3">
            {completedItems.map((item, i) => (
              <div key={i} className="bg-emerald-50/50 border border-emerald-200 rounded-xl p-4">
                <div className="flex justify-between items-center">
                  <span className="text-xs font-bold text-slate-900">{item.stage}</span>
                  <span className="text-xs font-extrabold bg-emerald-600 text-white px-2 py-0.5 rounded">
                    {item.pct}
                  </span>
                </div>
                <p className="text-xs text-slate-600 mt-1.5">{item.detail}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Remaining 20% for Final Viva */}
        <div className="lg:col-span-5 bg-white border border-slate-200 rounded-xl p-6 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-amber-800 uppercase tracking-wider">
            ⏳ Remaining 20% Work (Scheduled for Final Viva)
          </h3>
          <div className="space-y-3">
            {remainingItems.map((rem, i) => (
              <div key={i} className="bg-amber-50/60 border border-amber-200 rounded-xl p-4">
                <div className="flex justify-between items-start gap-2">
                  <span className="text-xs font-bold text-slate-900">{rem.title}</span>
                  <span className="text-[10px] font-bold bg-amber-200 text-amber-900 px-2 py-0.5 rounded whitespace-nowrap">
                    {rem.target}
                  </span>
                </div>
                <p className="text-xs text-slate-600 mt-1.5">{rem.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
