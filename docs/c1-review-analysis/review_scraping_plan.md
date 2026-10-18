# Super Plan: Scraping Tourism Reviews for Component 1

As the developer for **Component 1: Trust-Aware Review Analysis Engine**, your first major hurdle is acquiring high-quality, raw English tourism reviews from platforms like Google Maps and TripAdvisor. 

Here is a professional, enterprise-grade plan to build a robust, scalable, and stealthy review scraper.

---

## 1. Target Identification & Strategy

Different platforms have different defenses. We need a tailored approach for each:

| Platform | Difficulty | Best Approach | Data Richness |
| :--- | :--- | :--- | :--- |
| **Google Maps** | High | Unofficial APIs / Playwright (Browser Automation) | Extremely high (millions of reviews, great metadata). |
| **TripAdvisor** | Medium | BeautifulSoup / Scrapy (HTML Parsing) | High (detailed, aspect-specific reviews). |
| **Booking.com** | Medium | GraphQL / Hidden APIs | Good for hotel-specific verified reviews. |

> [!TIP]
> **Pro-Tip:** Don't start by parsing HTML if you don't have to. Always look for hidden APIs (XHR/Fetch requests in the browser network tab) first. Platforms often send clean JSON data to their front-end, which is much easier to intercept and parse than HTML.

---

## 2. Recommended Tech Stack

To scrape "like a pro," you need a stack that handles rate limits, dynamic content (JavaScript), and data parsing efficiently.

*   **Language:** Python (The undisputed king of scraping and NLP).
*   **Static Scraping:** `BeautifulSoup4` + `requests` (Fast, good for basic HTML).
*   **Dynamic Scraping (JavaScript heavy):** `Playwright` or `Selenium`. (Playwright is heavily recommended for its speed and modern architecture).
*   **Framework (Optional but Pro):** `Scrapy` (If you plan to scrape hundreds of thousands of reviews, Scrapy is an asynchronous beast).
*   **Data Storage:** `MongoDB` (NoSQL is perfect for unstructured review data) or `PostgreSQL` (if you want strict relational tables for your trust scores later).

---

## 3. Data Schema: What to Extract

To support your fake review detection and aspect-based sentiment analysis, you need more than just the text. Extract these fields:

```json
{
  "review_id": "unique_hash",
  "source": "GoogleMaps",
  "hotel_name": "Cinnamon Grand Colombo",
  "reviewer_id": "user_123",
  "reviewer_profile_stats": {
    "total_reviews_made": 15,    // CRITICAL for fake review detection
    "account_creation_date": "2022-01-01" 
  },
  "rating": 4.5,
  "review_text": "The food was amazing but the Wi-Fi was terrible.",
  "review_date": "2025-10-12",
  "language": "en",
  "helpful_votes": 12
}
```

> [!IMPORTANT]
> **Fake Review Detection Context:** Notice the `reviewer_profile_stats`. To build a "Trust-Aware" engine, you need metadata about the *author*, not just the text. A user who created their account yesterday and gave a 5-star review with no text is suspicious.

---

## 4. Bypassing Anti-Scraping Defenses

If you just run a basic python script, you will get blocked within minutes. You must implement evasion techniques:

1.  **Rotating Proxies:** Use a proxy provider (like BrightData, Smartproxy, or Webshare) to rotate your IP address. If scraping Sri Lankan hotels, try to use diverse IP locations.
2.  **User-Agent Rotation:** Randomize your `User-Agent` headers so you don't look like a Python script. Use the `fake-useragent` library.
3.  **Human-like Delays:** Don't scrape 100 pages a second. Introduce random sleep intervals (`time.sleep(random.uniform(1.5, 4.2))`).
4.  **Headless Browsers:** For Google Maps, use Playwright with the `stealth` plugin (`playwright-stealth`) to bypass bot detection like Cloudflare.

---

## 5. Execution Plan (Step-by-Step)

### Phase 1: Proof of Concept (Week 1)
*   [ ] Pick one target hotel on TripAdvisor.
*   [ ] Write a simple `BeautifulSoup` script to extract the first page of reviews.
*   [ ] Save the output to a simple CSV file.

### Phase 2: Pagination & API Reverse Engineering (Week 2)
*   [ ] Open the Network Tab in Chrome Developer Tools.
*   [ ] Click "Next Page" on the reviews.
*   [ ] Find the hidden API endpoint (usually returns JSON).
*   [ ] Replicate that API request in Python to bypass HTML parsing entirely.

### Phase 3: Scale & Stealth (Week 3)
*   [ ] Integrate proxy rotation.
*   [ ] Write a loop to iterate over a list of 100 Sri Lankan hotels/destinations.
*   [ ] Implement error handling (e.g., if blocked, wait 5 minutes and retry).
*   [ ] Push scraped data to a local SQLite or MongoDB database.

### Phase 4: Cleaning (Prep for NLP) (Week 4)
*   [ ] Write a script to drop duplicate `review_id`s.
*   [ ] Filter out any reviews where `language != 'en'`.
*   [ ] Remove emojis and special characters to prep for the Trust-Aware analysis.

---

## 6. Ethical & Legal Considerations

*   **Terms of Service:** Scraping usually violates the ToS of platforms like TripAdvisor. Keep your scraping volume reasonable (academic/research scale, not commercial scale).
*   **Anonymization:** As stated in your proposal (Page 6), *do not* store real usernames. Hash the `reviewer_id` immediately upon scraping so you can track users for fake-review detection without knowing who they are.
*   **Robots.txt:** Always check `website.com/robots.txt` to see which paths platforms explicitly ask bots not to crawl.

> [!CAUTION]
> Do not hit platforms with aggressive concurrent requests. Not only will you get IP banned, but it can be considered a Denial of Service (DoS) attack. Keep it slow and steady.
