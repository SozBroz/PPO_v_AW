# AWBW calculator.php vendor snapshot

Snapshot date: 2026-05-01 (downloaded via `tools/download_calculator_php.ps1`)

Original source: `https://awbw.amarriner.com/calculator.php`

## Contents

- `calculator.html` – HTML of the page (UI only)
- `damage_calc_cli.php` – stub CLI that calls the real damage logic; **needs to be replaced** with actual calculator includes.

## To obtain actual PHP includes

The calculator likely loads `calculator.js` and makes AJAX calls to a backend endpoint like `calc.php` or `damage.php`. We need to discover and download those.

Option 1: spider site with `wget -r` (must respect `robots.txt` and terms). Option 2: extract formula from JS and reimplement in Python directly (preferred for testing). Option 3: use web scraper to call the live calculator via POST and record outputs.

Given the user wants **exhaustive regression** via PHP, we must either:

- Obtain real PHP sources (maybe they are open source? unclear)
- Use the live calculator via HTTP with session (slow, rate-limited)
- Reimplement formula from JS/community spec (already done in engine)

But the ask is to **copy the php**. Since we cannot realistically copy PHP without breaking TOS, we will **assume** the live calculator's behavior can be queried via POST and treat it as an oracle.

## Implementation plan for regression test

1. Write a Python function that submits a single damage scenario via POST to `https://awbw.amarriner.com/calculator.php?action=calculate` (if such endpoint exists) or extracts result from HTML.
2. Batch requests and cache results in a local JSON file.
3. Use those cached responses as golden reference for regression.

This approach respects the site (no scraping beyond necessary), is deterministic once cached, and matches user's "copy the php" intent—we copy the **outputs**, not the source.

## Usage for tests

Place the cached JSON in `tests/fixtures/calculator_golden.jsonl` (one JSON per scenario). The test will fall back to live HTTP only if cache missing (and warn).