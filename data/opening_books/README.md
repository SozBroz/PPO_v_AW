# Opening books

- **`ranked_std_human_openings.jsonl`** — JSONL consumed by `train.py --opening-book`. Each line is one opening line for a `(map_id, seat)` with ordered `action_indices` (flat indices from `rl.env._action_to_flat`).
- **Empty file** — valid; `SelfPlayTrainer` disables the book with a warning if the path is missing, and an empty JSONL yields no indexed books (no runtime errors).
- **After building** from `tools/build_opening_book.py`, run **`python tools/validate_opening_book.py --in <path> --out <filtered_path>`** and point training at the filtered output so illegal lines never ship.

## Regenerate **both** seats at the **same** endpoint as the current P1-only book

The env can force books for learner and opponent. If `ranked_std_human_openings.jsonl` only has `seat: 1` lines, rebuild from demos that include **both** `active_player` values, truncated so the anchor seat (P1) takes exactly the same number of moves as today:

1. **Backup** the current book (needed as the depth anchor):

   `copy data\opening_books\ranked_std_human_openings.jsonl data\opening_books\ranked_std_human_openings_p1_anchor.jsonl`

2. **Filter** the replay manifest to the games referenced in that anchor:

   `python tools/extract_manifest_for_anchor_games.py --anchor data/opening_books/ranked_std_human_openings_p1_anchor.jsonl --manifest data/human_openings/raw/manifest_filtered.jsonl --out data/human_openings/raw/manifest_anchor_openings.jsonl`

3. **Regenerate demos** (requires zips under `--manifest-base-dir`, e.g. `data/human_openings/raw/games\*.zip`):

   `python scripts/replay_to_human_demos.py --manifest data/human_openings/raw/manifest_anchor_openings.jsonl --manifest-base-dir data/human_openings/raw --opening-only --both-seats --max-days-from-manifest --include-move --out data/human_openings/demos/opening_demos_both.jsonl`

4. **Emit** two lines per session (seat 0 + seat 1), same chronological endpoint as the anchor:

   `python tools/build_opening_book_both_anchored.py --anchor data/opening_books/ranked_std_human_openings_p1_anchor.jsonl --demos data/human_openings/demos/opening_demos_both.jsonl --out data/opening_books/ranked_std_human_openings.jsonl --strict`

`--strict` checks that every anchor-seat flat index matches the demo replay.

Historical note: an earlier placeholder line for map 123858 used toy indices that desynced immediately; that has been removed in favor of an empty default until you ingest real replays.
