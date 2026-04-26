# Opening books

- **`ranked_std_human_openings.jsonl`** — JSONL consumed by `train.py --opening-book`. Each line is one opening line for a `(map_id, seat)` with ordered `action_indices` (flat indices from `rl.env._action_to_flat`).
- **Empty file** — valid; `SelfPlayTrainer` disables the book with a warning if the path is missing, and an empty JSONL yields no indexed books (no runtime errors).
- **After building** from `tools/build_opening_book.py`, run **`python tools/validate_opening_book.py --in <path> --out <filtered_path>`** and point training at the filtered output so illegal lines never ship.

Historical note: an earlier placeholder line for map 123858 used toy indices that desynced immediately; that has been removed in favor of an empty default until you ingest real replays.
