import zipfile, gzip, io, re, sys

path = r"D:\AWBW\replays\replay_1630459_GL_STD_[T1]__Gronktastic_vs_justbored_2026-04-16.zip"
with zipfile.ZipFile(path) as z:
    print("Zip entries:", z.namelist())
    for name in z.namelist():
        raw = z.read(name)
        print(f"  {name}: {len(raw)} compressed bytes")
        try:
            with gzip.open(io.BytesIO(raw)) as gz:
                text = gz.read().decode("utf-8")
        except Exception as e:
            print(f"  Not gzip ({e}) — trying raw text")
            text = raw.decode("utf-8", errors="replace")

        lines = text.split("\n")
        print(f"  Decompressed: {len(text)} chars, {len(lines)} lines (turn snapshots)")

        line = lines[0]
        print("\n--- Key fields from turn 0 ---")

        # Extract value after a PHP string key
        def get_val(key, haystack=line):
            pat = f's:{len(key.encode("utf-8"))}:"{key}";'
            idx = haystack.find(pat)
            if idx < 0:
                return "<NOT FOUND>"
            rest = haystack[idx + len(pat):]
            # Parse next PHP value
            return rest[:60]

        for field in [
            "weather_type", "weather_code", "fog", "use_powers", "official",
            "capture_win", "type", "active", "win_condition", "league", "team",
            "starting_funds", "funds",
        ]:
            print(f"  {field:20s}: {get_val(field)!r}")

        # Also extract player 0's co_id and funds
        print("\n--- Players array snippet ---")
        player_idx = line.find("awbwPlayer")
        if player_idx >= 0:
            print("  ", repr(line[player_idx:player_idx+400]))

        # Show action file (p: prefix) if present
        break

    # Check for p: (action stream) file
    print("\n--- Checking for action stream ---")
    for name in z.namelist():
        raw = z.read(name)
        try:
            with gzip.open(io.BytesIO(raw)) as gz:
                text2 = gz.read().decode("utf-8")
        except Exception:
            text2 = raw.decode("utf-8", errors="replace")
        if text2.startswith("p:"):
            print(f"  Action stream found in '{name}': first 300 chars:")
            print("  ", repr(text2[:300]))
            break
    else:
        print("  No action stream found.")
