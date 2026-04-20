import zipfile, gzip, io, sys, glob

# Find the most recent generated replay
replays = sorted(glob.glob(r"C:\Users\phili\AWBW\replays\*.zip"), key=lambda p: not "replay_1630459" in p)
# Skip the real AWBW replay, use our generated ones
gen_replays = [r for r in replays if "1630459" not in r]
if not gen_replays:
    print("No generated replays found")
    sys.exit(1)

path = gen_replays[-1]
print(f"Inspecting: {path}")

with zipfile.ZipFile(path) as z:
    print("Zip entries:", z.namelist())
    for name in z.namelist():
        raw = z.read(name)
        try:
            with gzip.open(io.BytesIO(raw)) as gz:
                text = gz.read().decode("utf-8")
        except Exception:
            text = raw.decode("utf-8", errors="replace")

        lines = text.split("\n")
        line = lines[0]

        for field in ["weather_type", "weather_code", "weather_start", "starting_funds", "aet_interval", "boot_interval"]:
            kpat = f's:{len(field.encode())}:"{field}";'
            idx = line.find(kpat)
            if idx >= 0:
                val = line[idx + len(kpat): idx + len(kpat) + 25]
                print(f"  {field:20s}: {val!r}")
        break
