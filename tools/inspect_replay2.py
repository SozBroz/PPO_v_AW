import zipfile, gzip, io, re

path = r"C:\Users\phili\AWBW\replays\replay_1630459_GL_STD_[T1]__Gronktastic_vs_justbored_2026-04-16.zip"
with zipfile.ZipFile(path) as z:
    raw = z.read("1630459")
    with gzip.open(io.BytesIO(raw)) as gz:
        text = gz.read().decode("utf-8")

lines = text.split("\n")
line = lines[0]

# Parse out all top-level field key/value pairs from the awbwGame object
# Find opening brace after "awbwGame":36:{
start = line.find('O:8:"awbwGame"')
snippet = line[start:start+50]
print("Header:", repr(snippet))

# Extract each s:<n>:"<key>"; then its value type/content
keys_found = []
for m in re.finditer(r's:\d+:"([^"]+)";', line[:5000]):
    keys_found.append((m.start(), m.group(1)))

print("\nFirst 60 keys in turn-0 snapshot:")
for pos, key in keys_found[:60]:
    # Get value right after this key
    val_start = pos + len(m.group(0)) - len(m.group(0)) + len(f's:{len(key.encode())}:"{key}";')
    # Re-find this specific key
    kpat = f's:{len(key.encode())}:"{key}";'
    kidx = line.find(kpat)
    val = line[kidx + len(kpat):kidx + len(kpat) + 40]
    print(f"  {key:25s}: {val!r}")
    if len(keys_found) > 5 and key == "units":
        break

# Specifically check the weather fields
print("\n--- Weather fields ---")
for field in ["weather_type", "weather_code", "weather_start"]:
    kpat = f's:{len(field.encode())}:"{field}";'
    idx = line.find(kpat)
    if idx >= 0:
        val = line[idx + len(kpat): idx + len(kpat) + 30]
        print(f"  {field}: {val!r}")

# Check starting_funds and aet_interval
print("\n--- Other fixed fields ---")
for field in ["starting_funds", "aet_interval", "boot_interval", "type", "capture_win", "official"]:
    kpat = f's:{len(field.encode())}:"{field}";'
    idx = line.find(kpat)
    if idx >= 0:
        val = line[idx + len(kpat): idx + len(kpat) + 25]
        print(f"  {field:20s}: {val!r}")
