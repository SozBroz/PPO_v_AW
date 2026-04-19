import re
raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\9acda395-f9fa-4f32-b33d-f34e8693e840.txt").read()
# All .cs files mentioning "Power" or "Bar"
paths = re.findall(r'"path":\s*"([^"]+(?:Power|Bar)[^"]*\.cs)"', raw)
for p in sorted(set(paths)):
    # Also get the sha
    block_start = raw.find(f'"path": "{p}"')
    sha_m = re.search(r'"sha":\s*"([a-f0-9]+)"', raw[block_start:block_start+300])
    sha = sha_m.group(1) if sha_m else "?"
    print(f"{sha}  {p}")
