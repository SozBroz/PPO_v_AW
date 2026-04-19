import base64, re, json

raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\5a4fb8f6-0295-40e0-acc0-2718d5d2cb82.txt").read()

# The blob API returns JSON; find the content field
data = json.loads(raw)
src = base64.b64decode(data["content"]).decode("utf-8")

# Find onCOChange
# Find where ProgressPerBar is SET (initialization path)
idx = 0
found = []
while True:
    i = src.find("ProgressPerBar", idx)
    if i < 0:
        break
    found.append((i, src[max(0,i-50):i+120]))
    idx = i + 1

print(f"All ProgressPerBar occurrences ({len(found)}):")
for pos, ctx in found:
    print(f"  @{pos}: {ctx!r}")
    print()
