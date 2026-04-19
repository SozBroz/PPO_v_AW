import json, re

raw = open(r"C:\Users\phili\.cursor\projects\c-Users-phili-AWBW\agent-tools\9acda395-f9fa-4f32-b33d-f34e8693e840.txt").read()
data = json.loads(raw)
for obj in data["tree"]:
    path = obj.get("path","")
    if "Power" in path and ".cs" in path:
        print(obj["sha"], path)
