"""
Search the DLL for method body of onCOChange to find what it divides by.
Specifically look for patterns near 'co_max_power', 'co_max_spower', 'CoPower'.
"""
import re

dll = open(r"C:\Users\phili\AWBW\tools\awbw-player\lib\native\AWBWApp.Game.dll", "rb").read()

# Search for UTF-16-LE strings near each other that mention CO power fields
def find_utf16(pattern: str, context=200):
    needle = pattern.encode("utf-16-le")
    results = []
    idx = 0
    while True:
        pos = dll.find(needle, idx)
        if pos < 0:
            break
        # Grab surrounding context as utf-16-le
        start = max(0, pos - context)
        end = min(len(dll), pos + len(needle) + context)
        chunk = dll[start:end]
        # Decode printable utf-16 characters only
        try:
            decoded = chunk.decode("utf-16-le", errors="replace")
        except:
            decoded = repr(chunk)
        results.append((pos, decoded))
        idx = pos + len(needle)
    return results

# Look for "co_max" references in wide strings
for term in ["co_max_power", "co_max_spower", "CoPower", "CoMaxPower", "MaxCoPower", "tagsCo"]:
    hits = find_utf16(term)
    print(f"\n=== '{term}' ({len(hits)} hits) ===")
    for pos, ctx in hits[:3]:
        # Print only printable chars
        printable = "".join(c if c.isprintable() and c != "\x00" and ord(c) < 128 else " " for c in ctx)
        print(f"  @{pos:#x}: {printable[:300]!r}")
