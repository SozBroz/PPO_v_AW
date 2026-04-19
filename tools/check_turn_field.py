"""Check the ACTUAL game-level turn field in our homebrew replay, not player.turn."""
import gzip, io, zipfile
from pathlib import Path

path = Path(r"C:\Users\phili\AWBW\replays\126694.zip")
with zipfile.ZipFile(path) as z:
    raw = z.read(z.namelist()[0])
with gzip.open(io.BytesIO(raw)) as gz:
    line = gz.read().decode("utf-8").split("\n")[0]

# Find game-level 'turn' which comes after 'win_condition' in the awbwGame header
# and player-level 'turn' which is inside each awbwPlayer block
win_idx = line.find('s:13:"win_condition";')
if win_idx >= 0:
    after_wc = line[win_idx:]
    # find first turn after win_condition
    turn_pat = 's:4:"turn";'
    ti = after_wc.find(turn_pat)
    if ti >= 0:
        val = after_wc[ti+len(turn_pat):ti+len(turn_pat)+30]
        print(f"Game-level turn (after win_condition): {val!r}")

# Also check password field
pw_pat = 's:8:"password";'
pi2 = line.find(pw_pat)
if pi2 >= 0:
    val = line[pi2+len(pw_pat):pi2+len(pw_pat)+20]
    print(f"password: {val!r}")
