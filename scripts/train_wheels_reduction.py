import json
import time
import os

# Configuration
LOG_PATH = r"D:\awbw\logs\game_log.jsonl"
OVERRIDE_PATH = r"D:\awbw\fleet\pc-b\operator_train_args_override.json"

# Thresholds: Average "remaining" must be LESS THAN OR EQUAL TO these
INCOME_THRESHOLDS = {
    "neutral_income_remaining_by_day_7": 17,
    "neutral_income_remaining_by_day_9": 10,
    "neutral_income_remaining_by_day_11": 7,
    "neutral_income_remaining_by_day_13": 5
}

def get_latest_games(n=50):
    games = []
    if not os.path.exists(LOG_PATH): return []
    
    try:
        with open(LOG_PATH, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
            for line in reversed(lines):
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError: continue
                if len(games) >= n: break
    except Exception as e:
        print(f"Error reading log: {e}")
    return games

def check_criteria(games):
    # 1. Win Rate Check (Filter for 'hist' mode)
    hist_games = [g for g in games if g.get("async_rollout_mode") == "hist"]
    if not hist_games:
        return False, "No 'hist' mode games found."

    p0_wins = sum(1 for g in hist_games if g.get("winner") == 0)
    wr = p0_wins / len(hist_games)
    
    if wr < 0.55:
        return False, f"WR too low: {wr:.2%} (need 55% in 'hist')"

    # 2. Average Neutral Income Checks
    # We treat 'null' as 0 because it means the game ended and 0 neutrals remain.
    analysis_results = []
    for key, limit in INCOME_THRESHOLDS.items():
        avg = sum((g.get(key) or 0) for g in games) / len(games)
        if avg > limit:
            return False, f"Avg {key} too high: {avg:.1f} (limit {limit})"
        analysis_results.append(f"{key}: {avg:.1f}")

    return True, f"Passed! WR: {wr:.2%}, Avgs: {', '.join(analysis_results)}"

def update_override(current_greedy):
    try:
        with open(OVERRIDE_PATH, 'r') as f:
            data = json.load(f)
        
        new_greedy = round(current_greedy - 0.01, 3)
        current_gate = data["args"].get("--capture-move-gate", 1.0)
        new_gate = round(current_gate - 0.07, 3)
        
        # Ensure values don't go below 0
        new_greedy = max(0, new_greedy)
        new_gate = max(0, new_gate)
        
        data["args"]["--learner-greedy-mix"] = new_greedy
        data["args"]["--capture-move-gate"] = new_gate
        
        with open(OVERRIDE_PATH, 'w') as f:
            json.dump(data, f, indent=2)
        
        return new_greedy, new_gate
    except Exception as e:
        print(f"Error updating override: {e}")
        return None, None

def main():
    last_processed_id = -1
    waiting_for_greedy_sync = None
    
    print("Monitoring log for progression marks...")

    while True:
        latest_batch = get_latest_games(1)
        if not latest_batch:
            time.sleep(10)
            continue
            
        latest = latest_batch[0]
        game_id = latest.get("game_id", 0)
        current_greedy = latest.get("learner_greedy_mix")

        # Sync check: Wait for log to reflect the new override value
        if waiting_for_greedy_sync is not None:
            if current_greedy == waiting_for_greedy_sync:
                print(f"Log synced to {current_greedy}. Resuming monitoring.")
                waiting_for_greedy_sync = None
            else:
                time.sleep(15)
                continue

        # Trigger check at mod 50
        if game_id > 0 and game_id % 50 == 0 and game_id != last_processed_id:
            batch = get_latest_games(50)
            if len(batch) < 50:
                last_processed_id = game_id
                continue

            print(f"\n--- Game {game_id} Analysis ---")
            passed, msg = check_criteria(batch)
            print(msg)
            
            if passed:
                new_g, new_gate = update_override(current_greedy)
                if new_g is not None:
                    print(f"PROGRESSING: New Greedy {new_g}, New Gate {new_gate}")
                    waiting_for_greedy_sync = new_g
            else:
                print("STALLED: Did not meet all criteria.")
            
            last_processed_id = game_id

        time.sleep(20)

if __name__ == "__main__":
    main()
