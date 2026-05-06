"""Fix _scalar_errors in encoder_information.py - update to 20-scalar layout."""
INFO = r"d:\awbw\rl\encoder_information.py"

with open(INFO, "r", encoding="utf-8") as f:
    content = f.read()

# The old _scalar_errors function
old_func = '''def _scalar_errors(true_state: GameState, observer: int, scalars: np.ndarray) -> dict[str, float]:
    s = np.asarray(scalars, dtype=np.float64).reshape(-1)
    enemy = 1 - int(observer)
    co_me = true_state.co_states[observer]
    co_en = true_state.co_states[enemy]
    max_t = max(1, int(getattr(true_state, "max_turns", MAX_TURNS)))

    def norm_power_true(co_state) -> float:
        denom = co_state._scop_threshold
        if denom <= 0 or denom >= 10**11:
            return 0.0
        return min(1.0, float(co_state.power_bar) / denom)

    weather = getattr(true_state, "weather", "clear")
    n_income = sum(
        1 for p in true_state.properties if not p.is_comm_tower and not p.is_lab"
    )
    if n_income <= 0:
        share_t = 0.0
    else:
        share_t = float(true_state.count_income_properties(observer)) / float(
            n_income
        )

    # Expected encoded scalars (same as encoder)
    exp = [
        true_state.funds[observer] / 50_000.0,
        true_state.funds[enemy] / 50_000.0,
        norm_power_true(co_me),
        norm_power_true(co_en),
        float(co_me.cop_active),
        float(co_me.scop_active),
        float(co_en.cop_active),
        float(co_en.scop_active),
        true_state.turn / max_t,
        1.0 if int(true_state.active_player) == int(observer) else 0.0,
        co_me.co_id / 30.0,
        co_en.co_id / 30.0,
        1.0 if weather == "rain" else 0.0,
        1.0 if weather == "snow" else 0.0,
        getattr(true_state, "co_weather_segments_remaining", 0) / 2.0,
        share_t,
    ]
    out: dict[str, float] = {}
    for i, name in enumerate(SCALAR_NAMES):
        e = exp[i] if i < len(exp) else 0.0
        v = s[i] if i < s.shape[0] else 0.0
        out[name] = abs(float(e) - float(v))
    return out'''

# New _scalar_errors function with 20-scalar layout
new_func = '''def _scalar_errors(true_state: GameState, observer: int, scalars: np.ndarray) -> dict[str, float]:
    s = np.asarray(scalars, dtype=np.float64).reshape(-1)
    enemy = 1 - int(observer)
    co_me = true_state.co_states[observer]
    co_en = true_state.co_states[enemy]
    max_t = max(1, int(getattr(true_state, "max_turns", MAX_TURNS)))

    def _stars_norm(stars) -> float:
        if stars is None:
            return 0.0
        return min(10.0, float(stars)) / 10.0

    weather = getattr(true_state, "weather", "clear")
    n_income = sum(
        1 for p in true_state.properties if not p.is_comm_tower and not p.is_lab"
    )
    if n_income <= 0:
        share_t = 0.0
    else:
        share_t = float(true_state.count_income_properties(observer)) / float(
            n_income
        )

    # Expected encoded scalars (new 20-scalar layout)
    exp = [
        true_state.funds[observer] / 50_000.0,
        true_state.funds[enemy] / 50_000.0,
        co_me.power_bar / 50_000.0,           # raw power bar me
        _stars_norm(co_me.cop_stars),            # COP stars me (0 for Von Bolt)
        _stars_norm(co_me.scop_stars),           # SCOP stars me
        float(co_me.cop_active),
        float(co_me.scop_active),
        co_en.power_bar / 50_000.0,           # raw power bar enemy
        _stars_norm(co_en.cop_stars),            # COP stars enemy
        _stars_norm(co_en.scop_stars),           # SCOP stars enemy
        float(co_en.cop_active),
        float(co_en.scop_active),
        true_state.turn / max_t,
        1.0 if int(true_state.active_player) == int(observer) else 0.0,
        co_me.co_id / 30.0,
        co_en.co_id / 30.0,
        1.0 if weather == "rain" else 0.0,
        1.0 if weather == "snow" else 0.0,
        getattr(true_state, "co_weather_segments_remaining", 0) / 2.0,
        share_t,
    ]
    out: dict[str, float] = {}
    for i, name in enumerate(SCALAR_NAMES):
        e = exp[i] if i < len(exp) else 0.0
        v = s[i] if i < s.shape[0] else 0.0
        out[name] = abs(float(e) - float(v))
    return out'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open(INFO, "w", encoding="utf-8") as f:
        f.write(content)
    print("SUCCESS: Updated _scalar_errors to 20-scalar layout")
else:
    print("ERROR: Could not find old _scalar_errors function")
    # Debug: show what's around that area
    idx = content.find("_scalar_errors")
    if idx >= 0:
        print(f"Found _scalar_errors at index {idx}")
        print(repr(content[idx:idx+800]))
