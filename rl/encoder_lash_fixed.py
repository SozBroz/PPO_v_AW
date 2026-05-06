"""Lash fix: correct _co_tile_attack_bonus_for_category function."""
# This is a reference file showing the correct Lash block.

    # Lash (co_id=16): +10% per defense star from the attacker's tile.
    # AWBW canon: D2D = +10%/star; COP = no change (only DEF doubled);
    # SCOP = +20%/star (ATK doubled). 
    # Air/copter exclusion is handled in ``combat.py`` / ``attack_value_for_unit()``.
    # This is a map-position feature — the NN learns air units don't benefit.
    if co_id == 16:
        stars = max(0.0, float(defense_norm)) * 4.0
        if scop_active:
            return stars * 0.20   # +20%/star during SCOP
        # D2D and COP: +10%/star (COP only doubles defense, not ATK)
        return stars * 0.10
