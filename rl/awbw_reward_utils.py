"""
AWBW Reward Utilities

Specialized functions to calculate and apply punishments for base skipping
and other strategic considerations in RHEA's evaluation.
"""

from rl.engine import UnitType


def calculate_build_punishment(unit_type: UnitType, bases_skipped: int) -> float:
    """
    Calculate drastic punishment for base skipping during build actions.
    
    Parameters:
        unit_type: The unit type being built
        bases_skipped: Number of bases skipped in this candidate
        
    Returns:
        The punishment value to apply
    """
    # Drastic punishment for skipping bases
    base_skip_punishment = bases_skipped * -0.5  # -0.5 per skipped base
    
    # Drastic unit-specific modifiers
    if unit_type == UnitType.MECH:
        unit_modifier = -0.1  # Additional penalty for Mech
    elif unit_type == UnitType.INFANTRY:
        unit_modifier = 0.15  # Additional reward for Infantry
    else:
        unit_modifier = 0.0
    
    return base_skip_punishment + unit_modifier