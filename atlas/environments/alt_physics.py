"""Alternative physics configuration for validation tests."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    """Multipliers for fundamental constants. Default 1.0 = standard physics."""
    h_multiplier: float = 1.0
    c_multiplier: float = 1.0
    k_b_multiplier: float = 1.0
    e_multiplier: float = 1.0
    m_e_multiplier: float = 1.0
    r_h_multiplier: float = 1.0  # ENV-06 spectral constant


@contextlib.contextmanager
def altered_physics(config: PhysicsConfig):
    """Context manager that temporarily modifies physics constants in all quantum environments."""
    import atlas.environments.env_01 as m01
    import atlas.environments.env_02 as m02
    import atlas.environments.env_03 as m03
    import atlas.environments.env_04 as m04
    import atlas.environments.env_05 as m05
    import atlas.environments.env_06 as m06

    # Modules that have _H
    h_modules = [m01, m02, m03, m04, m05]

    originals = {}

    # Save and modify _H
    for mod in h_modules:
        if hasattr(mod, '_H'):
            originals[(id(mod), '_H')] = mod._H
            mod._H = mod._H * config.h_multiplier

    # Save and modify _M_E
    for mod in [m02, m03]:
        if hasattr(mod, '_M_E'):
            originals[(id(mod), '_M_E')] = mod._M_E
            mod._M_E = mod._M_E * config.m_e_multiplier

    # Save and modify _E
    for mod in [m01, m03]:
        if hasattr(mod, '_E'):
            originals[(id(mod), '_E')] = mod._E
            mod._E = mod._E * config.e_multiplier

    # Save and modify _C
    for mod in [m02, m05]:
        if hasattr(mod, '_C'):
            originals[(id(mod), '_C')] = mod._C
            mod._C = mod._C * config.c_multiplier

    # Save and modify _K_B
    for mod in [m05]:
        if hasattr(mod, '_K_B'):
            originals[(id(mod), '_K_B')] = mod._K_B
            mod._K_B = mod._K_B * config.k_b_multiplier

    # Save and modify _R_H (ENV-06 spectral constant)
    if hasattr(m06, '_R_H'):
        originals[(id(m06), '_R_H')] = m06._R_H
        m06._R_H = m06._R_H * config.r_h_multiplier

    # Update derived constant (_CHAR_LENGTH depends on _H, _M_E, _C)
    if hasattr(m02, '_CHAR_LENGTH'):
        originals[(id(m02), '_CHAR_LENGTH')] = m02._CHAR_LENGTH
        m02._CHAR_LENGTH = m02._H / (m02._M_E * m02._C)

    all_mods = list({id(m): m for m in h_modules + [m01, m02, m03, m05, m06]}.values())

    try:
        yield
    finally:
        # Restore all originals
        mod_by_id = {id(m): m for m in all_mods}
        for (mod_id, attr), val in originals.items():
            if mod_id in mod_by_id:
                setattr(mod_by_id[mod_id], attr, val)
