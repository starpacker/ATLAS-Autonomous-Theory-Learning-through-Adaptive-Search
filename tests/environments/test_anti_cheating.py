# tests/environments/test_anti_cheating.py
"""Anti-cheating tests: verify no physics knowledge leaks through environment interfaces."""
import re
import numpy as np
from atlas.environments.registry import get_all_environments
from atlas.types import KnobType


PHYSICS_TERMS = [
    "photon", "electron", "proton", "neutron", "quark",
    "frequency", "wavelength", "momentum", "energy", "mass",
    "spin", "charge", "voltage", "current", "magnetic",
    "slit", "diffraction", "interference", "spectrum",
    "planck", "bohr", "compton", "stern", "gerlach",
    "quantum", "classical", "particle", "wave",
    "temperature", "pressure", "force", "acceleration",
]


def test_all_knob_names_anonymous():
    for env in get_all_environments():
        schema = env.get_schema()
        for knob in schema.knobs:
            assert re.match(r"^knob_\d+$", knob.name), (
                f"{schema.env_id}: knob '{knob.name}' leaks physics semantics"
            )


def test_all_detector_names_anonymous():
    for env in get_all_environments():
        schema = env.get_schema()
        for det in schema.detectors:
            assert re.match(r"^detector_\d+$", det.name), (
                f"{schema.env_id}: detector '{det.name}' leaks physics semantics"
            )


def test_continuous_knobs_normalized():
    for env in get_all_environments():
        schema = env.get_schema()
        for knob in schema.knobs:
            if knob.knob_type == KnobType.CONTINUOUS:
                assert knob.range_min >= -1.0, (
                    f"{schema.env_id}/{knob.name}: range_min={knob.range_min} < -1"
                )
                assert knob.range_max <= 1.0, (
                    f"{schema.env_id}/{knob.name}: range_max={knob.range_max} > 1"
                )


def test_no_physics_terms_in_schema():
    for env in get_all_environments():
        schema = env.get_schema()
        schema_str = str(schema).lower()
        for term in PHYSICS_TERMS:
            assert term not in schema_str, (
                f"{schema.env_id}: schema contains physics term '{term}'"
            )


def test_all_environments_runnable():
    for env in get_all_environments():
        schema = env.get_schema()
        knobs = {}
        for knob in schema.knobs:
            if knob.knob_type == KnobType.DISCRETE:
                knobs[knob.name] = knob.options[0]
            elif knob.knob_type == KnobType.INTEGER:
                knobs[knob.name] = int((knob.range_min + knob.range_max) / 2)
            else:
                knobs[knob.name] = (knob.range_min + knob.range_max) / 2
        result = env.run(knobs)
        for det in schema.detectors:
            assert det.name in result, (
                f"{schema.env_id}: missing detector '{det.name}' in output"
            )
