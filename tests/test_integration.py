"""Integration test: verify the complete foundation layer works end-to-end."""
import numpy as np
from atlas.types import EnvSchema
from atlas.dsl.expr import Const, Var, BinOp, UnaryOp
from atlas.dsl.operators import Op, DSL_0
from atlas.dsl.canonicalize import canonicalize
from atlas.dsl.serialize import to_str, from_str, to_dict, from_dict
from atlas.environments.registry import get_environment, get_all_environments
from atlas.types import KnobType


def test_all_12_environments_registered():
    envs = get_all_environments()
    ids = {e.get_schema().env_id for e in envs}
    expected = {f"ENV_{i:02d}" for i in range(1, 13)}
    assert ids == expected, f"Missing: {expected - ids}, Extra: {ids - expected}"


def test_dsl_roundtrip():
    """Build an expression, canonicalize it, serialize it, deserialize it."""
    inner = BinOp(Op.MUL, Var("knob_0"), Const(3.14))
    expr = BinOp(Op.MUL, UnaryOp(Op.COS, inner), UnaryOp(Op.COS, inner))

    canon = canonicalize(expr)
    s = to_str(canon)
    recovered = from_str(s)
    assert recovered == canon

    d = to_dict(canon)
    recovered2 = from_dict(d)
    assert recovered2 == canon


def test_environment_data_generation():
    """Generate data from each environment and verify shapes."""
    for env in get_all_environments():
        schema = env.get_schema()
        knobs = {}
        for k in schema.knobs:
            if k.knob_type == KnobType.DISCRETE:
                knobs[k.name] = k.options[0]
            elif k.knob_type == KnobType.INTEGER:
                knobs[k.name] = int((k.range_min + k.range_max) / 2)
            else:
                knobs[k.name] = (k.range_min + k.range_max) / 2

        result = env.run(knobs)
        for det in schema.detectors:
            assert det.name in result, f"{schema.env_id}: missing {det.name}"
            val = result[det.name]
            if det.output_type == "scalar":
                assert isinstance(val, (float, np.floating, int, np.integer)), \
                    f"{schema.env_id}/{det.name}: expected scalar, got {type(val)}"
            elif det.output_type == "array_1d":
                assert isinstance(val, np.ndarray), \
                    f"{schema.env_id}/{det.name}: expected ndarray, got {type(val)}"
                if det.length is not None:
                    assert len(val) == det.length, \
                        f"{schema.env_id}/{det.name}: expected len {det.length}, got {len(val)}"


def test_dsl0_is_minimal():
    """DSL_0 should contain exactly 10 operators, no physics-specific ones."""
    assert len(DSL_0) == 10
    names = {op.value for op in DSL_0}
    forbidden = {"complex_abs", "complex_mul", "born_rule", "probability"}
    assert names & forbidden == set()
