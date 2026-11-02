"""Tests for aerodynamics (lift/drag estimates extracted from the wind tunnel app)."""
import numpy as np
import pytest

from aerodynamics import aero_estimates, camber_ratio, lift_drag_coefficients


def test_camber_ratio():
    assert camber_ratio("4412") == pytest.approx(0.04)
    assert camber_ratio("0012") == pytest.approx(0.0)
    with pytest.raises(ValueError):
        camber_ratio("12")
    with pytest.raises(ValueError):
        camber_ratio("abcd")


def test_symmetric_airfoil_zero_aoa():
    cl, cd = lift_drag_coefficients("0012", 0.0)
    assert cl == pytest.approx(0.0)
    assert cd == pytest.approx(0.015)  # profile drag only


def test_cambered_airfoil_zero_aoa():
    cl, cd = lift_drag_coefficients("4412", 0.0)
    assert cl == pytest.approx(2 * np.pi * 0.08, rel=1e-9)
    expected_cd = 0.015 + cl ** 2 / (np.pi * 0.8 * 6.0 * 3.0)
    assert cd == pytest.approx(expected_cd, rel=1e-9)


def test_negative_aoa_reduces_cl():
    cl, _ = lift_drag_coefficients("0012", -10.0)
    assert cl < 0.0
    assert cl == pytest.approx(-2 * np.pi * np.radians(10.0), rel=1e-9)


def test_aero_estimates_fields():
    est = aero_estimates("4412", 5.0, 20.0)
    q = 0.5 * 1.225 * 20.0 ** 2
    cl, cd = lift_drag_coefficients("4412", 5.0)
    assert est["cl"] == pytest.approx(cl)
    assert est["cd"] == pytest.approx(cd)
    assert est["lift"] == pytest.approx(q * cl)
    assert est["drag"] == pytest.approx(q * cd)
    assert est["weight"] == pytest.approx(abs(est["lift"]))
    assert est["thrust"] == pytest.approx(abs(est["drag"]))
    assert est["v_max"] == pytest.approx(30.0)
    assert est["p_dyn"] == pytest.approx(0.5 * 1.225 * (30.0 ** 2 - 20.0 ** 2))


def test_forces_scale_with_speed_squared():
    slow = aero_estimates("4412", 5.0, 10.0)
    fast = aero_estimates("4412", 5.0, 20.0)
    assert fast["lift"] == pytest.approx(4 * slow["lift"])
    assert fast["drag"] == pytest.approx(4 * slow["drag"])


def test_zero_speed_gives_zero_forces():
    est = aero_estimates("4412", 10.0, 0.0)
    assert est["lift"] == pytest.approx(0.0)
    assert est["drag"] == pytest.approx(0.0)
    assert est["weight"] == pytest.approx(0.0)
    assert est["thrust"] == pytest.approx(0.0)
    assert est["v_max"] == pytest.approx(0.0)
    assert est["p_dyn"] == pytest.approx(0.0)
