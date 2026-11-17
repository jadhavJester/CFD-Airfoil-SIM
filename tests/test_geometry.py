"""Smoke tests for geometry.generate_naca_airfoil (NACA 4-digit mesh generation).

PyVista is a heavy optional dependency: these tests skip when it is not
installed, so ``pytest`` still runs on minimal installs instead of aborting
the whole suite at collection. CI installs the full runtime requirements,
so the tests run there.
"""
import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from geometry import generate_naca_airfoil


def test_returns_nonempty_mesh():
    mesh = generate_naca_airfoil("0012", 0.0, num_points=60)
    assert isinstance(mesh, pv.DataSet)
    assert mesh.n_points > 0
    assert mesh.n_cells > 0


def test_symmetric_airfoil_is_symmetric():
    mesh = generate_naca_airfoil("0012", 0.0, chord=1.0, num_points=80)
    ys = mesh.points[:, 1]
    assert ys.max() > 0.05  # 12% thick -> ~0.06 half-thickness
    assert ys.min() < -0.05
    # symmetric about the chord line: extents mirror each other
    assert abs(ys.max() + ys.min()) < 0.01


def test_chord_scaling():
    mesh = generate_naca_airfoil("0012", 0.0, chord=2.0, num_points=60)
    xs = mesh.points[:, 0]
    # x spans [-0.25*chord, 0.75*chord] by construction
    assert xs.max() - xs.min() == pytest.approx(2.0, abs=1e-6)
    assert xs.min() == pytest.approx(-0.5, abs=1e-6)
    assert xs.max() == pytest.approx(1.5, abs=1e-6)


def test_cambered_airfoil_has_camber():
    mesh = generate_naca_airfoil("4412", 0.0, chord=1.0, num_points=80)
    ys = mesh.points[:, 1]
    assert ys.max() > 0.09  # 4% camber lifts the upper surface well above symmetric


def test_invalid_naca_string_raises():
    with pytest.raises(ValueError):
        generate_naca_airfoil("12", 0.0)
    with pytest.raises(ValueError):
        generate_naca_airfoil("00121", 0.0)
    with pytest.raises(ValueError, match="exactly 4 digits"):
        generate_naca_airfoil("abcd", 0.0)


def test_invalid_parameters_raise():
    # too few points collapses the outline to a single point and extrudes
    # an empty mesh, so it is rejected instead of failing silently
    with pytest.raises(ValueError, match="num_points"):
        generate_naca_airfoil("0012", 0.0, num_points=1)
    with pytest.raises(ValueError, match="num_points"):
        generate_naca_airfoil("0012", 0.0, num_points=2)
    with pytest.raises(ValueError, match="chord"):
        generate_naca_airfoil("0012", 0.0, chord=0.0)
    with pytest.raises(ValueError, match="span"):
        generate_naca_airfoil("0012", 0.0, span=-1.0)


def test_aoa_rotates_mesh():
    flat = generate_naca_airfoil("0012", 0.0, num_points=60)
    pitched = generate_naca_airfoil("0012", 10.0, num_points=60)
    assert not np.allclose(flat.points, pitched.points)
    # positive AoA (nose-up via -rotation) moves the leading edge down
    assert pitched.points[:, 1].min() < flat.points[:, 1].min()


def test_span_extrusion():
    mesh = generate_naca_airfoil("0012", 0.0, chord=1.0, span=0.5, num_points=60)
    zs = mesh.points[:, 2]
    assert zs.max() - zs.min() == pytest.approx(0.5, abs=1e-6)
