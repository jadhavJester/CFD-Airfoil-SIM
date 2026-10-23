"""Smoke tests for solver.FluidSolver (semi-Lagrangian advection + projection)."""
import numpy as np
import pyvista as pv
import pytest

from solver import FluidSolver


@pytest.fixture(scope="module")
def obstacle():
    # small closed box inside the default domain x[-1.5, 3.5], y[-1.5, 1.5]
    return pv.Box(bounds=(-0.1, 0.3, -0.1, 0.1, -1.0, 1.0))


@pytest.fixture()
def solver(obstacle):
    return FluidSolver(obstacle, nx=40, ny=30)


def test_init_field_shapes(solver):
    assert solver.u.shape == (30, 40)
    assert solver.v.shape == (30, 40)
    assert solver.density.shape == (30, 40)
    assert solver.obstacles.shape == (30, 40)
    assert solver.obstacles.dtype == bool


def test_init_detects_obstacle(solver):
    assert solver.obstacles.any(), "expected some cells flagged inside the box"
    assert not solver.obstacles.all(), "expected free-stream cells outside the box"


def test_advect_zero_velocity_preserves_interior(solver):
    rng = np.random.default_rng(0)
    field = rng.random((30, 40))
    zeros = np.zeros((30, 40))
    out = solver.advect(field, zeros, zeros, order=1)
    inner = np.s_[2:-2, 2:-2]
    free = ~solver.obstacles[inner]
    assert np.allclose(out[inner][free], field[inner][free])


def test_advect_zeroes_obstacle_cells(solver):
    field = np.ones((30, 40))
    zeros = np.zeros((30, 40))
    out = solver.advect(field, zeros, zeros, order=1)
    assert (out[solver.obstacles] == 0).all()


def test_project_zeroes_obstacle_velocity(solver):
    rng = np.random.default_rng(1)
    solver.u[:] = rng.random((30, 40))
    solver.v[:] = rng.random((30, 40))
    solver.project()
    assert (solver.u[solver.obstacles] == 0).all()
    assert (solver.v[solver.obstacles] == 0).all()


def test_step_runs_and_keeps_shapes(solver):
    solver.step(30.0)
    assert solver.u.shape == (30, 40)
    assert solver.v.shape == (30, 40)
    assert solver.density.shape == (30, 40)
    assert solver.final_vel.shape == (30 * 40, 3)
    assert np.isfinite(solver.u).all()
    assert np.isfinite(solver.v).all()


def test_step_decays_density(solver):
    solver.step(30.0)
    peak_after_first = solver.density.max()
    solver.step(30.0)
    assert solver.density.max() <= peak_after_first
    assert (solver.density >= 0).all()
