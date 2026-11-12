"""Offscreen smoke tests for the desktop app (main.WindTunnelApp).

Qt is optional: these tests skip when PyQt5/pyvistaqt are not installed, so
``pytest`` still runs on minimal installs. CI installs the full runtime
requirements, so the smoke test runs there and catches import errors and
widget-construction drift in the entry point that the other suites never
touch.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pyvistaqt")


@pytest.fixture(scope="module")
def app_window():
    from PyQt5.QtWidgets import QApplication

    import main

    app = QApplication.instance() or QApplication([])
    window = main.WindTunnelApp()
    yield window
    window.timer.stop()
    window.close()
    app.quit()


def test_window_builds_with_defaults(app_window):
    assert app_window.windowTitle().startswith("Virtual Wind Tunnel")
    assert app_window.naca_input.text() == "4412"
    assert app_window.aoa_slider.value() == 10
    assert app_window.speed_slider.value() == 20
    assert app_window.aircraft_slider.value() == 100


def test_run_simulation_updates_stats(app_window):
    app_window.run_simulation()
    assert app_window.fluid_solver is not None
    # stats labels are filled from aero_estimates (defaults: 4412, 10 deg, 120 m/s)
    assert app_window.lbl_lift.text() != "0.00 N"
    assert app_window.lbl_drag.text() != "0.00 N"
    assert app_window.lbl_vmax.text() != "0.00 m/s"


def test_update_simulation_steps_solver(app_window):
    app_window.run_simulation()
    app_window.update_simulation()
    assert app_window.fluid_solver.final_vel.shape[1] == 3
