"""Simplified aerodynamic estimates for the Virtual Wind Tunnel app.

The estimates here are the ones previously computed inline in
``main.run_simulation``: a thin-airfoil-style lift-curve slope with a
camber offset for the NACA 4-digit series, plus a parabolic drag polar.
They are intended as interactive feedback, not validated aerodynamics.
"""
import numpy as np  # type: ignore

RHO = 1.225  # air density, kg/m^3
CD_0 = 0.015  # profile drag coefficient
ASPECT_RATIO = 6.0  # effective wing aspect ratio
OSWALD_E = 0.8  # Oswald efficiency factor
REF_AREA = 1.0  # reference wing area, m^2


def camber_ratio(naca_str):
    """Return the max-camber ratio (m) of a validated 4-digit NACA string."""
    if len(naca_str) != 4 or not naca_str.isdigit():
        raise ValueError("NACA string must be exactly 4 digits")
    return int(naca_str[0]) / 100.0


def lift_drag_coefficients(naca_str, aoa_deg):
    """Return (cl, cd) for a 4-digit NACA airfoil at the given angle of attack."""
    m = camber_ratio(naca_str)
    aoa_rad = np.radians(aoa_deg)
    cl_0 = 2 * np.pi * m * 2
    cl = cl_0 + 2 * np.pi * aoa_rad
    # Virtual wind tunnels act somewhat like 2D infinite wings (less induced drag)
    cd = CD_0 + (cl ** 2) / (np.pi * OSWALD_E * ASPECT_RATIO * 3.0)
    return cl, cd


def aero_estimates(naca_str, aoa_deg, speed):
    """Return a dict of lift/drag/pressure/velocity estimates at `speed` m/s."""
    cl, cd = lift_drag_coefficients(naca_str, aoa_deg)
    lift = 0.5 * RHO * (speed ** 2) * REF_AREA * cl
    drag = 0.5 * RHO * (speed ** 2) * REF_AREA * cd
    v_max = speed * 1.5
    p_dyn = 0.5 * RHO * (v_max ** 2 - speed ** 2)
    return {
        "cl": cl,
        "cd": cd,
        "lift": lift,
        "drag": drag,
        "weight": np.abs(lift),
        "thrust": np.abs(drag),
        "v_max": v_max,
        "p_dyn": p_dyn,
    }
