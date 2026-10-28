import numpy as np # type: ignore
import pyvista as pv # type: ignore

def generate_naca_airfoil(naca_str, aoa_deg, chord=1.0, span=1.0, num_points=200):
    """
    Generates a 3D extruded PyVista mesh of a NACA 4-digit airfoil.
    """
    if len(naca_str) != 4:
        raise ValueError("NACA string must be exactly 4 digits")
    if not naca_str.isdigit():
        raise ValueError("NACA string must be exactly 4 digits")

    m = int(naca_str[0]) / 100.0
    p = int(naca_str[1]) / 10.0
    t = int(naca_str[2:]) / 100.0

    beta = np.linspace(0, np.pi, num_points)
    x = 0.5 * (1 - np.cos(beta))  
    
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if p > 0:
        mask1 = x < p
        mask2 = x >= p
        if np.any(mask1):
            yc[mask1] = (m / p**2) * (2 * p * x[mask1] - x[mask1]**2)  # type: ignore
            dyc_dx[mask1] = (2 * m / p**2) * (p - x[mask1])  # type: ignore
        if np.any(mask2):
            yc[mask2] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[mask2] - x[mask2]**2)  # type: ignore
            dyc_dx[mask2] = (2 * m / (1 - p)**2) * (p - x[mask2])  # type: ignore

    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    # Combine to form closed loop (trailing edge to leading edge, back to trailing edge)
    x_coords = np.concatenate((xu[::-1], xl[1:]))  # type: ignore
    y_coords = np.concatenate((yu[::-1], yl[1:]))  # type: ignore
    z_coords = np.zeros_like(x_coords)

    # Scale to chord length and center around aerodynamic center (approx 0.25 chord)
    points = np.column_stack((x_coords * chord - 0.25 * chord, y_coords * chord, z_coords))
    
    # Create the 2D polygon surface
    poly = pv.PolyData(points)
    faces = [len(points)] + list(range(len(points)))
    poly.faces = np.array(faces)
    
    # Triangulate to get a filled 2D surface
    surface = poly.delaunay_2d()
    
    # Extrude the surface along Z to create a 3D wing
    mesh = surface.extrude([0, 0, span], capping=True)
    mesh.translate([0, 0, -span / 2.0], inplace=True)
    
    # Rotate by angle of attack
    mesh.rotate_z(-aoa_deg, inplace=True)
    
    return mesh
