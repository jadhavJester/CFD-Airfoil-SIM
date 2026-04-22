import numpy as np # type: ignore
import pyvista as pv # type: ignore
import scipy.ndimage # type: ignore

class FluidSolver:
    def __init__(self, airfoil_mesh, nx=220, ny=160, x_min=-1.5, x_max=3.5, y_min=-1.5, y_max=1.5, dt=0.03):
        self.nx = nx
        self.ny = ny
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.dt = dt
        
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.density = np.zeros((ny, nx))
        
        self.Y, self.X = np.mgrid[0:ny, 0:nx]
        
        dx = (x_max - x_min) / (nx - 1)
        dy = (y_max - y_min) / (ny - 1)
        
        # Calculate obstacles footprint
        bounds_grid = pv.ImageData(
            dimensions=(nx, ny, 2),
            spacing=(dx, dy, 0.1),
            origin=(x_min, y_min, -0.05)
        )
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                select = bounds_grid.select_enclosed_points(airfoil_mesh, check_surface=False)
                arr = select['SelectedPoints']
            except AttributeError:
                select = bounds_grid.select_interior_points(airfoil_mesh, check_surface=False)
                arr = select['InteriorPoints'] if 'InteriorPoints' in select.array_names else select[select.array_names[0]]
                
        inside = arr.reshape((2, ny, nx)) == 1
        self.obstacles = inside[0, :, :] # type: ignore
        
        self.vis_grid = pv.ImageData(
            dimensions=(nx, ny, 1),
            spacing=(dx, dy, 1.0),
            origin=(x_min, y_min, 0.0)
        )
        self.vis_grid['Smoke'] = self.density.ravel()
        
    def advect(self, field, u, v, order=1):
        X_back = self.X - u * self.dt * self.nx
        Y_back = self.Y - v * self.dt * self.ny
        X_back = np.clip(X_back, 0.5, self.nx - 1.5)
        Y_back = np.clip(Y_back, 0.5, self.ny - 1.5)
        advected = scipy.ndimage.map_coordinates(field, [Y_back, X_back], order=order, mode='nearest')
        advected[self.obstacles] = 0
        return advected
        
    def project(self):
        h = 1.0 / self.nx
        div = np.zeros((self.ny, self.nx))
        div[1:-1, 1:-1] = -0.5 * h * (self.u[1:-1, 2:] - self.u[1:-1, :-2] + self.v[2:, 1:-1] - self.v[:-2, 1:-1])
        div[self.obstacles] = 0
        
        p = np.zeros((self.ny, self.nx))
        for _ in range(15):
            p_old = p.copy()
            p[1:-1, 1:-1] = (div[1:-1, 1:-1] + p_old[2:, 1:-1] + p_old[:-2, 1:-1] + p_old[1:-1, 2:] + p_old[1:-1, :-2]) / 4.0
            p[self.obstacles] = 0
            
        self.u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) / h
        self.v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) / h
        
        self.u[:, 0] = self.u[:, 1]
        self.u[:, -1] = self.u[:, -2]
        self.v[:, 0] = 0
        
        self.u[self.obstacles] = 0
        self.v[self.obstacles] = 0
        
    def step(self, speed):
        inlet_speed = speed / 30.0 
        
        self.u[:, 0:2] = inlet_speed
        for y in range(self.ny):
            if (y % 8) < 3: 
                self.density[y, 0:3] = 255.0
                
        self.u = self.advect(self.u, self.u, self.v, order=2)
        self.v = self.advect(self.v, self.u, self.v, order=2)
        self.project()
        self.density = self.advect(self.density, self.u, self.v, order=1)
        
        self.density *= 0.99
        
        Vz = np.zeros_like(self.u)
        self.final_vel = np.column_stack((self.u.ravel(), self.v.ravel(), Vz.ravel())) # type: ignore
