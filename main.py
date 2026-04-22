import sys
import numpy as np # type: ignore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QLineEdit, QPushButton, QFormLayout
from PyQt5.QtCore import Qt, QTimer # type: ignore
from pyvistaqt import QtInteractor # type: ignore
import pyvista as pv # type: ignore

from geometry import generate_naca_airfoil # type: ignore
from solver import FluidSolver # type: ignore
from typing import Any # type: ignore

class WindTunnelApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Virtual Wind Tunnel - Dynamic Navier-Stokes Dynamics")
        self.resize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#0a0a14")
        main_layout.addWidget(self.plotter.interactor, stretch=3)

        control_panel = QWidget()
        control_layout = QFormLayout(control_panel)
        main_layout.addWidget(control_panel, stretch=1)

        self.naca_input = QLineEdit("4412")
        control_layout.addRow(QLabel("NACA Airfoil:"), self.naca_input)

        self.aoa_slider = QSlider(Qt.Horizontal)
        self.aoa_slider.setRange(-20, 20)
        self.aoa_slider.setValue(10)
        self.aoa_label = QLabel("Angle of Attack: 10°")
        self.aoa_slider.valueChanged.connect(self.update_aoa_label)
        control_layout.addRow(self.aoa_label, self.aoa_slider)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(0, 200)
        self.speed_slider.setValue(20)
        self.speed_label = QLabel("Headwind Speed: 20 m/s")
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        control_layout.addRow(self.speed_label, self.speed_slider)

        self.aircraft_slider = QSlider(Qt.Horizontal)
        self.aircraft_slider.setRange(0, 500)
        self.aircraft_slider.setValue(100)
        self.aircraft_label = QLabel("Aircraft Speed: 100 m/s")
        self.aircraft_slider.valueChanged.connect(self.update_aircraft_label)
        control_layout.addRow(self.aircraft_label, self.aircraft_slider)

        self.update_btn = QPushButton("Calculate Flow & Run")
        self.update_btn.clicked.connect(self.run_simulation)
        control_layout.addRow(self.update_btn)

        # Real-time Stats Panel
        self.stats_group = QWidget()
        self.stats_layout = QFormLayout(self.stats_group)
        self.stats_layout.setContentsMargins(0, 30, 0, 0)
        
        self.lbl_vmax = QLabel("0.00 m/s")
        self.lbl_vmax.setStyleSheet("font-family: monospace; font-weight: bold; color: #ff5555;")
        self.stats_layout.addRow(QLabel("Max Flow Velocity:"), self.lbl_vmax)
        
        self.lbl_press = QLabel("0.00 Pa")
        self.lbl_press.setStyleSheet("font-family: monospace; font-weight: bold; color: #55aaff;")
        self.stats_layout.addRow(QLabel("Max Dynamic Pressure:"), self.lbl_press)
        
        self.lbl_lift = QLabel("0.00 N")
        self.lbl_lift.setStyleSheet("font-family: monospace; font-weight: bold; color: #55ff55;")
        self.stats_layout.addRow(QLabel("Lift Force (L):"), self.lbl_lift)

        self.lbl_drag = QLabel("0.00 N")
        self.lbl_drag.setStyleSheet("font-family: monospace; font-weight: bold; color: #ffaa00;")
        self.stats_layout.addRow(QLabel("Drag Force (D):"), self.lbl_drag)
        
        self.lbl_weight = QLabel("0.00 N")
        self.lbl_weight.setStyleSheet("font-family: monospace; font-weight: bold; color: #ffffff;")
        self.stats_layout.addRow(QLabel("Req. Weight (W=L):"), self.lbl_weight)

        self.lbl_thrust = QLabel("0.00 N")
        self.lbl_thrust.setStyleSheet("font-family: monospace; font-weight: bold; color: #ff7700;")
        self.stats_layout.addRow(QLabel("Req. Thrust (T=D):"), self.lbl_thrust)
        
        control_layout.addRow(self.stats_group)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.fluid_solver: Any = None

    def update_aoa_label(self, value):
        self.aoa_label.setText(f"Angle of Attack: {value}°")

    def update_speed_label(self, value):
        self.speed_label.setText(f"Headwind Speed: {value} m/s")

    def update_aircraft_label(self, value):
        self.aircraft_label.setText(f"Aircraft Speed: {value} m/s")

    def run_simulation(self):
        self.timer.stop()
        
        naca = self.naca_input.text()
        aoa = self.aoa_slider.value()
        wind_speed = self.speed_slider.value()
        air_speed = self.aircraft_slider.value()
        speed = wind_speed + air_speed

        try:
            airfoil_mesh = generate_naca_airfoil(naca, aoa)
        except Exception as e:
            print(f"Error: {e}")
            return
        
        # Aerodynamics Math
        v_max = speed * 1.5 
        rho = 1.225
        p_dyn = 0.5 * rho * (v_max**2 - speed**2)
        
        aoa_rad = np.radians(aoa)
        
        camber = int(naca[0]) / 100.0 if naca.isdigit() else 0
        cl_0 = 2 * np.pi * camber * 2 
        cl = cl_0 + 2 * np.pi * aoa_rad
        
        cd_0 = 0.015
        AR = 6.0 
        e = 0.8  
        # Virtual wind tunnels act somewhat like 2D infinite wings (less induced drag)
        cd = cd_0 + (cl**2) / (np.pi * e * AR * 3.0)
        S = 1.0  
        
        lift = 0.5 * rho * (speed**2) * S * cl
        drag = 0.5 * rho * (speed**2) * S * cd
        
        weight = np.abs(lift) 
        thrust = np.abs(drag) 
        
        self.lbl_vmax.setText(f"{v_max:.2f} m/s")
        self.lbl_press.setText(f"{p_dyn:.2f} Pa")
        self.lbl_lift.setText(f"{lift:.2f} N")
        self.lbl_drag.setText(f"{drag:.2f} N")
        self.lbl_weight.setText(f"{weight:.2f} N")
        self.lbl_thrust.setText(f"{thrust:.2f} N")
        
        self.fluid_solver = FluidSolver(airfoil_mesh)
        
        self.plotter.clear()
        self.plotter.add_mesh(airfoil_mesh, color="white", metallic=0.7, roughness=0.3)
        v_inlet = speed / 30.0
        self.max_p = 0.5 * 1.225 * (v_inlet**2)
        self.fluid_solver.vis_grid['Pressure'] = np.zeros(self.fluid_solver.vis_grid.n_points)
        self.plotter.add_mesh(self.fluid_solver.vis_grid, scalars='Pressure', cmap='jet', opacity=0.85, clim=[-self.max_p*2, self.max_p], show_scalar_bar=False)

        self.plotter.view_xy()
        self.plotter.camera.position = (0.5, 0, 5)
        self.plotter.camera.focal_point = (0.5, 0, 0)
        
        self.timer.start(30)

    def update_simulation(self):
        if self.fluid_solver is None:
            return
            
        wind_speed = self.speed_slider.value()
        air_speed = self.aircraft_slider.value()
        speed = wind_speed + air_speed
        
        self.fluid_solver.step(speed)
        
        speed_mag = np.sqrt(self.fluid_solver.u**2 + self.fluid_solver.v**2) # type: ignore
        v_inlet = speed / 30.0
        P_field = 0.5 * 1.225 * (v_inlet**2 - speed_mag**2) # type: ignore
        
        self.fluid_solver.vis_grid.point_data['Pressure'] = P_field.ravel()
        self.fluid_solver.vis_grid.set_active_scalars('Pressure')
        
        # Real-time wiggling streamlines computed from dynamic Navier-Stokes velocity array
        self.fluid_solver.vis_grid.point_data['Velocity'] = self.fluid_solver.final_vel
        self.fluid_solver.vis_grid.set_active_vectors('Velocity')
        
        num_lines = 80
        seed_y = np.linspace(-1.3, 1.3, num_lines)
        seed_points = np.zeros((num_lines, 3))
        seed_points[:, 0] = self.fluid_solver.x_min + 0.1
        seed_points[:, 1] = seed_y
        seed_points[:, 2] = 0.0
        seed = pv.PolyData(seed_points)
        
        try:
            streams = self.fluid_solver.vis_grid.streamlines_from_source(
                seed, vectors='Velocity', integration_direction='forward', max_steps=800
            )
            if hasattr(self, 'stream_actor'):
                self.plotter.remove_actor(self.stream_actor)
            if streams.n_points > 0:
                self.stream_actor = self.plotter.add_mesh(streams.tube(radius=0.003), color='white', lighting=False)
        except Exception:
            pass
            
        self.plotter.render()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = WindTunnelApp()
    window.show()
    QTimer.singleShot(500, window.run_simulation)
    sys.exit(app.exec_())
