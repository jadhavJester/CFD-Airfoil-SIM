# Virtual Wind Tunnel

A desktop wind tunnel demo that renders a NACA 4-digit airfoil, visualizes a lightweight 2D flow field, and shows basic lift/drag estimates in real time. Includes a separate browser-based 2D CFD toy in `web-windy/`.

## Features
- PyQt5 + PyVista 3D view of an extruded NACA airfoil
- Real-time flow visualization with dynamic streamlines
- Interactive controls for angle of attack and wind/aircraft speed
- On-screen estimates for lift, drag, and dynamic pressure
- Optional web demo for mouse-drawn obstacles and 2D flow

## Requirements
- Python 3.9+
- Windows, macOS, or Linux with OpenGL support

Python packages are listed in `requirements.txt`:
- numpy, scipy
- pyvista, pyvistaqt
- PyQt5

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the desktop app
```bash
python main.py
```

## Run the web demo
Open `web-windy/index.html` in a browser.

## Notes
- The flow solver is a stylized 2D visualization, not a validated CFD solver.
- The lift/drag math is a simplified estimate for interactive feedback.

## Project layout
```
virtual-wind-tunnel/
  main.py
  geometry.py
  solver.py
  requirements.txt
  style.css
  web-windy/
    index.html
    style.css
    fluid.js
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
