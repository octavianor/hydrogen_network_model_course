# Hydrogen Network Model Course

Course materials for exploring steady-state gas-network hydraulics and a
simplified transient linepack model. The examples use illustrative network
data and gas properties; they are intended for learning, not for operational
design or prediction of a real hydrogen network.

## Contents

- `notebooks/class_work_1.ipynb` - network topology, mass conservation,
  squared-pressure pipe losses, and steady-state scenarios.
- `notebooks/class_work_2.ipynb` - a step change in demand, linepack, and
  spatial discretization.
- `notebooks/pandapipes.ipynb` - optional independent network-model lab using
  pandapipes.
- `src/network_topology.py` - shared example network data and a network plot.
- `src/steady_state_solver.py` - nonlinear steady-state flow and pressure
  solver.
- `src/transient_solver.py` - lumped, compressible-storage pipeline example.
- `Hydrogen Transport.pdf` - accompanying course reference material.

## Setup

Python 3.9 or later is recommended. From the repository directory, create and
activate a virtual environment, then install the listed dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, run the commands below using
`.venv\Scripts\python.exe` instead of `python`.

## Run the course materials

Start JupyterLab:

```powershell
jupyter lab
```

Open the notebooks in `notebooks/` and run their cells in order. The Python
examples can also be run directly from the repository root:

```powershell
python src\network_topology.py
python src\steady_state_solver.py
python src\transient_solver.py
```

## Model scope

The steady-state solver uses an isothermal squared-pressure Darcy-Weisbach
relationship with constant compressibility factor and prescribed friction
factor. The transient example is a spatially lumped RC/compartment model with
compressible storage; it is not a full transient momentum-equation model.
Values such as pipe dimensions, demands, and gas properties are illustrative.
The optional pandapipes notebook has its own dependency and assumptions; install
it separately with `python -m pip install pandapipes` if you want to run it.
