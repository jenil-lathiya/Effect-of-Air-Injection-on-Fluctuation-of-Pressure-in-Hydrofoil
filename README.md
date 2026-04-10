Overview

This project presents a numerical investigation of pressure fluctuations in a 3D hydrofoil under varying air injection rates using Computational Fluid Dynamics (CFD).
The study focuses on how controlled air injection influences:
- Pressure distribution
- Flow stability
- Hydrodynamic forces (lift & drag)
- Vortex structures
The simulations were performed using OpenFOAM with a custom multiphase solver.

Objectives:
Analyze the effect of air injection on pressure fluctuations
Study surface pressure distribution under different injection rates
Evaluate lift and drag coefficients
Investigate flow structures (velocity, vorticity, wake behavior)
Identify optimal injection rate for flow stabilization

Numerical Approach:
Finite Volume Method (FVM)
Transient 3D simulation
Multiphase flow modeling using Volume of Fluid (VOF)
Turbulence model: k–ω SST

Governing Equations:
Continuity Equation
Momentum Equation
Volume of Fluid (VOF) Equation

These equations were solved using a custom OpenFOAM solver for three-phase flow (air–water–vapor).

Tools & Technologies:
OpenFOAM – CFD simulation
ParaView – Visualization
Python (NumPy, Matplotlib, SciPy) – Data analysis

Simulation Setup:
Geometry: Clark-Y Hydrofoil
Flow Type: Transient, incompressible, multiphase
Angle of Attack: ~8°
Inlet Velocity: 10.45 m/s
Injection Location: Near leading edge (x/c ≈ 0.04)
Mesh: ~500,000 cells with local refinement

Key Results
-Pressure Fluctuations
   Moderate injection rates (0.2 – 0.4 L/min) reduce pressure fluctuations
   Optimal performance observed at 0.3 L/min
   High injection rates increase instability
    According to the analysis, RMS pressure values were lowest at moderate injection rates
-Flow Behavior
    Air injection modifies vortex structures
    Stabilizes boundary layer at moderate rates
    High injection leads to flow disturbance
-Hydrodynamic Performance
    Lift coefficient remains nearly unchanged
    Drag and moment show minor variations
    Flow stability improves at optimal injection
simpleFoam / custom solver
Visualize results in ParaView
