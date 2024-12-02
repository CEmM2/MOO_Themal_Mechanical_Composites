# MOO_Themal_Mechanical_Composites
Efficient Multi-Objective Optimization of Composite Microstructures for Thermal Protection Systems


This repository contains the implementation of the surrogate neural network model described in "Surrogate model based Thermo-mechanical optimization of honeycomb inspired composite RVE". The model predicts effective thermo-mechanical properties of periodic Representative Volume Elements (RVEs) based on their geometric parameters.

## Input Parameters

The model accepts 10 geometric parameters that define the RVE architecture:

1. `web` [1.0 - 3.0]: Honeycomb web thickness
2. `iso` [0.25 - 0.75]: Fraction of lightweight phase within the web
3. `theta1` - `theta8` [0° - 180°]: Reinforcement angles for each of the 8 cells

## Output Properties

The model predicts 6 effective properties:

1. `rho` [kg/m³]: Effective density
2. `E11` [GPa]: Elastic modulus in direction 1
3. `E22` [GPa]: Elastic modulus in direction 2
4. `G12` [GPa]: Shear modulus
5. `K1` [W/mK]: Thermal conductivity in direction 1
6. `K2` [W/mK]: Thermal conductivity in direction 2



## Example

```python
from model.model import create_model, run_model

# Create model
model = create_model()

# Example input parameters
inputs = [
    1.5,    # web thickness
    0.5,    # iso fraction
    0.0,    # theta1
    45.0,   # theta2
    90.0,   # theta3
    135.0,  # theta4
    180.0,  # theta5
    135.0,  # theta6
    90.0,   # theta7
    45.0    # theta8
]

# Get predictions
outputs = run_model(model, inputs)
print("Predicted properties:", outputs)
```

## Repository Structure

```
your-repo/
├── model/
│   ├── nn_constants_parsed.json    # Model weights and constants
│   ├── model.py                    # Model implementation
│   └── example_usage.py           # Example usage script (requires [pymoo](https://pymoo.org/index.html) )
├── README.md
└── requirements.txt
```

## Data Availability

## Citation

If you use this model in your research, please cite:

```bibtex
[Citation details will be added upon publication]
```

## License

see the LICENSE file for details.

## Contact

shmuliko@technion.ac.il

