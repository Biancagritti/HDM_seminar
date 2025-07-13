# Balanced Truncation Model Order Reduction for Thermal Systems

This project demonstrates the application of **Balanced Truncation (BT)**, a classical model order reduction (MOR) technique, to a linear time-invariant (LTI) thermal system. The system models heat exchange between 6 interconnected rooms with heaters at both ends.

The project also explores the limitations of BT by applying it to a time-varying system where the system matrix is perturbed sinusoidally.

## Requirements

- Python 3.x  
- NumPy  
- SciPy  
- Matplotlib

Install dependencies via pip if needed:

```bash
pip install numpy scipy matplotlib

```
Note: For MacOS it sould be necessary to create a virtual space with the command 
```bash
python3 -m venv env
source env/bin/activate
```
To run it:
```bash
python3 6_rooms_heat_exchange.py

```