# NDESolver
multi-group neutron diffusion equation solver

There are several benchmark problems for solving multi-group neutron diffusion problems:
- LRA BWR-2D
- IAEA PWR-2D

Usage:
```bash
python main.py --config PWR-2D.toml --refine 4
# config: toml path
# refine: fineness of mesh
```
