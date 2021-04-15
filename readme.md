# Poisson Demo

Solves a 3D Poisson problem using Firedrake with a variety of solver options.

You will require a working [Firedrake](https://www.firedrakeproject.org) install to run this code.

The simulation code is in `poisson_gmg.py` and the built in help for command line parameters can be seen by running `python poisson_gmg.py --help`.

Example usage:

```bash
mpiexec -n 8 python poisson_gmg.py \
    --resultsdir results/poisson_patch \
    --baseN 12 \
    --nref 3 \
    --solver_params "MG F-cycle PatchPC" \
    --telescope_factor 1
```

`baseN` corresponds to the size of the coarsest grid in the multigrid solver and `nref` corresponds to the number of multigrid refinements (multigrid levels minus 1).

An example submission script for the Isambard UK tier 2 HPC facility and strong scaling submission script is provided also.
