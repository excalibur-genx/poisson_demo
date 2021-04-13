import argparse
import csv

from firedrake import *
from pyop2.profiling import timed_stage
from pathlib import Path
from pprint import pformat
from time import time

# For debugging:
# ~ from firedrake.petsc import PETSc
# ~ PETSc.Sys.popErrorHandler()


class ResultsCSV(object):

    fields = ['baseN', 'nref', 'degree', 'solver name',
              'error', 'dofs', 'dofs_core', 'runtime']

    def __init__(self, args, comm=COMM_WORLD):
        self.args = args
        self.comm = comm
        results = Path(self.args.resultsdir)
        csvfilename = str(comm.size) + '_results'
        csvfilename += '_baseN' + str(self.args.baseN)
        csvfilename += '_nref' + str(self.args.nref)
        csvfilename += '.csv'
        self.filepath = results/csvfilename
        if self.comm.rank == 0:
            with open(self.filepath, 'w') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=self.fields)
                writer.writeheader()

    def record_result(self, name, error, runtime, dofs):
        singleresult = {}
        singleresult['baseN'] = self.args.baseN
        singleresult['nref'] = self.args.nref
        singleresult['degree'] = self.args.degree
        singleresult['solver name'] = name
        singleresult['error'] = error
        singleresult['dofs'] = dofs
        singleresult['dofs_core'] = dofs/self.comm.size
        singleresult['runtime'] = runtime
        if self.comm.rank == 0:
            with open(self.filepath, 'a') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=self.fields)
                writer.writerow(singleresult)
                csvf.flush()


parser = argparse.ArgumentParser()
parser.add_argument('--baseN',
                    type=int,
                    default=8,
                    help='base mesh size')
parser.add_argument('--nref',
                    type=int,
                    default=3,
                    help='number of mesh refinements')
parser.add_argument('--degree',
                    type=int,
                    default=3,
                    help='degree of CG element')
parser.add_argument('--resultsdir',
                    type=str,
                    default='results',
                    help='directory to save results in')
parser.add_argument('--solver_params',
                    type=str,
                    default='MG F-cycle PatchPC',
                    choices=[
                        'LU',
                        'LU MUMPS',
                        'LU SuperLU_dist',
                        'MG V-cycle',
                        'MG F-cycle',
                        'MG F-cycle telescope',
                        'MG F-cycle LU coarse MUMPS',
                        'MG F-cycle LU coarse SuperLU_dist',
                        'MG F-cycle Cholesky coarse MUMPS',
                        'MG F-cycle Cholesky coarse SuperLU_dist',
                        'MG F-cycle PatchPC',
                        'MG F-cycle PatchPC telescope',
                        'MG F-cycle PatchPC telescope SuperLU_dist',
                        'MG F-cycle ASMStarPC',
                        'MG F-cycle ASMStarPC TinyASM'
                    ],
                    help='name of dict to take solver parameters from')
parser.add_argument('--telescope_factor',
                    type=int,
                    default=1,
                    help='Telescope factor for telescoping solvers (set to number of NODES)')
args, unknown = parser.parse_known_args()

results = Path(args.resultsdir)
if (not results.is_dir()) and (COMM_WORLD.rank == 0):
    try:
        results.mkdir()
    except FileExistsError:
        print('File', cwd,
              'already exists, cannot create directory with the same name')

csvfile = ResultsCSV(args)

with timed_stage('Mesh_hierarchy'):
    distribution_parameters = {
        "partition": True,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
    }
    mesh = BoxMesh(args.baseN, args.baseN, args.baseN, 1, 1, 1,
                   distribution_parameters=distribution_parameters)
    hierarchy = MeshHierarchy(mesh, args.nref)
    mesh = hierarchy[-1]

with timed_stage('Problem_setup'):
    V = FunctionSpace(mesh, "CG", args.degree)
    dofs = V.dim()
    print('DOFs', dofs)
    if COMM_WORLD.rank == 0:
        with open(results/f'{COMM_WORLD.size}_DOFS.txt', 'w') as fh:
            fh.write(str(dofs)+'\n')

    u = TrialFunction(V)
    v = TestFunction(V)

    bcs = DirichletBC(V, zero(), (1, 2, 3, 4, 5, 6))

    x, y, z = SpatialCoordinate(mesh)

    simplerhs = False
    if simplerhs:
        # Very simple RHS
        k = [Constant(1.0), Constant(1.0), Constant(1.0)]
        exact = sin(k[0]*pi*x)*sin(k[1]*pi*y)*sin(k[2]*pi*z)
        f = ((k[0]**2 + k[1]**2 + k[2]**2)*(pi**2))*exact
    else:
        # Less simple RHS
        a = Constant(1)
        b = Constant(2)
        exact = sin(pi*x)*tan(pi*x/4)*sin(a*pi*y)*sin(b*pi*z)
        f = -pi**2 / 2
        f *= 2*cos(pi*x) - cos(pi*x/2) - 2*(a**2 + b**2)*sin(pi*x)*tan(pi*x/4)
        f *= sin(a*pi*y)*sin(b*pi*z)

    a = dot(grad(u), grad(v))*dx
    L = f*v*dx


def run_solve(parameters):
    if COMM_WORLD.rank == 0:
        with open(results/f'{COMM_WORLD.size}_SOLVER_OPTS.txt', 'w') as fh:
            fh.write(pformat(parameters)+'\n')
    u = Function(V)
    solve(a == L, u, bcs=bcs, solver_parameters=parameters)
    return u


def error(u):
    expect = Function(V).interpolate(exact)
    return norm(assemble(u - expect))


# All the different solver dictionaries we want to try
lu = {"snes_view": None, "ksp_type": "preonly", "pc_type": "lu"}

lu_mumps = {"snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"}

lu_slud = {"snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "superlu_dist"}

vmg = {"snes_view": None, "ksp_type": "cg", "pc_type": "mg", "pc_mg_log": None}

fmg = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi"
}

fmg_telescope = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled": {
        "mat_type": "aij",
        "pc_type": "telescope",
        "pc_telescope_reduction_factor": args.telescope_factor,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": "superlu_dist"
    }
}

fmg_lu_coarse_mumps = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

fmg_cholesky_coarse_mumps = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "cholesky",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

fmg_lu_coarse_slud = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "superlu_dist"
}

fmg_cholesky_coarse_slud = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels_ksp_type": "chebyshev",
    "mg_levels_ksp_max_it": 2,
    "mg_levels_pc_type": "jacobi",
    "mg_coarse_pc_type": "cholesky",
    "mg_coarse_pc_factor_mat_solver_type": "superlu_dist"
}

fmg_patch = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "ksp_norm_type": "unpreconditioned",
        "ksp_convergence_test": "skip",
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_construct_type": "star",
        "patch_pc_patch_construct_dim": 0,
        "patch_pc_patch_dense_inverse": True,
        "patch_pc_patch_partition_of_unity": False,
        "patch_pc_patch_precompute_element_tensors": True,
        "patch_pc_patch_save_operators": True
    },
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

fmg_patch_telescope = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "ksp_norm_type": "unpreconditioned",
        "ksp_convergence_test": "skip",
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_construct_type": "star",
        "patch_pc_patch_construct_dim": 0,
        "patch_pc_patch_dense_inverse": True,
        "patch_pc_patch_partition_of_unity": False,
        "patch_pc_patch_precompute_element_tensors": True,
        "patch_pc_patch_save_operators": True
    },
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled": {
        "mat_type": "aij",
        "pc_type": "telescope",
        "pc_telescope_reduction_factor": args.telescope_factor,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": "mumps"
    }
}

fmg_patch_telescope_slud = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "ksp_norm_type": "unpreconditioned",
        "ksp_convergence_test": "skip",
        "pc_type": "python",
        "pc_python_type": "firedrake.PatchPC",
        "patch_pc_patch_construct_type": "star",
        "patch_pc_patch_construct_dim": 0,
        "patch_pc_patch_dense_inverse": True,
        "patch_pc_patch_partition_of_unity": False,
        "patch_pc_patch_precompute_element_tensors": True,
        "patch_pc_patch_save_operators": True
    },
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled": {
        "mat_type": "aij",
        "pc_type": "telescope",
        "pc_telescope_reduction_factor": args.telescope_factor,
        "pc_telescope_subcomm_type": "contiguous",
        "telescope_pc_type": "lu",
        "telescope_pc_factor_mat_solver_type": "superlu_dist"
    }
}

fmg_asmstar = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "ksp_norm_type": "unpreconditioned",
        "ksp_convergence_test": "skip",
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_construct_dim": 0,
        "pc_star_sub_ksp_type": "preonly",
        "pc_star_sub_pc_type": "lu",
        "pc_star_sub_pc_factor_mat_solver_type": "mumps"
    },
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

fmg_tinyasmstar = {
    "snes_view": None,
    "ksp_type": "preonly",
    "pc_type": "mg",
    "pc_mg_log": None,
    "pc_mg_type": "full",
    "mg_levels": {
        "ksp_type": "chebyshev",
        "ksp_max_it": 2,
        "ksp_norm_type": "unpreconditioned",
        "ksp_convergence_test": "skip",
        "pc_type": "python",
        "pc_python_type": "firedrake.ASMStarPC",
        "pc_star_backend": "tinyasm",
        "pc_star_construct_dim": 0,
    },
    "mg_coarse_pc_type": "lu",
    "mg_coarse_pc_factor_mat_solver_type": "mumps"
}

# Create a dictionary of these parameters
param = {
    'LU': lu,
    'LU MUMPS': lu_mumps,
    'LU SuperLU_dist': lu_slud,
    'MG V-cycle': vmg,
    'MG F-cycle': fmg,
    'MG F-cycle telescope': fmg_telescope,
    'MG F-cycle LU coarse MUMPS': fmg_lu_coarse_mumps,
    'MG F-cycle LU coarse SuperLU_dist': fmg_lu_coarse_slud,
    'MG F-cycle Cholesky coarse MUMPS': fmg_cholesky_coarse_mumps,
    'MG F-cycle Cholesky coarse SuperLU_dist': fmg_cholesky_coarse_slud,
    'MG F-cycle PatchPC': fmg_patch,
    'MG F-cycle PatchPC telescope': fmg_patch_telescope,
    'MG F-cycle PatchPC telescope SuperLU_dist': fmg_patch_telescope_slud,
    'MG F-cycle ASMStarPC': fmg_asmstar,
    'MG F-cycle ASMStarPC TinyASM': fmg_tinyasmstar
}

runlist = [args.solver_params]
for key in runlist:
    with timed_stage(key):
        t = time()
        u = run_solve(param[key])
        t = time() - t

    recerror = error(u)
    csvfile.record_result(key, recerror, t, dofs)
    print(key, 'error', recerror, 'in', t, 's')
