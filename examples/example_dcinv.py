# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT
"""

import numpy as np
import os, sys, time
from argparse import ArgumentParser
from copy import deepcopy
try:
    from mpi4py import MPI
    mpi_exist = True
except ImportError:
    mpi_exist = False
try:
    from evodcinv import DispersionCurve, LayeredModel, progress, ThomsonHaskell
except ImportError:
    sys.path.append("../")
    from evodcinv import DispersionCurve, LayeredModel, progress, ThomsonHaskell



def to_group_velocity(phase_velocity, faxis):
    """
    Convert phase velocity to group velocity by
    differentiating phase_velocity

    Only works if frequencies are evenly spaced.
    """

    omega = 2 * np.pi * faxis
    domega = omega[1] - omega[0]
    if not np.allclose(np.diff(omega), domega, rtol=10 ** -2):
        raise ValueError("""Frequencies not evenly spaced. 
               Could not convert from phase velocity to group velocity""")
    dphase_domega = np.gradient(phase_velocity, domega)
    # group_velocity = phase_velocity  + omega * dphase_domega
    group_velocity = phase_velocity / (1 - omega / phase_velocity * dphase_domega)

    return group_velocity


if __name__ == "__main__":
    # Initialize MPI
    if mpi_exist:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_rank = 0
        
    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type = int, default = 8)
    parser.add_argument("dtype", type=str, default="phase")
    args = parser.parse_args()
    dtype = args.dtype
        
    # Parameters
    ny = 200                        # Number of velocity discretization points
    max_run = 10                    # Number of runs
    popsize = 10
    max_iter = 100
    outdir = "output2"               # Output directory
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Inversion boundaries
    beta = np.array([ [ 100., 1000. ], [ 500., 2000. ], [ 500., 2000. ] ])
    thickness = np.array([ [ 100., 1000. ], [ 100., 500. ], [ 99999., 99999. ] ])
    
    # Initialize dispersion curves
    disp_param = [
        ( "data/rayleigh_mode0.txt", "rayleigh", 0 ),
        ( "data/rayleigh_mode1.txt", "rayleigh", 1 ),
        ( "data/love_mode0.txt", "love", 0 ),
        ( "data/love_mode1.txt", "love", 1 ),
        ]
    
    dcurves = []
    for param in disp_param:
        filename, wtype, mode = param
        faxis, disp, uncertainties = np.loadtxt(filename, unpack = True)
        if dtype == 'group':
            disp = to_group_velocity(disp, faxis)
        dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype=dtype)
        dc.dtype = dtype
        dcurves.append(dc)

    # Evolutionary optimizer parameters
    evo_kws = dict(popsize = popsize, max_iter = max_iter, constrain = True, mpi = mpi_exist)
    opt_kws = dict(solver = "cpso")
        
    # Multiple inversions
    if mpi_rank == 0:
        starttime = time.time()
        os.makedirs(outdir, exist_ok=True)
        progress(-1, max_run, "perc", prefix="Inverting dispersion curves: ")

    models = []
    for i in range(max_run):
        lm = LayeredModel()
        lm.invert(dcurves, beta, thickness, ny=ny, n_threads=args.num_threads,
                  evo_kws=evo_kws, opt_kws=opt_kws)
        if mpi_rank == 0:
            lm.save("%s/run%d.pickle" % (outdir, i + 1))
            models.append(deepcopy(lm))
            progress(i, max_run, "perc", prefix="Inverting dispersion curves: ")
        
    if mpi_rank == 0:
        print("\n")
        misfits = [ m.misfit for m in models ]
        print(models[np.argmin(misfits)])
        print("Elapsed time: %.2f seconds\n" % (time.time() - starttime))

        best_model = models[np.argmin(misfits)]
        best_model.save("%s/best_model.pickle" % outdir)
        vel = best_model.params2lay()
        th = ThomsonHaskell(vel, dcurves[0].wtype)
        th.propagate(dcurves[0].faxis, ny=ny, domain="fc")
        inverted_dc = th.pick([dcurves[0].mode])[0]
        inverted_dc.dtype = dcurves[0].dtype
        inverted_dc.save(f"output/best_model_{dtype}_dc.txt")
