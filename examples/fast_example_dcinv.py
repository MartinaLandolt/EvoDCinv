# -*- coding: utf-8 -*-

"""
Author: Keurfon Luu <keurfon.luu@mines-paristech.fr>
License: MIT

Script converging faster due to tighter constraints and smaller population.
Main purpose is integration testing.
"""

import numpy as np
import os, sys, time
from argparse import ArgumentParser
from pdb import set_trace as bp
from copy import deepcopy
import sys

#sys.path.insert(5,'/home_c/LANDOLT/evodcinv_old_merge_uncertainties_and_groupV/evodcinv')
#sys.path.remove('/home_c/LANDOLT/evodcinv_old_version/src/evodcinv')

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


if __name__ == "__main__":
    # Initialize MPI
    if mpi_exist:
        mpi_comm = MPI.COMM_WORLD
        mpi_rank = mpi_comm.Get_rank()
    else:
        mpi_rank = 0

    # Parse arguments
    parser = ArgumentParser()
    parser.add_argument("-n", "--num_threads", type=int, default=8)
    parser.add_argument("dtype", type=str, default="phase")
    parser.add_argument("output_name", help="Enter output file name",
                        type=str, default="output")
    args = parser.parse_args()
    dtype = args.dtype
    output_name = parser.parse_args().output_name

    # Parameters
    ny = 200  # Number of velocity discretization points
    max_run = 10  # Number of runs
    outdir = "output/" + output_name  # Output directory

    # Inversion boundaries
    beta = np.array([[200., 800.], [600., 1500.], [200., 800.]])
    thickness = np.array([[300., 800.], [100., 500.], [99999., 99999.]])

    # Initialize dispersion curves
    # disp_param = [("data/groupV_model/group_rayleigh_mode0.txt", "rayleigh", 0)] # group dispersion curve
    disp_param = [("data/rayleigh_mode0.txt", "rayleigh", 0)] # phase dispersion curve

    dcurves = []
    for param in disp_param:
        filename, wtype, mode = param
        faxis, disp, uncertainties = np.loadtxt(filename, unpack=True)
        if 'group' in filename:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='group')
        else:
            dc = DispersionCurve(disp, faxis, mode, uncertainties, wtype, dtype='phase')
        dc.dtype = dtype
        dcurves.append(dc)

    # Evolutionary optimizer parameters
    evo_kws = dict(popsize=10, max_iter=100, constrain=True, mpi=mpi_exist, random_state=1)
    opt_kws = dict(solver="cpso")

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
        misfits = [m.misfit for m in models]
        print(models[np.argmin(misfits)])
        print("Elapsed time: %.2f seconds\n" % (time.time() - starttime))

        best_model = models[np.argmin(misfits)]
        best_model.save("%s/best_model.pickle" % outdir)
        vel = best_model.params2lay()
        th = ThomsonHaskell(vel, dcurves[0].wtype)
        th.propagate(dcurves[0].faxis, ny=ny, domain="fc")
        inverted_dc = th.pick([dcurves[0].mode])[0]
        inverted_dc.dtype = dcurves[0].dtype
        inverted_dc.save("%s/best_model_%s_dc.txt" % (outdir, dtype))
