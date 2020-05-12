""" 
A script checking if the inverted dispersion curve is fitting the 
input synthetic dispersion curve.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np

import argparse

from evodcinv import DispersionCurve

def get_data(filename, dtype, conversion=True):
         freq, vel = [], []
         with open(filename, 'r') as csvfile:
             reader = csv.reader(csvfile, delimiter=' ')
             for line in reader:
                 floats = [float(x) for x in line]
                 freq.append(floats[0])
                 vel.append(floats[1])
         if conversion:
             dc = DispersionCurve(vel, freq, mode=0, wtype="rayleigh", dtype="phase")
             dc.dtype = dtype
         else:
             dc = DispersionCurve(vel, freq, mode=0, wtype="rayleigh", dtype=dtype)
         return dc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dtype", help="Choose between phase velocity (default) or group velocity",
            type=str, default="phase")
    dtype = parser.parse_args().dtype


    ref_file = "data/rayleigh_mode0.txt"
    res_file = f"output/output_{dtype}_dc.txt"

    ref_dc = get_data(ref_file, dtype, conversion=True)
    res_dc = get_data(res_file, dtype, conversion=False)

    plt.plot(ref_dc.faxis, ref_dc.dtype_velocity, marker="o", label="Reference")
    plt.plot(res_dc.faxis, res_dc.dtype_velocity, marker="o", label="Inverted")
    plt.ylabel(f"{dtype} velocity (m/s)")
    plt.xlabel("Frequency (Hz)")
    plt.legend()
    plt.savefig(f"example_{dtype}_dcinv.png")
