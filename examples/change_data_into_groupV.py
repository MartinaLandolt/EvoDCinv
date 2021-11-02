# -*- coding: utf-8 -*-

import numpy as np
import os

def to_group_velocity(phase_velocity,faxis):
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
    data_dir = 'data'
    filenames = os.listdir("%s/" % data_dir)
    for name in filenames:
        if 'rayleigh' in name or 'love' in name :
            file = np.loadtxt("%s/%s" % (data_dir, name), unpack = True)
            gV = to_group_velocity(file[1], file[0])
            file[1] = gV
            file = np.transpose(file)
            np.savetxt("%s/group_%s" % (data_dir, name), file)

