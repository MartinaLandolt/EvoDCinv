import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms


from check_dc import get_data

input_file = "./data/rayleigh_cropped_mode0.txt"
dc = get_data(input_file, dtype="phase", conversion=False)

def stable_derivative(curve):
    """
    Computes derivative of a curve (aka data with curve.faxis as x-axis)

    Computes polynomial fit.
    Derivative is then computed analytically.
    """
    df = curve.faxis[1] - curve.faxis[0]
    tmp_axis = np.tile(curve.faxis, (2,1))
    tmp_axis[1,:] += df / 2
    tmp_axis = np.sort(tmp_axis.flatten())

    phase_velocity = np.interp(tmp_axis, curve.faxis, curve.phase_velocity)
    dphase_domega = np.gradient(phase_velocity, df/2) 
    return dphase_domega[::2]


interp_derivative = stable_derivative(dc)
df = dc.faxis[1] - dc.faxis[0]
raw_derivative = np.gradient(dc.phase_velocity, df)

rms = np.sqrt(np.sum((raw_derivative-interp_derivative)**2) / len(interp_derivative))
print(rms)
"""
plt.plot(raw_derivative, interp_derivative, 'o', markersize=2)

ax  = plt.gca()
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)

plt.show()
"""

plt.plot(dc.faxis, interp_derivative, label="Interpolated")
plt.plot(dc.faxis, raw_derivative, label="Raw")
plt.legend()
plt.show()
