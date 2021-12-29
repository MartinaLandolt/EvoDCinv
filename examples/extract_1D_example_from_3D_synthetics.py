import h5py
from pathlib import Path
import numpy as np

src_dir = Path(__file__).parent.resolve()
upper1_dir = src_dir.parent.resolve()


def export_1D(file_in, folder_out, i_cell, j_cell):
    # read hdf5 file
    # select line (ordinal)
    # export to data/cell_i_wavetype_data (with true model)
    with h5py.File(file_in, "r") as fin:
        freq = fin["Frequency"][:]
        dcurve = fin["DispersionCurve"][i_cell, j_cell, :]
        uncert = fin["Uncertainties"][i_cell, j_cell, :]
        true_model = fin["true_model"][i_cell, j_cell, :]
        file_name = fin.attrs["velocity_mode"] + '_' + fin.attrs["wavetype"] + '_mode' + \
                    str(file_in).split('.')[-2][-1] + '.txt'
    file_path_out_disp_curve = folder_out.joinpath(file_name)
    file_path_out_model = folder_out.joinpath(
        'true_model_cell' + '.txt')

    # + '_i' + str(i_cell) + '_j' + str(j_cell) +

    true_model_clean = true_model[np.where(~np.isnan(true_model[:, 1]))[0], :]
    np.savetxt(str(file_path_out_model), true_model_clean)

    X = np.stack((freq, dcurve, uncert), axis=1)
    np.savetxt(str(file_path_out_disp_curve), X, "%.8f")
    pass


if __name__ == '__main__':

    file_in = upper1_dir.joinpath('make_synthetics').joinpath('synthetics_chemery'). \
        joinpath('disp_curves_rayleigh_group_mode_0.h5')
    folder_out = src_dir.joinpath('data_custom')
    i_cell = 5
    j_cell = 5
    if not file_in.exists():
        raise Exception("".join(['file_in does not exist : ', str(file_in)]))
    if not folder_out.exists():
        folder_out.mkdir()
        # raise Exception("".join(['folder_out does not exist : ', str(folder_out)]))
    print(file_in)
    print(folder_out)

    export_1D(file_in, folder_out, i_cell, j_cell)
