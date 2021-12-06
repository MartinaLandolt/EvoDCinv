import make_synthetics
from make_synthetics import settings_synthetics
from pathlib import Path
import glob
import pandas as pd
import numpy as np


def read_model_fmt1(path_model, nb_interfaces):
    """read layers and velocities
    get nx, ny
    save temporary variables with velocities as dataframes
    """
    nb_layers = nb_interfaces - 1
    path_model_depth = Path(path_model).joinpath("Depth")
    path_model_vel = Path(path_model).joinpath("Velocity")

    interface_order = []

    interface_file_list = glob.glob1(path_model_depth, "*.csv")
    velocity_file_list = glob.glob1(path_model_vel, "*.csv")

    interface_list = []
    velocity_list = []
    thickness_list = []

    # read all interfaces
    for (i, interface_file_i) in enumerate(interface_file_list):
        df_interface_i = pd.read_csv(path_model_depth.joinpath(interface_file_i))
        interface_list.append(df_interface_i)
        interface_order.append(i)

    # read all velocities
    for (i, velocity_file_i) in enumerate(velocity_file_list):
        df_velocity_i = pd.read_csv(path_model_vel.joinpath(velocity_file_i))
        velocity_list.append(df_velocity_i)

    # order interfaces by increasing depth
    flag_order_correct = False
    counter = -1
    while (not flag_order_correct) & (counter < nb_interfaces**2) :
        counter = counter + 1
        print('number of layer reoderings: ', counter)
        flag_order_correct = True
        interface_list = [interface_list[j] for j in interface_order]
        interface_i_minus_1 = interface_list[0]
        vals_i_minus1 = interface_i_minus_1.iloc[:, -1].values
        vals_i_minus1[vals_i_minus1==-9999.] = np.nan
        for (i, interface_i) in enumerate(interface_list):
            if i > 1:
                vals_i = interface_i.iloc[:,-1].values
                vals_i[vals_i==-9999.] = np.nan
                diff_z = vals_i - vals_i_minus1
                if np.nanmin(diff_z) < 0:
                    if np.nanmax(diff_z) > 0:
                        raise Exception("".join(["crossing interfaces : ", interface_i.columns.values[-1], " and ",
                                                 interface_i_minus_1.columns.values[-1]]))
                    flag_order_correct = False
                    interface_order[i] = i - 1
                    interface_order[i - 1] = i
                interface_i_minus_1 = interface_i
                vals_i_minus1 = vals_i_minus1

    if counter == nb_interfaces**2:
        raise Exception("something went wrong : the interface ordering counter reached max limit")

    # get layer thickness & velocity
    for (i, interface_i) in enumerate(interface_list):
        vals_i = interface_i.iloc[:, -1].values
        vals_i[vals_i == -9999.] = np.nan
        if i > 1:
            diff_z = vals_i - vals_i_minus1
            velocity_i = velocity_list[i-1]
            thickness_i = velocity_i.copy()
            thickness_i.columns.values[-1].replace('Vint', 'Thickness')
            thickness_i.iloc[:,-1] = diff_z
            thickness_list.append(thickness_i)
        interface_i_minus_1 = interface_i
        vals_i_minus1 = interface_i_minus_1.iloc[:, -1].values
        vals_i_minus1[vals_i_minus1 == -9999.] = np.nan

    return interface_list, thickness_list, velocity_list


def get_interface_number_fmt1(path_model):
    """count number of folders in "Depth" folder """
    path_model_depth = Path(path_model).joinpath("Depth")
    if not path_model_depth.exists():
        raise Exception("".join([path_model_depth, ' not found']))
    path_model_vel = Path(path_model).joinpath("Velocity")
    if not path_model_vel.exists():
        raise Exception("".join([path_model_vel, ' not found']))
    nb_interfaces = len(glob.glob1(path_model_depth, "*.csv"))
    nb_layers = len(glob.glob1(path_model_vel, "*.csv"))
    if nb_interfaces - nb_layers != 1:
        raise Exception("".join(['nb_interfaces - nb_layers should be 1, but found ', nb_interfaces - nb_layers]))
    return nb_interfaces, nb_layers


def make_1d_model_for_cell():
    pass


def get_dispersion_curve():
    pass


def loop_on_cells():
    # make_1d_model_for_cell():

    # get_dispersion_curve()

    # save_h5()
    pass


def save_h5():
    pass


if __name__ == '__main__':
    print('model format : ', settings_synthetics.type_vel_model)
    if settings_synthetics.type_vel_model == 1:
        path_model_in = make_synthetics.data_format_1
        if not Path(path_model_in).exists():
            raise Exception("".join(['Data not found : ', path_model_in]))
        path_synthetics_out = make_synthetics.path_out_format_1
        if not Path(path_synthetics_out).exists():
            Path(path_synthetics_out).mkdir()

    # get number of layers
    if settings_synthetics.n_layers == 'auto':
        n_interfaces, n_layers = get_interface_number_fmt1(path_model_in)
    else:
        raise Exception("User-fixed layer number not yet supported. Number of layers should be auto")

    interface_list, thickness_list, velocity_list = read_model_fmt1(path_model_in, n_interfaces)

    # loop_on_cells


