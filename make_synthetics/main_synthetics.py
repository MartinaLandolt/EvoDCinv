import make_synthetics
from make_synthetics import settings_synthetics
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from evodcinv import LayeredModel, ThomsonHaskell, params2lay
import pyproj
from pyproj import CRS
from pyproj import Transformer


def remove_first_digit(x):
    """in order to remove the front "2" digit in Lambert II Y coordinate"""
    str_x = str(x)
    str_x_new = str_x[1:]
    x_out = float(str_x_new)
    return x_out


def convert_coordinates(x_in, y_in, espg_in=27562, espg_out=3043):
    """
    27562 is Lambert 2 (central France)
    3043 is UTM 31 N (Europe in general)
    """
    crs1 = CRS.from_epsg(espg_out)
    crs2 = CRS.from_epsg(espg_in)
    transformer = Transformer.from_crs(crs2, crs1)
    x_out, y_out = transformer.transform(x_in, y_in)
    return x_out, y_out


def process_df_coordinates(df_in):
    """ convert df_in fields X and Y to UTM 31N, from Lambert 2 as in most Storengy files """

    df_out = df_in.copy()

    # remove the front "2" digit in the Lambert II Y coordinate (Storengy specificity)
    y = df_out['Y'].values
    x = df_out['X'].values
    y_clean = y
    for (i, y_i) in enumerate(y):
        y_clean[i] = remove_first_digit(y_i)

    # convert to UTM
    x_UTM, y_UTM = convert_coordinates(x, y_clean, espg_in=27562, espg_out=3043)
    df_out['X'] = x_UTM
    df_out['Y'] = y_UTM
    return df_out


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
        df_interface_i = process_df_coordinates(df_interface_i)
        vals_i= df_interface_i.iloc[:, -1].values
        df_interface_i.iloc[:, -1].values[vals_i == -9999.] = np.nan
        interface_list.append(df_interface_i)
        interface_order.append(i)

    # read all velocities
    for (i, velocity_file_i) in enumerate(velocity_file_list):
        df_velocity_i = pd.read_csv(path_model_vel.joinpath(velocity_file_i))
        df_velocity_i = process_df_coordinates(df_velocity_i)
        vals_i= df_velocity_i.iloc[:, -1].values
        df_velocity_i.iloc[:, -1].values[vals_i == -9999.] = np.nan
        velocity_list.append(df_velocity_i)

    # order interfaces by increasing depth
    interface_list = reorder_interfaces_by_depth(interface_list, interface_order)

    # successively add depths as columns to a mega dataframe, only keeping common coordinate points across levels
    for (i, interface_i) in enumerate(interface_list):
        if i==0:
            df_interfaces_global = interface_i
        else:
            df_interfaces_global = pd.merge(df_interfaces_global, interface_i, how='inner', on=['X', 'Y'])

    for (i, velocity_layer_i) in enumerate(velocity_list):
        if i==0:
            df_velocity_global = velocity_layer_i
        else:
            df_velocity_global = pd.merge(df_velocity_global, velocity_layer_i, how='inner', on=['X', 'Y'])

    columns_velocity = list(df_velocity_global)
    columns_interfaces = list(df_interfaces_global)

    df_thickness_global = df_interfaces_global[['X', 'Y']].copy()
    for (i, col_vel) in enumerate(columns_velocity[2:]):
        j=i+2
        z_i_minus_1 = df_interfaces_global.iloc[:,j].values
        z_i = df_interfaces_global.iloc[:, j+1].values
        col_thick = col_vel.replace('Vint', 'Thickness')
        df_thickness_global[col_thick] = z_i - z_i_minus_1

    # keep only common (X,Y) between thickness and velocity
    columns_thickness = list(df_thickness_global)
    df_thickness_global_merge = pd.merge(df_velocity_global, df_thickness_global, how='inner', on=['X', 'Y'])
    df_interfaces_global_merge = pd.merge(df_velocity_global, df_interfaces_global, how='inner', on=['X', 'Y'])
    df_thickness_global = df_thickness_global_merge[columns_thickness]
    df_interfaces_global = df_interfaces_global_merge[columns_interfaces]

    return df_velocity_global, df_thickness_global, df_interfaces_global


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


def check_velocity_column_name_consistency():
    pass


def reorder_interfaces_by_depth(interface_list, interface_order):
    nb_interfaces = len(interface_list)
    flag_order_correct = False
    counter = -1
    while (not flag_order_correct) & (counter < nb_interfaces ** 2):
        counter = counter + 1
        print('number of layer reoderings: ', counter)
        flag_order_correct = True
        interface_list = [interface_list[j] for j in interface_order]
        interface_i_minus_1 = interface_list[0]
        vals_i_minus1 = interface_i_minus_1.iloc[:, -1].values
        for (i, interface_i) in enumerate(interface_list):
            if i > 1:
                vals_i = interface_i.iloc[:, -1].values
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

    if counter == nb_interfaces ** 2:
        raise Exception("something went wrong : the interface ordering counter reached max limit")

    return interface_list


def interpolate_model_per_layer(df_in, xmin=None, xmax=None, ymin=None, ymax=None, dx=None, dy=None):
    """ 2D interpolation of NaN values in each horizon independently.
    The first 2 columns of df_in should be X and Y, followed by a column per horizon"""
    df_out = df_in.copy()
    cols = list(df_out)
    x_all = df_out[['X']].values[:,0]
    y_all = df_out[['Y']].values[:,0]
    for (i, col_i) in enumerate(cols[2:]):
        j=i+2
        vals_j = df_out[col_i].values
        x_good = x_all[~np.isnan(vals_j)]
        y_good = y_all[~np.isnan(vals_j)]
        vals_good = vals_j[~np.isnan(vals_j)]
        vals_all = griddata((x_good, y_good), vals_good, (x_all, y_all))
        df_out[col_i] = vals_all
    return df_out


def make_1d_model_for_cell(thickness_array_in, vp_array_in, last_layer_vel=6000, last_layer_thickness=99999.):
    thickness_array_out = np.hstack((thickness_array_in[thickness_array_in > 10], last_layer_thickness))
    vp_array_out = np.hstack((vp_array_in[thickness_array_in > 10], last_layer_vel))
    if max(vp_array_out) <= 0:
        raise Exception('Encountered non-strictly positive velocity after removing zero-thickness layers')
    return thickness_array_out, vp_array_out


def convert_1d_model_to_th_format(vp_array, thickness_array,
                           vp_over_vs_ratio=2):
    nb_layers = len(vp_array)
    # create model in single line
    m = np.nan*np.zeros(nb_layers*3)
    m[0 : nb_layers] = 1/vp_over_vs_ratio * vp_array
    m[nb_layers : 2*nb_layers] = thickness_array
    m[2*nb_layers : 3*nb_layers] = vp_over_vs_ratio
    # convert to model usable with Thomson-Haskell
    l = params2lay(m)

    return l


def get_dispersion_curve(l, f_axis, ny, modes, wavetype, velocity_mode):
    th = ThomsonHaskell(l, wtype=wavetype)
    th.propagate(f_axis, ny=ny, domain="fc", n_threads=1)
    dc_calculated = th.pick(modes=modes)
    if velocity_mode == "group":
        for dcurve in dc_calculated:
            dcurve.dtype = velocity_mode
    return dc_calculated


def loop_on_cells(df_velocity_global, df_thickness_global, vp_over_vs_ratio,
                  fmin, fmax, nb_f, wavetype, modes, velocity_mode, ny):

    # set common frequency axis
    f = np.linspace(fmin, fmax, nb_f)

    # initialize dict with an array n_cells x n_f for dispersion curves per mode
    dispersion_dict = dict()
    dispersion_dict['wavetype'] = wavetype
    dispersion_dict['velocity_mode'] = velocity_mode
    dispersion_dict['f_axis'] = f
    for mode in modes:
        dispersion_dict["".join(['mode_', str(mode)])] = np.nan * np.zeros((len(df_velocity_global), nb_f))

    cell_count = 0
    for (i, df_velocity_cell), (j, df_thickness_cell) in zip(df_velocity_global.iterrows(),
                                                                  df_thickness_global.iterrows()):
        thickness_array = df_thickness_cell.iloc[2:].values
        velocity_array = df_velocity_cell.iloc[2:].values
        thickness_array_clean, velocity_array_clean = \
            make_1d_model_for_cell(thickness_array, velocity_array)
        l = convert_1d_model_to_th_format(velocity_array_clean, thickness_array_clean,
                                          vp_over_vs_ratio=vp_over_vs_ratio)
        try:
            dc_calculated = get_dispersion_curve(l, f, ny, modes, wavetype, velocity_mode)
        except:
            dc_calculated = get_dispersion_curve(l, f, ny, modes, wavetype, velocity_mode)

        # write in the array
        for mode in modes:
            vel_vals = getattr(dc_calculated[mode], "".join([velocity_mode, "_velocity"]))
            f_vals = getattr(dc_calculated[mode], 'faxis')
            i_f = [i for (i, x) in enumerate(f) if (x in f_vals)]
            try:
                dispersion_dict["".join(['mode_', str(mode)])][cell_count, i_f] = vel_vals
            except:
                print('debug')

        cell_count += 1
        print('cell ', cell_count, ' out of ', len(df_velocity_global))

    # save_h5() # separately for each mode
    return dispersion_dict


def save_h5():
    # create a mesh from the data
    # interpolate the dispersion curves on the mesh at each frequency
    # create the MissingSamples
    # create the uncertainties
    # write to h5 file
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

    # read model
    df_velocity_global, df_thickness_global, df_interfaces_global = \
        read_model_fmt1(path_model_in, n_interfaces)

    # select only points where all the horizons are well defined
    df_interfaces_valid = df_interfaces_global[~df_interfaces_global.isnull().any(axis=1)]
    df_thickness_valid = df_thickness_global[~df_interfaces_global.isnull().any(axis=1)]
    df_velocity_valid = df_velocity_global[~df_interfaces_global.isnull().any(axis=1)]

    # fill points where velocity is lacking
    df_velocity_interp = interpolate_model_per_layer(df_velocity_valid)

    # select only points where all values are well defined
    df_interfaces_valid = df_interfaces_valid[~df_velocity_interp.isnull().any(axis=1)]
    df_thickness_valid = df_thickness_valid[~df_velocity_interp.isnull().any(axis=1)]
    df_velocity_valid = df_velocity_interp[~df_velocity_interp.isnull().any(axis=1)]

    # compute dispersion curves
    nb_f = int(np.ceil((settings_synthetics.f_stop - settings_synthetics.f_start)/settings_synthetics.f_step))+1
    dispersion_dict = loop_on_cells(df_velocity_valid, df_thickness_valid, settings_synthetics.vp_over_vs,
                  settings_synthetics.f_start, settings_synthetics.f_stop, nb_f,
                  settings_synthetics.wavetype, settings_synthetics.modes,
                  settings_synthetics.velocity_mode, settings_synthetics.ny)


