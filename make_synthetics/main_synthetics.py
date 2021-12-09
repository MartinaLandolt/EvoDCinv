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
import h5py
from scipy.signal import medfilt2d


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
    transformer = Transformer.from_crs(crs2, crs1, always_xy=True)
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
        if i==0:
            dx_in, dy_in = find_grid_step(df_interface_i['X'].values, df_interface_i['Y'].values)
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

    return df_velocity_global, df_thickness_global, df_interfaces_global, dx_in, dy_in


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


def find_grid_step(x_in, y_in):
    dx_in = np.nanmedian(np.diff(np.unique(x_in)))
    dy_in = np.nanmedian(np.diff(np.unique(x_in)))
    return dx_in, dy_in


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


def interpolate_model_per_layer(df_in, xmin=None, xmax=None, ymin=None, ymax=None, n_cells=None,
                               lateral_smooth=False, smooth_length=100, dx_in=None,  dy_in=None):
    """ 2D interpolation of NaN values in each horizon independently.
    The first 2 columns of df_in should be X and Y, followed by a column per horizon"""
    df_out = pd.DataFrame()
    cols = list(df_in)
    x_in = df_in[['X']].values[:,0]
    y_in = df_in[['Y']].values[:,0]

    if lateral_smooth:
        if dx_in != dy_in:
            raise Exception('smoothing not implemented for non-square cells')
        n_smooth_kernel = int(np.ceil(smooth_length/dx_in))
        if (n_smooth_kernel % 2) == 0:
            n_smooth_kernel += 1 # ensure the number is odd (scipy median filter module requirement)
        n_cells_y_in = int(np.ceil( (np.nanmax(y_in) - np.nanmin(y_in)) / dy_in))
        n_cells_x_in = int(np.ceil( (np.nanmax(x_in) - np.nanmin(x_in)) / dx_in))
        x_axis_in = np.linspace(np.nanmin(x_in), np.nanmax(x_in), n_cells_x_in)
        y_axis_in = np.linspace(np.nanmin(y_in), np.nanmax(y_in), n_cells_y_in)
        x_mesh_in, y_mesh_in = np.meshgrid(x_axis_in, y_axis_in)

    x_axis = np.linspace(xmin, xmax, n_cells)
    y_axis = np.linspace(ymin, ymax, n_cells)
    x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)

    x_out = x_mesh.flatten()
    y_out = y_mesh.flatten()
    df_out['X'] = x_out
    df_out['Y'] = y_out

    # for each layer
    for (i, col_i) in enumerate(cols[2:]):
        j=i+2
        vals_j = df_in[col_i].values
        x_good = x_in[~np.isnan(vals_j)]
        y_good = y_in[~np.isnan(vals_j)]
        vals_good = vals_j[~np.isnan(vals_j)]
        # if smoothing necessary
        if lateral_smooth:
            # interpolate on grid_in
            vals_tmp = griddata((x_good, y_good), vals_good, (x_mesh_in, y_mesh_in))
            # smooth 2D
            vals_tmp_smooth = medfilt2d(vals_tmp, n_smooth_kernel)
            # update & flatten x_good, y_good, vals_good
            x_good = x_mesh_in.flatten()
            y_good = y_mesh_in.flatten()
            vals_good = vals_tmp_smooth.flatten()
            x_good = x_good[~np.isnan(vals_good)]
            y_good = y_good[~np.isnan(vals_good)]
            vals_good = vals_good[~np.isnan(vals_good)]
        vals_all = griddata((x_good, y_good), vals_good, (x_out, y_out))
        df_out[col_i] = vals_all
    return df_out


def make_1d_model_for_cell(thickness_array_in, vp_array_in, last_layer_vel=6000, last_layer_thickness=99999.):
    """ ignores layers thinner than 10 m """
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
    dispersion_dict['modes'] = modes
    dispersion_dict['velocity_mode'] = velocity_mode
    dispersion_dict['f_axis'] = f
    dispersion_dict['X'] = df_velocity_global['X']
    dispersion_dict['Y'] = df_velocity_global['Y']
    for mode in modes:
        dispersion_dict["".join(['mode_', str(mode)])] = np.nan * np.zeros((len(df_velocity_global), nb_f))

    cell_count = 0
    for (i, df_velocity_cell), (j, df_thickness_cell) in zip(df_velocity_global.iterrows(),
                                                                  df_thickness_global.iterrows()):
        thickness_array = df_thickness_cell.iloc[2:].values
        velocity_array = df_velocity_cell.iloc[2:].values
        if np.isnan(velocity_array).any():
            for mode in modes:
                dispersion_dict["".join(['mode_', str(mode)])][cell_count, :] = np.nan
        else:
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

    return dispersion_dict


def save_h5(dispersion_dict, file_out):
    # resize coordinates to mesh format
    x_mesh = np.reshape(dispersion_dict['X'].values, (settings_synthetics.n_cells, settings_synthetics.n_cells))
    y_mesh = np.reshape(dispersion_dict['Y'].values, (settings_synthetics.n_cells, settings_synthetics.n_cells))
    dict_h5_list = []
    for mode in dispersion_dict['modes']:
        freq = dispersion_dict['f_axis']
        dispersion_array = dispersion_dict["".join(['mode_', str(mode)])]
        nan_test = np.sum(~np.isnan(dispersion_array), axis=1)
        # create the MissingSamples
        mask = (nan_test < 0.5*len(freq)).astype('int')
        # create the uncertainties
        vel_uncert = dispersion_array * 0.01
        # resize
        disp_curves = np.reshape(dispersion_array,
                                 (settings_synthetics.n_cells, settings_synthetics.n_cells, len(freq)))
        mask = np.reshape(mask, (settings_synthetics.n_cells, settings_synthetics.n_cells,))
        vel_uncert = np.reshape(vel_uncert, (settings_synthetics.n_cells, settings_synthetics.n_cells, len(freq)))
        # write to h5 file
        file_out_mode = "".join([file_out, '_mode_', str(mode), '.h5'])
        with h5py.File(file_out_mode, "w") as fout:
            fout.create_dataset("Frequency", data=freq)
            fout.create_dataset("DispersionCurve", data=disp_curves)
            fout.create_dataset("Uncertainties", data=vel_uncert)
            fout.create_dataset("MissingSamples", data=mask)
            fout.create_dataset("X_coord", data=x_mesh)
            fout.create_dataset("Y_coord", data=y_mesh)

        dict_h5={'Frequency':freq,
                'DispersionCurve':disp_curves,
                'Uncertainties':vel_uncert,
                'MissingSamples':mask,
                'X_coord':x_mesh,
                'Y_coord':y_mesh}

        dict_h5_list.append(dict_h5)

    return dict_h5_list


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
    df_velocity_global, df_thickness_global, df_interfaces_global, dx_in, dy_in = \
        read_model_fmt1(path_model_in, n_interfaces)

    # select only points where all the horizons are well defined
    df_interfaces_valid = df_interfaces_global[~df_interfaces_global.isnull().any(axis=1)]
    df_thickness_valid = df_thickness_global[~df_interfaces_global.isnull().any(axis=1)]
    df_velocity_valid = df_velocity_global[~df_interfaces_global.isnull().any(axis=1)]

    # interpolate on desired grid
    if settings_synthetics.bounds_mode=='auto':
        xmin = df_velocity_valid['X'].min()
        ymin = df_velocity_valid['Y'].min()
        xmax = df_velocity_valid['X'].max()
        ymax = df_velocity_valid['Y'].max()
    elif settings_synthetics.bounds_mode=='manual':
        xmin = settings_synthetics.xmin
        ymin = settings_synthetics.ymin
        xmax = settings_synthetics.xmax
        ymax = settings_synthetics.ymax
    else:
        raise Exception('unknown bounds_mode value')
    df_velocity_interp = interpolate_model_per_layer(df_velocity_valid, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                                     n_cells=settings_synthetics.n_cells,
                                                     lateral_smooth=settings_synthetics.lateral_smooth,
                                                     smooth_length=settings_synthetics.smooth_length,
                                                     dx_in = dx_in, dy_in = dy_in)
    df_thickness_interp = interpolate_model_per_layer(df_thickness_valid, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                                     n_cells=settings_synthetics.n_cells,
                                                     lateral_smooth=settings_synthetics.lateral_smooth,
                                                     smooth_length=settings_synthetics.smooth_length,
                                                     dx_in = dx_in, dy_in = dy_in)

    # select only points where all values are well defined
    # df_interfaces_valid = df_interfaces_valid[~df_velocity_interp.isnull().any(axis=1)]
    # df_thickness_valid = df_thickness_interp[~df_velocity_interp.isnull().any(axis=1)]
    # df_velocity_valid = df_velocity_interp[~df_velocity_interp.isnull().any(axis=1)]

    # compute dispersion curves
    nb_f = int(np.ceil((settings_synthetics.f_stop - settings_synthetics.f_start)/settings_synthetics.f_step))+1
    dispersion_dict = loop_on_cells(df_velocity_interp, df_thickness_interp, settings_synthetics.vp_over_vs,
                  settings_synthetics.f_start, settings_synthetics.f_stop, nb_f,
                  settings_synthetics.wavetype, settings_synthetics.modes,
                  settings_synthetics.velocity_mode, settings_synthetics.ny)

    file_out = str(Path(make_synthetics.path_out_format_1).joinpath(make_synthetics.file_out_format_1))
    dict_h5_list = save_h5(dispersion_dict, file_out)
