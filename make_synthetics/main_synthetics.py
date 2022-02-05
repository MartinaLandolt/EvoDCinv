# -*- coding: utf-8 -*-

import pickle
import make_synthetics
from make_synthetics import settings_synthetics
from make_synthetics.tomo_plugin import *
from pathlib import Path
import glob
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from evodcinv import LayeredModel, ThomsonHaskell, params2lay
from make_synthetics.cps_plugin import get_cps_dispersion
import pyproj
from pyproj import CRS
from pyproj import Transformer
import h5py
from scipy.signal import medfilt2d
from scipy.spatial import ConvexHull, Delaunay
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.ioff()


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0

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

    # columns_velocity = list(df_velocity_global)
    columns_interfaces = list(df_interfaces_global)

    # df_thickness_global = df_interfaces_global[['X', 'Y']].copy()
    # for (i, col_vel) in enumerate(columns_velocity[2:]):
    #     j=i+2
    #     z_i_minus_1 = df_interfaces_global.iloc[:,j].values
    #     z_i = df_interfaces_global.iloc[:, j+1].values
    #     col_thick = col_vel.replace('Vint', 'Thickness')
    #     df_thickness_global[col_thick] = z_i - z_i_minus_1

    # keep only common (X,Y) between thickness and velocity
    # columns_thickness = list(df_thickness_global)
    #df_thickness_global_merge = pd.merge(df_velocity_global, df_thickness_global, how='inner', on=['X', 'Y'])
    df_interfaces_global_merge = pd.merge(df_velocity_global, df_interfaces_global, how='inner', on=['X', 'Y'])
    # df_thickness_global = df_thickness_global_merge[columns_thickness]
    df_interfaces_global = df_interfaces_global_merge[columns_interfaces]

    return df_velocity_global, df_interfaces_global, dx_in, dy_in


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


def compute_thickness(df_interfaces, df_velocity):

    columns_velocity = list(df_velocity)
    columns_interfaces = list(df_interfaces)

    df_thickness = df_interfaces[['X', 'Y']].copy()
    for (i, col_vel) in enumerate(columns_velocity[2:]):
        j = i+2
        z_i_minus_1 = df_interfaces.iloc[:, j].values
        z_i = df_interfaces.iloc[:, j+1].values
        col_thick = col_vel.replace('Vint', 'Thickness')
        df_thickness[col_thick] = z_i - z_i_minus_1

    return df_thickness


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


def remove_outliers(df_in, std_thresh=3):
    cols = list(df_in)
    for col_i in cols[2:]:
        vals_i = df_in[col_i].values
        mean_val = np.nanmean(vals_i)
        std_val = np.nanstd(vals_i)
        vals_i[np.abs(vals_i - mean_val) > std_thresh * std_val] = np.nan
        df_in[col_i] = vals_i

    return df_in


def edit_velocities_as_suggested_by_catherine(df_velocity):
    df_velocity['(H01-H02) Vint'] = 1600.       # sables de l'orléanais
    df_velocity['(H02-H03) Vint'] = 1900.       # calcaires de beauce
    df_velocity['(H03-H04) Vint'] = 2200.       # craie
    df_velocity['(H04-H10) Vint'] = 2200.       # merge cenomanien and craie
    return df_velocity


def replace_velocities_by_mean_values_except_layer1(df_in):
    cols = list(df_in)
    for col_i in cols[3:]:
        vals_i = df_in[col_i].values
        mean_val = np.nanmean(vals_i)
        df_in[col_i] = mean_val

    return df_in


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


def make_1d_model_for_cell(thickness_array_in, vp_array_in, vs_array_in,
                           last_layer_vp=6000,  last_vp_over_vs=2, last_layer_thickness=99999.,
                           min_thickness_tol=1):
    """ ignores layers thinner than 10 m """
    thickness_array_out = np.hstack((thickness_array_in[thickness_array_in > min_thickness_tol], last_layer_thickness))
    vp_array_out = np.hstack((vp_array_in[thickness_array_in > min_thickness_tol], last_layer_vp))
    vs_array_out = np.hstack((vs_array_in[thickness_array_in > min_thickness_tol], last_layer_vp/last_vp_over_vs))
    if max(vp_array_out) <= 0:
        raise Exception('Encountered non-strictly positive velocity after removing zero-thickness layers')
    return thickness_array_out, vp_array_out, vs_array_out


def apply_vp_over_vs_ratio(vp_array, vp_over_vs_ratio=[2], n_layers_vp_over_vs=[]):
    assert len(vp_over_vs_ratio) == len(n_layers_vp_over_vs) + 1
    vs_array = 1 / vp_over_vs_ratio[-1] * vp_array
    # apply vp/vs per layers
    n_cum = 0
    for (i, n) in enumerate(n_layers_vp_over_vs):
        vs_array[n_cum: n_cum + n] = 1 / vp_over_vs_ratio[i] * vp_array[n_cum: n_cum+n]
        n_cum = n_cum + n

    return vs_array


def apply_vp_over_vs_ratio_on_dataframe(df_vp, vp_over_vs_ratio=[2], n_layers_vp_over_vs=[]):
    assert len(vp_over_vs_ratio) == len(n_layers_vp_over_vs) + 1
    df_vs = df_vp.copy()
    df_vp_over_vs = df_vp.copy()
    df_vs.iloc[:, 2:] = 1 / vp_over_vs_ratio[-1] * df_vp.iloc[:, 2:]
    # apply vp/vs per layers
    n_cum = 0
    for (i, n) in enumerate(n_layers_vp_over_vs):
        df_vs.iloc[:, 2 + n_cum: 2 + n_cum + n] = 1 / vp_over_vs_ratio[i] * \
                                           df_vp.iloc[:, 2 + n_cum: 2 + n_cum + n]
        n_cum = n_cum + n

    df_vp_over_vs.iloc[:, 2:] = df_vp.iloc[:, 2:] / df_vs.iloc[:, 2:]

    return df_vs, df_vp_over_vs

def convert_1d_model_to_th_format(vp_array, vs_array, thickness_array):

    nb_layers = len(vp_array)
    # create model in single line
    m = np.nan*np.zeros(nb_layers*3)
    m[0 : nb_layers] = vs_array
    m[nb_layers : 2*nb_layers] = thickness_array
    m[2*nb_layers : 3*nb_layers] = vp_array/vs_array

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


def compare_to_cps(l, f_axis, ny, modes, wavetype, fig_name):
    """ function to compare Thomson Haskell & CPS forward predictions """
    dc_calculated = get_dispersion_curve(l, f_axis, ny, modes, wavetype, 'group')
    src_dir = Path(__file__).parent.resolve()
    tmp_folder = Path(src_dir).joinpath('tmp_cps_from_evodcinv')
    if wavetype == 'rayleigh':
        cps_wavetype = 'R'
    elif wavetype == 'love':
        cps_wavetype = 'L'
    nmod = max(modes)
    cps_results = get_cps_dispersion(l, f_axis, nmod, cps_wavetype, tmp_folder)
    fig_folder = tmp_folder.joinpath('figs_out')
    if not fig_folder.exists():
        fig_folder.mkdir()
    dict_compare = dict()
    for (i, mode) in enumerate(modes):
        f_axis_evo = dc_calculated[i].faxis
        c_evo = dc_calculated[i].phase_velocity
        u_evo = dc_calculated[i].group_velocity
        field_cps = cps_wavetype + str(mode)
        f_axis_cps = 1./cps_results[field_cps]['T']
        c_cps = cps_results[field_cps]['C'] * 1000
        u_cps = cps_results[field_cps]['U'] * 1000
        dict_mode = dict()
        for variable in ["f_axis_evo", "c_evo", "u_evo",
                         "f_axis_cps", "c_cps", "u_cps"]:
            dict_mode[variable] = eval(variable)
        dict_compare[field_cps] = dict_mode
        fig, ax = plt.subplots(1)
        ax.plot(f_axis_cps, c_cps, color='k', linestyle='--')
        ax.plot(f_axis_evo, c_evo, color='k')
        ax.plot(f_axis_cps, u_cps, color='r', linestyle='--')
        ax.plot(f_axis_evo, u_evo, color='r')
        plt.legend(['phase cps', 'phase evo', 'group cps', 'group evo'])
        plt.grid()
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('m/s')
        plt.title(field_cps)
        figname_generic = field_cps
        figname_full = figname_generic + fig_name + '.png'
        fig_path = Path(fig_folder).joinpath(figname_full)
        fig.savefig(fig_path)
        plt.close(fig)

def loop_on_cells(df_velocity_global, df_thickness_global, vp_over_vs_ratio, n_layers_vp_over_vs,
                  fmin, fmax, nb_f, wavetype, modes, velocity_mode, ny,
                  vel_last_layer, bool_compare_to_cps=False):

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
    nb_layers_max = len(set(df_velocity_global)) - 2 + 1    # minus 'X' and 'Y' columns, plus infinite half space
    dispersion_dict['true_model'] = np.nan * np.zeros((len(df_velocity_global), nb_layers_max, 4))

    for mode in modes:
        dispersion_dict["".join(['mode_', str(mode)])] = np.nan * np.zeros((len(df_velocity_global), nb_f))

    cell_count = 0
    for (i, df_velocity_cell), (j, df_thickness_cell) in zip(df_velocity_global.iterrows(),
                                                                  df_thickness_global.iterrows()):
        thickness_array = df_thickness_cell.iloc[2:].values
        vp_array = df_velocity_cell.iloc[2:].values
        vs_array = apply_vp_over_vs_ratio(vp_array,
                                              vp_over_vs_ratio=vp_over_vs_ratio,
                                              n_layers_vp_over_vs=n_layers_vp_over_vs)
        if np.isnan(vp_array).any():
            for mode in modes:
                dispersion_dict["".join(['mode_', str(mode)])][cell_count, :] = np.nan
        else:
            thickness_array_clean, vp_array_clean, vs_array_clean = \
                make_1d_model_for_cell(thickness_array, vp_array, vs_array,
                                       last_layer_vp=vel_last_layer,
                                       last_vp_over_vs=vp_over_vs_ratio[-1])
            l = convert_1d_model_to_th_format(vp_array_clean, vs_array_clean, thickness_array_clean)
            dispersion_dict['true_model'][cell_count, 0:l.shape[0], :] = l

            # try:
            if bool_compare_to_cps:
                fig_name = '_cell_' + str(i) + '_' + str(j)
                compare_to_cps(l, f, ny, modes, wavetype, fig_name)
            dc_calculated = get_dispersion_curve(l, f, ny, modes, wavetype, velocity_mode)
            # except:
            #     dc_calculated = get_dispersion_curve(l, f, ny, modes, wavetype, velocity_mode)

            # write in the array
            for mode in modes:
                vel_vals = getattr(dc_calculated[mode], "".join([velocity_mode, "_velocity"]))
                f_vals = getattr(dc_calculated[mode], 'faxis')
                i_f = [i for (i, x) in enumerate(f) if (x in f_vals)]
                try:
                    dispersion_dict["".join(['mode_', str(mode)])][cell_count, i_f] = vel_vals
                except:
                    print('cell ', cell_count, ' out of ', len(df_velocity_global))
                    print('debug')

        cell_count += 1
        print('cell ', cell_count, ' out of ', len(df_velocity_global))

    return dispersion_dict


def save_h5(dispersion_dict, file_out):
    # resize coordinates to mesh format
    x_mesh = np.reshape(dispersion_dict['X'].values, (settings_synthetics.n_cells, settings_synthetics.n_cells))
    y_mesh = np.reshape(dispersion_dict['Y'].values, (settings_synthetics.n_cells, settings_synthetics.n_cells))
    freq = dispersion_dict['f_axis']
    true_model = dispersion_dict['true_model']
    true_model_reshape = np.reshape(true_model, (settings_synthetics.n_cells, settings_synthetics.n_cells,
                                                 true_model.shape[1], true_model.shape[2]))
    dispersion_dict['wavetype']
    dispersion_dict['velocity_mode']
    dict_h5_list = []
    for mode in dispersion_dict['modes']:
        dispersion_array = dispersion_dict["".join(['mode_', str(mode)])]
        nan_test = np.sum(~np.isnan(dispersion_array), axis=1)
        # create the MissingSamples
        mask = (nan_test < len(freq)).astype('int')
        # create the uncertainties
        vel_uncert = dispersion_array * 0.03
        # resize
        disp_curves = np.reshape(dispersion_array,
                                 (settings_synthetics.n_cells, settings_synthetics.n_cells, len(freq)))
        mask = np.reshape(mask, (settings_synthetics.n_cells, settings_synthetics.n_cells,))
        vel_uncert = np.reshape(vel_uncert, (settings_synthetics.n_cells, settings_synthetics.n_cells, len(freq)))
        # write to h5 file
        file_out_mode = "".join([file_out, '_',
                                 dispersion_dict['wavetype'], '_', dispersion_dict['velocity_mode'],
                                 '_mode_', str(mode), '.h5'])
        with h5py.File(file_out_mode, "w") as fout:
            fout.create_dataset("Frequency", data=freq)
            fout.create_dataset("DispersionCurve", data=disp_curves)
            fout.create_dataset("Uncertainties", data=vel_uncert)
            fout.create_dataset("MissingSamples", data=mask)
            fout.create_dataset("X_coord", data=x_mesh)
            fout.create_dataset("Y_coord", data=y_mesh)
            fout.create_dataset("true_model", data=true_model_reshape)
            fout.attrs["wavetype"] = dispersion_dict['wavetype']
            fout.attrs["velocity_mode"] = dispersion_dict['velocity_mode']

        dict_h5={'Frequency':freq,
                'DispersionCurve':disp_curves,
                'Uncertainties':vel_uncert,
                'MissingSamples':mask,
                'X_coord':x_mesh,
                'Y_coord':y_mesh,
                'true_model':true_model_reshape,
                'wavetype':dispersion_dict['wavetype'],
                'velocity_mode':dispersion_dict['velocity_mode']}

        dict_h5_list.append(dict_h5)

    return dict_h5_list


def save_model_plots(df_data, name_data, n_cells_x, n_cells_y, folder_out, vp_over_vs=None):
    """writes the following pngs :
    - Z of horizons
    - H of horizons
    - V of layers
    - cross-sections V(x,z) and V(y,z) for specified X and Y values
    Assumes the model has been intepolated on a rectagular grid"""

    x = df_data['X'].values
    y = df_data['Y'].values
    x_mesh = np.reshape(x, (n_cells_y, n_cells_x)) / 1000   # convert to km
    y_mesh = np.reshape(y, (n_cells_y, n_cells_x)) / 1000   # convert to km

    path_data = Path(folder_out).joinpath(name_data + '_plots')
    if name_data == 'interface':
        str_cbar = 'Depth bsl (m) '
    elif name_data == 'vp':
        str_cbar = 'Interval Vp (m/s)'
    elif name_data == 'vs':
        str_cbar = 'Interval Vs (m/s)'
    elif name_data == 'thickness':
        str_cbar = 'Interval thickness (m)'
    else:
        raise Exception("".join(['unknown name_data: ', name_data]))

    # plot data
    if not path_data.exists():
        path_data.mkdir()
    cols_data = list(df_data)
    for col_i in cols_data[2:]:
        z = df_data[col_i].values
        val_min = np.nanmin(z) - 1
        val_max = np.nanmax(z) + 1
        z_mesh = np.reshape(z, (n_cells_y, n_cells_x))
        fig, ax = plt.subplots()
        h_im = ax.pcolormesh(x_mesh, y_mesh, z_mesh)
        h_im.set_clim(val_min, val_max)
        h_cbar = plt.colorbar(mappable=h_im)
        h_cbar.ax.set_ylabel(str_cbar, rotation=270)
        plt.grid(True, which='major', linestyle='-')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_title(col_i)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')
        fig.savefig(str(path_data.joinpath(col_i)) + '.png')
        plt.close(fig)


def save_cross_section_plots_along_with_tomo_outputs(df_interfaces, dispersion_dict,
                                                     n_cells_x, n_cells_y, folder_out,
                             fixed_coord='x', coord_val=None,
                             delta_x_disp_curves=None, mode_field=None, dispersion_curves_tomo_file=None):
    """ OBSOLETE DO NOT USE """

    dict_disp_curves_tomo = h5_to_dict(str(dispersion_curves_tomo_file))
    x_axis_tomo = dict_disp_curves_tomo['X_coord'][0, :]
    y_axis_tomo = dict_disp_curves_tomo['Y_coord'][:, 0]
    faxis_tomo = dict_disp_curves_tomo['Frequency']

    path_data = Path(folder_out).joinpath('cross_section_plots')
    if not path_data.exists():
        path_data.mkdir()

    x_axis = np.unique(df_interfaces['X'].values)
    y_axis = np.unique(df_interfaces['Y'].values)

    # initialize 2d slice
    if fixed_coord == 'x':
        other_coord = 'y'
        n_cells = n_cells_y
        i_cell = [i for i in range(len(x_axis)) if np.abs(x_axis[i] - coord_val) == min(np.abs(x_axis - coord_val))]
        coord_val_true = x_axis[i_cell][0]
        coord_axis = y_axis
        str_xlabel = 'Y (km)'
    elif fixed_coord == 'y':
        other_coord = 'x'
        n_cells = n_cells_x
        i_cell = [i for i in range(len(y_axis)) if np.abs(y_axis[i] - coord_val) == min(np.abs(y_axis - coord_val))]
        coord_val_true = y_axis[i_cell][0]
        coord_axis = x_axis
        str_xlabel = 'X (km)'
    else:
        raise Exception("".join(['unknown fixed coord: ', fixed_coord]))

    # plot dispersion curves along section
    cmap = matplotlib.cm.get_cmap('brg')
    fig, ax = plt.subplots()
    vals_all = dispersion_dict[mode_field]
    faxis = dispersion_dict['f_axis']
    str_cbar = dispersion_dict['velocity_mode'] + ' velocity (m/s)'

    coord_list_disp_curves = np.arange(np.ceil(np.nanmin(coord_axis) / 1000),
                                       np.floor(np.nanmax(coord_axis)) / 1000,
                                       delta_x_disp_curves) * 1000
    h_plot_list = []
    str_legend = []
    for (ii, coord_i) in enumerate(coord_list_disp_curves):
        if fixed_coord == 'x':
            i_other = [i for i in range(len(y_axis)) if
                       np.abs(y_axis[i] - coord_i) == min(np.abs(y_axis - coord_i))]
            coord_i_true = y_axis[i_other][0]
            i_cell = np.where((dispersion_dict['X'] == coord_val_true) &
                              (dispersion_dict['Y'] == coord_i_true))
            dist_to_tomo_x = np.abs(x_axis_tomo - coord_val_true)
            dist_to_tomo_y = np.abs(y_axis_tomo - coord_i_true)
        else:
            i_other = [i for i in range(len(x_axis)) if
                       np.abs(x_axis[i] - coord_i) == min(np.abs(x_axis - coord_i))]
            coord_i_true = x_axis[i_other][0]
            i_cell = np.where((dispersion_dict['Y'] == coord_val_true) &
                              (dispersion_dict['X'] == coord_i_true))
            dist_to_tomo_x = np.abs(x_axis_tomo - coord_i_true)
            dist_to_tomo_y = np.abs(y_axis_tomo - coord_val_true)

        dist_to_tomo = np.sqrt(min(dist_to_tomo_x)**2 + min(dist_to_tomo_y)**2)
        i_cell_tomo_x = np.where(dist_to_tomo_x == min(dist_to_tomo_x))
        i_cell_tomo_y = np.where(dist_to_tomo_y == min(dist_to_tomo_y))
        # WARNING : transposed tomo output !!!
        if (dist_to_tomo < settings_synthetics.cell_size) & \
            (dict_disp_curves_tomo['MissingSamples'][i_cell_tomo_x, i_cell_tomo_y]==0):
            disp_curve_tomo = \
                dict_disp_curves_tomo['DispersionCurve'][i_cell_tomo_x, i_cell_tomo_y].squeeze()
            uncert_tomo = \
                dict_disp_curves_tomo['Uncertainties'][i_cell_tomo_x, i_cell_tomo_y].squeeze()
            disp_curve_i = vals_all[i_cell, :].flatten()
            h_plot, = ax.plot(faxis, disp_curve_i,
                              color=cmap(float(ii / len(coord_list_disp_curves))),
                              linestyle='--')
            ax.plot(faxis_tomo, disp_curve_tomo,
                              color=cmap(float(ii / len(coord_list_disp_curves))),
                              linestyle='-')
            # ax.errorbar(faxis_tomo, disp_curve_tomo, uncert_tomo,
            #                   color=cmap(float(ii / len(coord_list_disp_curves))),
            #                   linestyle='-')
            h_plot_list.append(h_plot)
            # print(coord_i_true/1000)
            str_legend.append("".join([other_coord, " = ", "{:.1f}".format(coord_i_true / 1000), ' km']))
            ax.set_ylabel(str_cbar, rotation=270)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_title("".join(['Dispersion curves along section at ',
                                  fixed_coord, ' = ', "{:.1f}".format(coord_val_true / 1000), ' km']))
            plt.grid(True, which='major', linestyle='-')
    #f_max_all = max(max(faxis), max(faxis_tomo))
    plt.legend(h_plot_list, str_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig.savefig(str(path_data.joinpath('dispersion_curves_along_section_compared' +
                                       "".join([fixed_coord + "{:.1f}".format(coord_val / 1000) + 'km']) +
                                       '.png')))
    plt.close(fig)


def save_cross_section_plots(df_interfaces, df_velocity, dispersion_dict, n_cells_x, n_cells_y, folder_out,
                             fixed_coord='x', coord_val=None, dz=5,
                             z_margin_bottom=200, z_margin_top=20, vel_last_layer=None,
                             delta_x_disp_curves=None, mode_field=None,
                             data_type='vs',
                             compare_tomo=False, dispersion_curves_tomo_file=None,
                             clim_model=None, ylim_disp_curves=None,
                             plot_model_cross_section=False):

    if compare_tomo:
        dict_disp_curves_tomo = h5_to_dict(str(dispersion_curves_tomo_file))
        x_axis_tomo = dict_disp_curves_tomo['X_coord'][0, :]
        y_axis_tomo = dict_disp_curves_tomo['Y_coord'][:, 0]
        faxis_tomo = dict_disp_curves_tomo['Frequency']

    path_data = Path(folder_out).joinpath('cross_section_plots')
    if not path_data.exists():
        path_data.mkdir()

    # compute nz
    zmin = np.nanmin(df_interfaces.iloc[:, 2:].values) - z_margin_top
    zmax = np.nanmax(df_interfaces.iloc[:, 2:].values) + z_margin_bottom
    nz = int(np.ceil(zmax/dz))

    # create z_axis
    z_axis = np.linspace(zmin, zmax, nz)
    x_axis = np.unique(df_interfaces['X'].values)
    y_axis = np.unique(df_interfaces['Y'].values)

    # initialize 2d slice
    if fixed_coord == 'x':
        other_coord = 'y'
        n_cells = n_cells_y
        i_cell = [i for i in range(len(x_axis)) if np.abs(x_axis[i] - coord_val) == min(np.abs(x_axis - coord_val))]
        coord_val_true = x_axis[i_cell][0]
        coord_axis = y_axis
        str_xlabel = 'Y (km)'
    elif fixed_coord == 'y':
        other_coord = 'x'
        n_cells = n_cells_x
        i_cell = [i for i in range(len(y_axis)) if np.abs(y_axis[i] - coord_val) == min(np.abs(y_axis - coord_val))]
        coord_val_true = y_axis[i_cell][0]
        coord_axis = x_axis
        str_xlabel = 'X (km)'
    else:
        raise Exception("".join(['unknown fixed coord: ', fixed_coord]))
    slice_section = np.nan * np.zeros((nz, n_cells))

    if plot_model_cross_section:
        fig, ax = plt.subplots()
        # find ground level
        z_ground_level_all_vals = df_interfaces.iloc[:, 2].values
        z_ground_level_reshape = np.reshape(z_ground_level_all_vals, (n_cells_y, n_cells_x))
        if fixed_coord == 'x':
            z_ground_level_slice = z_ground_level_reshape[:, i_cell].flatten()
        else:
            z_ground_level_slice = z_ground_level_reshape[i_cell, :].flatten()

        # loop on velocities
        cols = list(df_interfaces)
        z_minus_1_slice = z_ground_level_slice
        for (i, col_i) in enumerate(cols[3:]):
            # i interface
            z_i = df_interfaces[col_i].values
            v_i = df_velocity.iloc[:, i+2].values
            z_i_reshape = np.reshape(z_i, (n_cells_y, n_cells_x))
            v_i_reshape = np.reshape(v_i, (n_cells_y, n_cells_x))
            # extract rows corresponding to the fixed coordinate
            if fixed_coord == 'x':
                z_i_slice = z_i_reshape[:, i_cell].flatten()
                v_i_slice = v_i_reshape[:, i_cell].flatten()
            else:
                z_i_slice = z_i_reshape[i_cell, :].flatten()
                v_i_slice = v_i_reshape[i_cell, :].flatten()
            # loop on rows i_r
            for j in range(n_cells):
                slice_section[np.where((z_axis>z_minus_1_slice[j]) & (z_axis<=z_i_slice[j])), j] = v_i_slice[j]
            # update i-1 interface
            z_minus_1_slice = z_i_slice
        # fill bottom layer
        for j in range(n_cells):
            slice_section[np.where(z_axis > z_i_slice[j]), j] = vel_last_layer
        h_im = ax.pcolormesh(coord_axis/1000, z_axis, slice_section)
        h_cbar = plt.colorbar(mappable=h_im)
        if data_type == 'vs':
            h_cbar.ax.set_ylabel('Interval Vs (m/s)', rotation=270)
        elif data_type == 'vp':
            h_cbar.ax.set_ylabel('Interval Vp (m/s)', rotation=270)
        elif data_type == 'vp_over_vs':
            h_cbar.ax.set_ylabel('Interval Vp/Vs', rotation=270)
        plt.grid(True, which='major', linestyle='-')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        str_title = "".join(['Cross-section at ', fixed_coord, ' = ', str(coord_val / 1000), ' km'])
        ax.set_title(str_title)
        ax.set_xlabel(str_xlabel)
        ax.set_ylabel('Depth bsl (m)')
        if clim_model is not None:
            h_im.set_clim(clim_model)
        ax.invert_yaxis()

        # plot all the interfaces on top of the image
        for (i, col_i) in enumerate(cols[2:]):
            # i interface
            z_i = df_interfaces[col_i].values
            z_i_reshape = np.reshape(z_i, (n_cells_y, n_cells_x))
            # extract rows corresponding to the fixed coordinate
            if fixed_coord == 'x':
                z_i_slice = z_i_reshape[:, i_cell].flatten()
            else:
                z_i_slice = z_i_reshape[i_cell, :].flatten()
            ax.plot(coord_axis / 1000, z_i_slice, color='k')

        fig.savefig(str(path_data.joinpath('section_' + data_type + '_' +
                    "".join([fixed_coord + str(coord_val / 1000) + 'km']) +
                    '.png')))
        plt.close(fig)

    if delta_x_disp_curves is not None:

        crit_cell_exists_in_tomo = True

        # plot dispersion curves along section
        cmap = matplotlib.cm.get_cmap('brg')
        fig, ax = plt.subplots()
        vals_all = dispersion_dict[mode_field]
        faxis = dispersion_dict['f_axis']
        str_cbar = dispersion_dict['velocity_mode'] + ' velocity (m/s)'

        coord_list_disp_curves = np.arange(np.ceil(np.nanmin(coord_axis)/1000),
                                           np.floor(np.nanmax(coord_axis))/1000,
                                           delta_x_disp_curves) * 1000
        h_plot_list = []
        str_legend = []
        for (ii, coord_i) in enumerate(coord_list_disp_curves):
            if fixed_coord == 'x':
                i_other = [i for i in range(len(y_axis)) if
                          np.abs(y_axis[i] - coord_i) == min(np.abs(y_axis - coord_i))]
                coord_i_true = y_axis[i_other][0]
                i_cell = np.where((dispersion_dict['X'] == coord_val_true) &
                                  (dispersion_dict['Y'] == coord_i_true))
                if compare_tomo:
                    dist_to_tomo_x = np.abs(x_axis_tomo - coord_val_true)
                    dist_to_tomo_y = np.abs(y_axis_tomo - coord_i_true)
            else:
                i_other = [i for i in range(len(x_axis)) if
                           np.abs(x_axis[i] - coord_i) == min(np.abs(x_axis - coord_i))]
                coord_i_true = x_axis[i_other][0]
                i_cell = np.where((dispersion_dict['Y'] == coord_val_true) &
                                  (dispersion_dict['X'] == coord_i_true))
                if compare_tomo:
                    dist_to_tomo_x = np.abs(x_axis_tomo - coord_i_true)
                    dist_to_tomo_y = np.abs(y_axis_tomo - coord_val_true)
            if compare_tomo:
                # WARNING : transposed tomo output !!!
                dist_to_tomo = np.sqrt(min(dist_to_tomo_x) ** 2 + min(dist_to_tomo_y) ** 2)
                i_cell_tomo_x = np.where(dist_to_tomo_x == min(dist_to_tomo_x))
                i_cell_tomo_y = np.where(dist_to_tomo_y == min(dist_to_tomo_y))
                crit_cell_exists_in_tomo = \
                    (dist_to_tomo < settings_synthetics.cell_size) & \
                    (dict_disp_curves_tomo['MissingSamples'][i_cell_tomo_x, i_cell_tomo_y] == 0)
            if crit_cell_exists_in_tomo:
                disp_curve_i = vals_all[i_cell, :].flatten()
                h_plot, = ax.plot(faxis, disp_curve_i,
                                  color=cmap(float(ii/len(coord_list_disp_curves))),
                                  linestyle='--')
                h_plot_list.append(h_plot)
                # print(coord_i_true/1000)
                str_legend.append("".join([other_coord, " = ", "{:.1f}".format(coord_i_true/1000), ' km']))
                ax.set_ylabel(str_cbar, rotation=270)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_title("".join(['Dispersion curves along section at ',
                                      fixed_coord, ' = ', "{:.1f}".format(coord_val_true/1000), ' km']))
                plt.grid(True, which='major', linestyle='--')
                if compare_tomo:
                    disp_curve_tomo = \
                        dict_disp_curves_tomo['DispersionCurve'][i_cell_tomo_x, i_cell_tomo_y].squeeze()
                    uncert_tomo = \
                        dict_disp_curves_tomo['Uncertainties'][i_cell_tomo_x, i_cell_tomo_y].squeeze()
                    ax.plot(faxis_tomo, disp_curve_tomo,
                            color=cmap(float(ii / len(coord_list_disp_curves))),
                            linestyle='-')
                    # ax.errorbar(faxis_tomo, disp_curve_tomo, uncert_tomo,
                    #                   color=cmap(float(ii / len(coord_list_disp_curves))),
                    #                   linestyle='-')

        plt.legend(h_plot_list, str_legend, bbox_to_anchor=(1.05, 1), loc='upper left')
        if ylim_disp_curves is not None:
            ax.set_ylim(ylim_disp_curves)
        plt.tight_layout()
        if compare_tomo:
            fig_title = "dispersion_curves_along_section_compared_"
        else:
            fig_title = "dispersion_curves_along_section_"
        fig.savefig(str(path_data.joinpath(fig_title +
                                           "".join([fixed_coord + "{:.1f}".format(coord_val / 1000) + 'km']) +
                                           '.png')))
        plt.close(fig)


def save_dispersion_plots_along_with_tomo_outputs(dispersion_dict, field, n_cells_x, n_cells_y,
                                                  folder_out, n_skip=1,
                                                  tomo_folder=None, plot_mode='relative'):
    """OBSOLETE DO NOT USE"""

    assert dispersion_dict['velocity_mode'] == 'group'

    x = dispersion_dict['X'].values/1000
    y = dispersion_dict['Y'].values/1000
    coords_synthetics = np.vstack((x, y)).T
    x_mesh = np.reshape(x, (n_cells_y, n_cells_x))   # convert to km
    y_mesh = np.reshape(y, (n_cells_y, n_cells_x))  # convert to km

    if field in list(dispersion_dict):
        path_data = Path(folder_out).joinpath(field + '_plots')
    else:
        raise Exception("".join(['unknown mode: ', field]))

    if plot_mode == 'relative':
        str_cbar = dispersion_dict['velocity_mode'] + ' relative anomaly'
    else:
        str_cbar = dispersion_dict['velocity_mode'] + ' velocity (m/s)'

    if not path_data.exists():
        path_data.mkdir()

    vals_all = dispersion_dict[field]
    faxis = dispersion_dict['f_axis']

    # read tomo results in a dictionnary
    pattern = 'pass_2*'
    dict_tomo_global = read_tomo_files_h5(tomo_folder, pattern)

    # define a convex envelope of valid data
    f_tomo_str = dict_tomo_global.keys()
    f_tomo = [float(f_str) for f_str in f_tomo_str]
    for (i, freq_str) in enumerate(f_tomo_str):
        dict_freq = dict_tomo_global[freq_str]
        if i==0:
            reference_point = dict_freq['reference_frame']/1000
            X_mesh = dict_freq['X_mesh']/1000 + reference_point[0]
            Y_mesh = dict_freq['Y_mesh']/1000 + reference_point[1]
            vel_anom_non_nan = ~np.isnan(dict_freq['velocity_anomaly']).T
        else:
            vel_anom_non_nan += ~np.isnan(dict_freq['velocity_anomaly']).T

    vel_anom_non_nan_reshape = np.reshape(vel_anom_non_nan, len(np.ravel(vel_anom_non_nan)))
    X_mesh_reshape = np.reshape(X_mesh, len(np.ravel(vel_anom_non_nan)))
    Y_mesh_reshape = np.reshape(Y_mesh, len(np.ravel(vel_anom_non_nan)))
    points_valid = np.vstack((X_mesh_reshape[vel_anom_non_nan_reshape], Y_mesh_reshape[vel_anom_non_nan_reshape])).T
    x_min = np.min(points_valid[:, 0]) - 0.5
    x_max = np.max(points_valid[:, 0]) + 0.5
    y_min = np.min(points_valid[:, 1]) - 0.5
    y_max = np.max(points_valid[:, 1]) + 0.5
    # plot maps of velocity per frequency

    # loop on frequencies from tomo dictionnary
    for (i, freq_str) in enumerate(f_tomo_str):
        dict_freq = dict_tomo_global[freq_str]
        f = float(freq_str)
        # check the synthetics dictionnary contains
        # the desired frequency to within some tolerance
        i_f_synth = np.where(np.abs(faxis - f) < 0.01)[0]
        if len(i_f_synth) > 0:
            # get group velocity values for synthetics
            f_synth = faxis[i_f_synth[0]]
            z = vals_all[:, i_f_synth[0]]

            # replace values outside the polygon by nans
            i_valid = in_hull(coords_synthetics, points_valid)
            z[i_valid == False] = np.nan

            # read velocity anomalies from tomo dict (corresponding to f)
            vel_anom_tomo = - dict_freq['velocity_anomaly'].T
            vel_ref_tomo = dict_freq['apriori_velocity']

            # if mode is relative, convert synthetics to anomaly, or the other way
            if plot_mode == 'relative':
                vel_ref_synth = np.nanmean(z)
                vel_synth_plot = (z - vel_ref_synth)/vel_ref_synth
                vel_tomo_plot = vel_anom_tomo
            else:
                vel_synth_plot = z
                vel_tomo_plot = (vel_anom_tomo + 1) * vel_ref_tomo
            vel_synth_plot_mesh = np.reshape(vel_synth_plot, (n_cells_y, n_cells_x))

            # define common value bounds
            val_min = min(np.nanmin(np.ravel(vel_tomo_plot)), np.nanmin(np.ravel(vel_synth_plot)))
            val_max = max(np.nanmax(np.ravel(vel_tomo_plot)), np.nanmax(np.ravel(vel_synth_plot)))

            # create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # plot synthetics
            h_im1 = ax1.pcolormesh(x_mesh, y_mesh, vel_synth_plot_mesh)
            h_im1.set_clim(val_min, val_max)
            ax1.grid(True, which='major', linestyle='-')
            ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.set_aspect('equal', 'box')
            ax1.set_title("".join(["Synthetics, f = ", "{:.2f}".format(f_synth), ' Hz']))
            ax1.set_xlabel('X (km)')
            ax1.set_ylabel('Y (km)')
            h_cbar1 = plt.colorbar(mappable=h_im1, ax=ax1)

            # plot tomo results
            h_im2 = ax2.pcolormesh(X_mesh, Y_mesh, vel_tomo_plot)
            h_im2.set_clim(val_min, val_max)
            ax2.grid(True, which='major', linestyle='-')
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax2.set_title("".join(["Tomography, f = ", "{:.2f}".format(f), ' Hz']))
            ax2.set_xlim(x_min, x_max)
            ax2.set_ylim(y_min, y_max)
            ax2.set_aspect('equal', 'box')
            ax2.set_xlabel('X (km)')
            ax2.set_ylabel('Y (km)')
            h_cbar2 = plt.colorbar(mappable=h_im2, ax=ax2)
            h_cbar2.ax.set_ylabel(str_cbar, rotation=270)

            fig.savefig(str(path_data.joinpath("".join(["{:.2f}".format(f), '_Hz_compared_', plot_mode]))) + '.png')
            plt.close(fig)


def save_dispersion_plots(dispersion_dict, field, n_cells_x, n_cells_y,
                          folder_out, n_skip=1,
                          compare_tomo=False,
                          tomo_folder=None, plot_mode='relative'):
    """plot group velocity maps and compare with tomo results"""

    if compare_tomo:
        assert dispersion_dict['velocity_mode'] == 'group'

    x = dispersion_dict['X'].values/1000
    y = dispersion_dict['Y'].values/1000
    coords_synthetics = np.vstack((x, y)).T
    x_mesh = np.reshape(x, (n_cells_y, n_cells_x))   # convert to km
    y_mesh = np.reshape(y, (n_cells_y, n_cells_x))  # convert to km

    if field in list(dispersion_dict):
        path_data = Path(folder_out).joinpath(field + '_plots')
    else:
        raise Exception("".join(['unknown mode: ', field]))

    if plot_mode == 'relative':
        str_cbar = dispersion_dict['velocity_mode'] + ' relative anomaly'
    else:
        str_cbar = dispersion_dict['velocity_mode'] + ' velocity (m/s)'

    if not path_data.exists():
        path_data.mkdir()

    vals_all = dispersion_dict[field]
    faxis = dispersion_dict['f_axis']

    if compare_tomo:
        # read tomo results in a dictionnary
        pattern = 'pass_2*'
        dict_tomo_global = read_tomo_files_h5(tomo_folder, pattern)

        # define a convex envelope of valid data
        f_tomo_str = dict_tomo_global.keys()
        f_tomo = [float(f_str) for f_str in f_tomo_str]
        for (i, freq_str) in enumerate(f_tomo_str):
            dict_freq = dict_tomo_global[freq_str]
            if i == 0:
                reference_point = dict_freq['reference_frame'] / 1000
                X_mesh = dict_freq['X_mesh'] / 1000 + reference_point[0]
                Y_mesh = dict_freq['Y_mesh'] / 1000 + reference_point[1]
                vel_anom_non_nan = ~np.isnan(dict_freq['velocity_anomaly']).T
            else:
                vel_anom_non_nan += ~np.isnan(dict_freq['velocity_anomaly']).T

            vel_anom_non_nan_reshape = np.reshape(vel_anom_non_nan, len(np.ravel(vel_anom_non_nan)))
            X_mesh_reshape = np.reshape(X_mesh, len(np.ravel(vel_anom_non_nan)))
            Y_mesh_reshape = np.reshape(Y_mesh, len(np.ravel(vel_anom_non_nan)))
            points_valid = np.vstack((X_mesh_reshape[vel_anom_non_nan_reshape], Y_mesh_reshape[vel_anom_non_nan_reshape])).T
            x_min = np.min(points_valid[:, 0]) - 0.5
            x_max = np.max(points_valid[:, 0]) + 0.5
            y_min = np.min(points_valid[:, 1]) - 0.5
            y_max = np.max(points_valid[:, 1]) + 0.5

    if compare_tomo:
        f_loop = [float(f_i) for f_i in f_tomo_str]
    else:
        f_loop = faxis
    # plot maps of velocity per frequency
    for (i, f) in enumerate(f_loop):
        freq_str = "{:.2f}".format(f)
        if compare_tomo:
            dict_freq = dict_tomo_global[freq_str]
            f = float(freq_str)
            # check the synthetics dictionnary contains
            # the desired frequency to within some tolerance
            i_f_synth = np.where(np.abs(faxis - f) < 0.01)[0]
            if len(i_f_synth) > 0:
                # get group velocity values for synthetics
                f_synth = faxis[i_f_synth[0]]
                z = vals_all[:, i_f_synth[0]]

                # replace values outside the polygon by nans
                i_valid = in_hull(coords_synthetics, points_valid)
                z[i_valid == False] = np.nan

                # read velocity anomalies from tomo dict (corresponding to f)
                vel_anom_tomo = - dict_freq['velocity_anomaly'].T
                vel_ref_tomo = dict_freq['apriori_velocity']

                # if mode is relative, convert synthetics to anomaly, or the other way
                if plot_mode == 'relative':
                    vel_tomo_plot = vel_anom_tomo
                else:
                    vel_tomo_plot = (vel_anom_tomo + 1) * vel_ref_tomo

        else:
            z = vals_all[:, i]
        # if mode is relative, convert synthetics to anomaly, or the other way
        if plot_mode == 'relative':
            vel_ref_synth = np.nanmean(z)
            vel_synth_plot = (z - vel_ref_synth) / vel_ref_synth
        else:
            vel_synth_plot = z
        vel_synth_plot_mesh = np.reshape(vel_synth_plot, (n_cells_y, n_cells_x))

        if compare_tomo:
            # define common value bounds
            val_min = min(np.nanmin(np.ravel(vel_tomo_plot)), np.nanmin(np.ravel(vel_synth_plot)))
            val_max = max(np.nanmax(np.ravel(vel_tomo_plot)), np.nanmax(np.ravel(vel_synth_plot)))
            nb_subplots = 2
            fig_size = (12,6)
        else:
            val_min = np.nanmin(np.ravel(vel_synth_plot))
            val_max = np.nanmax(np.ravel(vel_synth_plot))
            nb_subplots = 1
            fig_size = (7,6)

        fig, _ = plt.subplots(1, nb_subplots, figsize=fig_size)
        ax_list = fig.axes
        h_im1 = ax_list[0].pcolormesh(x_mesh, y_mesh, vel_synth_plot_mesh)
        h_im1.set_clim(val_min, val_max)
        ax_list[0].grid(True, which='major', linestyle='-')
        ax_list[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_list[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_list[0].set_aspect('equal', 'box')
        ax_list[0].set_title("".join(["Synthetics, f = ", freq_str, ' Hz']))
        ax_list[0].set_xlabel('X (km)')
        ax_list[0].set_ylabel('Y (km)')
        h_cbar1 = plt.colorbar(mappable=h_im1, ax=ax_list[0])
        str_add_to_title = ''

        if compare_tomo:
            ax_list[0].set_title("".join(["Synthetics, f = ", "{:.2f}".format(f_synth), ' Hz']))
            ax_list[0].set_xlim(x_min, x_max)
            ax_list[0].set_ylim(y_min, y_max)
            h_im2 = ax_list[1].pcolormesh(X_mesh, Y_mesh, vel_tomo_plot)
            h_im2.set_clim(val_min, val_max)
            ax_list[1].grid(True, which='major', linestyle='-')
            ax_list[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax_list[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax_list[1].set_title("".join(["Tomography, f = ", "{:.2f}".format(f), ' Hz']))
            ax_list[1].set_xlim(x_min, x_max)
            ax_list[1].set_ylim(y_min, y_max)
            ax_list[1].set_aspect('equal', 'box')
            ax_list[1].set_xlabel('X (km)')
            ax_list[1].set_ylabel('Y (km)')
            h_cbar2 = plt.colorbar(mappable=h_im2, ax=ax_list[1])
            h_cbar2.ax.set_ylabel(str_cbar, rotation=270)
            str_add_to_title = '_compared'

        fig.savefig(str(path_data.joinpath("".join(["{:.2f}".format(f), '_Hz_',
                                                    plot_mode, str_add_to_title]))) + '.png')
        plt.close(fig)

    # plot all dispersion curves in one plot for checking mode jumps
    fig, ax = plt.subplots()
    all_indices = range(len(x))
    subindices = all_indices[::n_skip]
    for i in subindices:
        disp_curve_i = vals_all[i, :]
        ax.plot(faxis, disp_curve_i)
        ax.set_ylabel(str_cbar, rotation=270)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_title('All dispersion curves')
        plt.grid(True, which='major', linestyle='-')
    fig.savefig(str(path_data.joinpath('all_curves' + '.png')))
    plt.close(fig)


if __name__ == '__main__':

    recompute_dispersion = settings_synthetics.recompute_dispersion
    reload_interfaces = settings_synthetics.reload_interfaces

    if recompute_dispersion:
        assert reload_interfaces

    print('model format : ', settings_synthetics.type_vel_model)
    if settings_synthetics.type_vel_model == 1:
        path_model_in = make_synthetics.data_format_1
        if not Path(path_model_in).exists():
            raise Exception("".join(['Data not found : ', path_model_in]))
        path_synthetics_out = make_synthetics.path_out_format_1
        if not Path(path_synthetics_out).exists():
            Path(path_synthetics_out).mkdir()

    if reload_interfaces:
        # get number of layers
        if settings_synthetics.n_layers == 'auto':
            n_interfaces, n_layers = get_interface_number_fmt1(path_model_in)
        else:
            raise Exception("User-fixed layer number not yet supported. Number of layers should be auto")

        # read model
        df_vp_global, df_interfaces_global, dx_in, dy_in = \
            read_model_fmt1(path_model_in, n_interfaces)
        df_thickness_global = compute_thickness(df_interfaces_global, df_vp_global)

        # make manual edits as suggested by specialist of chémery
        df_vp_global = edit_velocities_as_suggested_by_catherine(df_vp_global)

        # select only points where all the horizons are well defined
        df_interfaces_valid = df_interfaces_global[~df_interfaces_global.isnull().any(axis=1)]
        df_thickness_valid = df_thickness_global[~df_interfaces_global.isnull().any(axis=1)]
        df_vp_valid = df_vp_global[~df_interfaces_global.isnull().any(axis=1)]

        # interpolate on desired grid
        if settings_synthetics.bounds_mode=='auto':
            xmin = df_vp_valid['X'].min()
            ymin = df_vp_valid['Y'].min()
            xmax = df_vp_valid['X'].max()
            ymax = df_vp_valid['Y'].max()
        elif settings_synthetics.bounds_mode=='manual':
            xmin = settings_synthetics.xmin
            ymin = settings_synthetics.ymin
            xmax = settings_synthetics.xmax
            ymax = settings_synthetics.ymax
        else:
            raise Exception('unknown bounds_mode value')

        df_vp_valid = remove_outliers(df_vp_valid, std_thresh=2)

        df_vp_interp = interpolate_model_per_layer(df_vp_valid, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                                         n_cells=settings_synthetics.n_cells,
                                                         lateral_smooth=settings_synthetics.lateral_smooth,
                                                         smooth_length=settings_synthetics.smooth_length,
                                                         dx_in=dx_in, dy_in=dy_in)
        df_interfaces_interp = interpolate_model_per_layer(df_interfaces_valid, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                                         n_cells=settings_synthetics.n_cells,
                                                         lateral_smooth=settings_synthetics.lateral_smooth,
                                                         smooth_length=settings_synthetics.smooth_length,
                                                         dx_in=dx_in, dy_in=dy_in)

        df_thickness_interp = compute_thickness(df_interfaces_interp, df_vp_interp)

        df_vp_interp = replace_velocities_by_mean_values_except_layer1(df_vp_interp)
        df_vs_interp, df_vp_over_vs_interp = apply_vp_over_vs_ratio_on_dataframe(df_vp_interp,
                                                           vp_over_vs_ratio=settings_synthetics.vp_over_vs,
                                                           n_layers_vp_over_vs=settings_synthetics.n_layers_vp_over_vs)

        if settings_synthetics.plot_interfaces:
            save_model_plots(df_interfaces_interp, 'interface',
                             settings_synthetics.n_cells, settings_synthetics.n_cells,
                             str(Path(make_synthetics.path_out_format_1)))

        if settings_synthetics.plot_thicknesses:
            save_model_plots(df_thickness_interp, 'thickness',
                             settings_synthetics.n_cells, settings_synthetics.n_cells,
                             str(Path(make_synthetics.path_out_format_1)))

        if settings_synthetics.plot_velocities:
            save_model_plots(df_vp_interp, 'vs',
                             settings_synthetics.n_cells, settings_synthetics.n_cells,
                             str(Path(make_synthetics.path_out_format_1)))

        # select only points where all values are well defined
        # df_thickness_valid = df_thickness_interp[~df_thickness_interp.isnull().any(axis=1)]
        # df_vp_valid = df_vp_interp[~df_thickness_interp.isnull().any(axis=1)]


    # compute dispersion curves
    file_out = str(Path(make_synthetics.path_out_format_1).joinpath(make_synthetics.file_out_format_1))
    nb_f = int(np.ceil((settings_synthetics.f_stop - settings_synthetics.f_start)/settings_synthetics.f_step))+1
    if recompute_dispersion:
        dispersion_dict = loop_on_cells(df_vp_interp, df_thickness_interp,
                                        settings_synthetics.vp_over_vs,
                                        settings_synthetics.n_layers_vp_over_vs,
                      settings_synthetics.f_start, settings_synthetics.f_stop, nb_f,
                      settings_synthetics.wavetype, settings_synthetics.modes,
                      settings_synthetics.velocity_mode, settings_synthetics.ny,
                      settings_synthetics.vel_last_layer,
                                        bool_compare_to_cps=settings_synthetics.compare_cps)

        dict_h5_list = save_h5(dispersion_dict, file_out)
        with open(file_out + '.pickle', 'wb') as f1:
            pickle.dump(dispersion_dict, f1, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_out + '.pickle', 'rb') as f1:
            dispersion_dict = pickle.load(f1)

    if settings_synthetics.plot_dispersion_maps:
        save_dispersion_plots(dispersion_dict, 'mode_0',
                         settings_synthetics.n_cells, settings_synthetics.n_cells,
                         str(Path(make_synthetics.path_out_format_1)))
        save_dispersion_plots(dispersion_dict, 'mode_0',
                         settings_synthetics.n_cells, settings_synthetics.n_cells,
                         str(Path(make_synthetics.path_out_format_1)),
                              plot_mode='absolute')

        if settings_synthetics.compare_tomo:
            save_dispersion_plots(dispersion_dict, 'mode_0',
                             settings_synthetics.n_cells, settings_synthetics.n_cells,
                             str(Path(make_synthetics.path_out_format_1)),
                              compare_tomo=settings_synthetics.compare_tomo,
                              tomo_folder=settings_synthetics.tomo_folder,
                                  plot_mode='relative')

            save_dispersion_plots(dispersion_dict, 'mode_0',
                             settings_synthetics.n_cells, settings_synthetics.n_cells,
                             str(Path(make_synthetics.path_out_format_1)),
                              compare_tomo=settings_synthetics.compare_tomo,
                              tomo_folder=settings_synthetics.tomo_folder,
                                  plot_mode='absolute')

        # if settings_synthetics.compare_tomo:
        #     save_dispersion_plots_along_with_tomo_outputs(dispersion_dict, 'mode_0',
        #                      settings_synthetics.n_cells, settings_synthetics.n_cells,
        #                      str(Path(make_synthetics.path_out_format_1)), n_skip=1,
        #                                                   tomo_folder=settings_synthetics.tomo_folder,
        #                                                   plot_mode='relative')
        #
        #     save_dispersion_plots_along_with_tomo_outputs(dispersion_dict, 'mode_0',
        #                      settings_synthetics.n_cells, settings_synthetics.n_cells,
        #                      str(Path(make_synthetics.path_out_format_1)), n_skip=1,
        #                                                   tomo_folder=settings_synthetics.tomo_folder,
        #                                                   plot_mode='absolute')

    if settings_synthetics.plot_cross_sections & reload_interfaces:
        for x_section in settings_synthetics.x_cross_sections:
            # if settings_synthetics.compare_tomo:
            #     save_cross_section_plots_along_with_tomo_outputs(df_interfaces_interp, dispersion_dict,
            #                                                      settings_synthetics.n_cells, settings_synthetics.n_cells,
            #                                                      str(Path(make_synthetics.path_out_format_1)),
            #                                                      fixed_coord='x', coord_val=x_section,
            #                                                      delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
            #                                                      mode_field='mode_0',
            #                                                          dispersion_curves_tomo_file=Path(settings_synthetics.dispersion_curves_tomo_file))


            save_cross_section_plots(df_interfaces_interp, df_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='x', coord_val=x_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer/settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
                                     data_type='vs',
                                     compare_tomo=settings_synthetics.compare_tomo,
                                     dispersion_curves_tomo_file=Path(settings_synthetics.dispersion_curves_tomo_file),
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vs_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='x', coord_val=x_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer/settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
                                     data_type='vs', plot_model_cross_section=False,
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vp_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='x', coord_val=x_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer,
                                     mode_field='mode_0', delta_x_disp_curves=None,
                                     data_type='vp',
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vp_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vp_over_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='x', coord_val=x_section,
                                     vel_last_layer=settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=None,
                                     data_type='vp_over_vs',
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vp_over_vs_bounds)
        for y_section in settings_synthetics.y_cross_sections:
            # if settings_synthetics.compare_tomo:
            #     save_cross_section_plots_along_with_tomo_outputs(df_interfaces_interp, dispersion_dict,
            #                                                      settings_synthetics.n_cells, settings_synthetics.n_cells,
            #                                                      str(Path(make_synthetics.path_out_format_1)),
            #                                                      fixed_coord='y', coord_val=y_section,
            #                                                      delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
            #                                                      mode_field='mode_0',
            #                                                      dispersion_curves_tomo_file=Path(settings_synthetics.dispersion_curves_tomo_file))
            save_cross_section_plots(df_interfaces_interp, df_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='y', coord_val=y_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer/settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
                                     data_type='vs',
                                     compare_tomo=settings_synthetics.compare_tomo,
                                     dispersion_curves_tomo_file=Path(settings_synthetics.dispersion_curves_tomo_file),
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vs_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='y', coord_val=y_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer/settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=settings_synthetics.plot_disp_curve_every_km,
                                     data_type='vs', plot_model_cross_section=False,
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vp_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='y', coord_val=y_section,
                                     vel_last_layer=settings_synthetics.vel_last_layer,
                                     mode_field='mode_0', delta_x_disp_curves=None,
                                     data_type='vp',
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vp_bounds)
            save_cross_section_plots(df_interfaces_interp, df_vp_over_vs_interp, dispersion_dict,
                                     settings_synthetics.n_cells, settings_synthetics.n_cells,
                                     str(Path(make_synthetics.path_out_format_1)),
                                     fixed_coord='y', coord_val=y_section,
                                     vel_last_layer=settings_synthetics.vp_over_vs[-1],
                                     mode_field='mode_0', delta_x_disp_curves=None,
                                     data_type='vp_over_vs',
                                     ylim_disp_curves=settings_synthetics.disp_curves_bounds,
                                     clim_model=settings_synthetics.vp_over_vs_bounds)
