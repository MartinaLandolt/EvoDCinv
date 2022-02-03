# -*- coding: utf-8 -*-

from pathlib import Path
import h5py
import configparser
import os
import numpy as np
import ast
from itertools import takewhile


def h5_to_dict(filename):
    with h5py.File(filename, "r") as fin:
        field_list = [key for key in fin.keys()]
        dict_out = dict()
        for variable in field_list:
            dict_out[variable] = fin[variable][()]
            attr_keys = fin[variable].attrs.keys()
            attr_list = [attr for attr in attr_keys]
            if len(attr_list) > 0:
                dict_out[variable + '_attrs'] = dict()
                for attr in attr_list:
                    dict_out[variable+'_attrs'][attr] = fin[variable].attrs[attr]

    return dict_out


def read_tomo_files_h5(folder, pattern):
    file_list_gen = Path(folder).rglob(pattern)
    dict_global = dict()
    for file in file_list_gen:
        dict_file = h5_to_dict(file)
        freq = dict_file['velocity_anomaly_attrs']['frequency']
        dict_global["{:.2f}".format(freq)] = dict_file
    # settings_path_gen = Path(folder).rglob('*settings.ini')
    # settings_list = [settings_path for settings_path in settings_path_gen]
    # assert len(settings_list) == 1
    # settings_path = settings_list[0]
    # ###Read config file
    # config = configparser.ConfigParser(inline_comment_prefixes=";")
    # config.read(settings_path)
    #
    # ########## BEGIN VARIABLE UNWRAPPING ###########


    return dict_global


if __name__ == "__main__":
    filename_h5 = Path(r'/home/alex/EvoDCinv/make_synthetics/tomo_outputs_reg10/dispersion_curves.h5')
    dict_disp_curves_tomo = h5_to_dict(str(filename_h5))
    foldername_h5 = Path(r'/home/alex/EvoDCinv/make_synthetics/tomo_outputs_reg10')
    pattern = 'pass_2*'
    dict_tomo_global = read_tomo_files_h5(foldername_h5, pattern)