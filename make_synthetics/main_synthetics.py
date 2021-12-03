import make_synthetics
from make_synthetics import settings_synthetics
from pathlib import Path


def read_model_fmt1():
    """read layers and velocities
    """

    pass


def get_layer_number_fmt1(path_model):

    pass


def make_1d_model_for_cell():
    pass


def get_dispersion_curve():
    pass


def loop_on_cells():
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
        n_layers = get_layer_number_fmt1(path_model_in)
    else:
        raise Exception("User-fixed layer number not yet supported. Number of layers should be auto")
