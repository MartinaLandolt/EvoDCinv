import configparser
import os
from pathlib import *
import ast

settings_extension='*.ini'

src_dir = Path(__file__).parent.resolve()
settings_path_gen = Path(src_dir).glob(settings_extension)
settings_file_list = []
for file in settings_path_gen:
    settings_file_list.append(file)
    
if len(settings_file_list)==0:
    raise OSError("no settings file found")
if len(settings_file_list)>1:
    raise OSError("multiple settings files found")
if not settings_file_list[0].exists():
    raise OSError("path to settings does not exist")

settings_path = settings_file_list[0]

# Read config file
config = configparser.ConfigParser(inline_comment_prefixes=";")
config.read(settings_path)

# Start unwrapping
path_paths = config["PATHS"]
type_vel_model = int(path_paths["type_vel_model"])

path_vel = config["VELOCITY_MODEL"]
vp_over_vs = int(path_vel["vp_over_vs"])
bounds_mode = str(path_vel["bounds_mode"])
xmin = float(path_vel["xmin"])
ymin = float(path_vel["ymin"])
cell_size = float(path_vel["cell_size"])
n_cells = int(path_vel["n_cells"])
xmax = xmin + cell_size * n_cells
ymax = ymin + cell_size * n_cells

path_f_band = config["FREQUENCY_BAND"]
f_step = float(path_f_band["BinFrequencyEvery"])
f_start =float(path_f_band["FirstBin"])
f_stop = float(path_f_band["LastBin"])

path_w_type = config["WAVE_TYPE"]
wavetype = str(path_w_type["wavetype"])
velocity_mode = str(path_w_type["velocity_mode"])
modes = ast.literal_eval(path_w_type["modes"])

path_solver_params = config["SOLVER_PARAMS"]
n_layers = str(path_solver_params["n_layers"])
ny = int(path_solver_params["ny"])
