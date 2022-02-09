from pathlib import Path
import pandas as pd

percentage_uncert = 0.05

folder_dispersion_files = Path(r'/home/alex/EvoDCinv/examples/data')
folder_out = Path(r'/home/alex/EvoDCinv/examples/data_processed')
if not folder_out.exists():
    folder_out.mkdir()
dipsersion_file_gen = folder_dispersion_files.glob('*.txt')
for file in dipsersion_file_gen:
    file_out = folder_out.joinpath(file.parts[-1])
    df_disp_curve = pd.read_csv(folder_dispersion_files.joinpath(file), sep="\s+|;|:|,", header=None)
    df_disp_curve.iloc[:, -1] = percentage_uncert * df_disp_curve.iloc[:, -2]
    df_disp_curve.to_csv(file_out, index=False, header=False, sep=' ')
