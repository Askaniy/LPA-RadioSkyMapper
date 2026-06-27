# System
from pathlib import Path
# Interface and code
import re
from typing import cast
from argparse import ArgumentParser
# Math and arrays
import numpy as np
import numpy.typing as npt
# Astronomic calculations
from astropy.time import Time
# Image processing
from PIL import Image
from astropy.stats import sigma_clipped_stats


# === Console Input Processing ===

parser = ArgumentParser(
    prog='LPA maps stacking script',
    description='Combining maps derived from archival LPA data',
    epilog='Askaniy Anpilogov, aaskaniy@gmail.com'
)

parser.add_argument('date1', type=str, help='Start date of observation interval in YYYY-MM-DD format')
parser.add_argument('date2', type=str, help='End date in YYYY-MM-DD format')

args = parser.parse_args()


# === Data Locations ===

save_path = Path(__file__).resolve().parent

# - Data loading path
arrays_path = save_path/'arrays'

# - Result saving paths
stacks_path = save_path/'stacks'
#diffs_path = save_path/'diffs'


# === Script Settings ===

# - Results
stat_names = ('mean', 'median', 'stddev')

# - Map parameters
width = 3600
height = 645
channels = 6

# - Maximum brightness in calibration step units
br_max_preview = 2

# - Color procession
# Matrix to compress 6 channels into 3 colors
rgb_matrix = np.array((
    (3, 2, 1, 0, 0, 0),
    (0, 1, 2, 2, 1, 0),
    (0, 0, 0, 1, 2, 3)
)) / 6



def gamma_correction(arr0: npt.NDArray) -> npt.NDArray:
    """ Applies gamma correction in sRGB implementation to the array """
    arr1 = np.copy(arr0)
    mask = arr0 < 0.0031308
    arr1[mask] *= 12.92
    arr1[~mask] = 1.055 * np.power(arr1[~mask], 1./2.4) - 0.055
    return arr1

def main_process(start_date: str, end_date: str):

    # Date validation
    if start_date == end_date:
        raise ValueError('Specified observation interval is empty!')
    try:
        mjd0 = cast(float, Time(start_date, format='iso').mjd)
        mjd1 = cast(float, Time(end_date, format='iso').mjd) + 1
    except ValueError as e:
        raise ValueError(f'Date format error: {e}')
    if mjd0 > mjd1:
        raise ValueError(f'Observation interval cannot start ({start_date}) later than it ends ({end_date})')

    # File names contain MJD epoch at the end of the map
    # Regex to find the float value after 'array_' and before '.npz'
    mjd_pattern = re.compile(r'array_([\d.]+)\.npz')

    # Collecting the list of maps within the specified time range
    files = []
    for file in arrays_path.glob('array_*.npz'):
        match = mjd_pattern.search(file.name)
        if match:
            mjd = float(match.group(1))
            if mjd0 < mjd - 0.5 <= mjd1:
                files.append(file)

    n_maps = len(files)
    if n_maps == 0:
        raise FileNotFoundError(f'No maps found within the time range of {mjd0:.5f}—{mjd1:.5f} MJD')

    # Calculate array size
    shape = (n_maps, height, width, channels)
    memory_gb = np.prod(shape) * np.dtype(np.float32).itemsize / 1024**3

    # User confirmation
    print(f'- The processing requires approximately {memory_gb * 7:.2f} GB of RAM')
    input('  Press Enter to continue...')

    # Pre-allocate the array
    all_maps = np.empty(shape, dtype=np.float32)

    # Filling the array
    for i, file in enumerate(files):
        with np.load(file) as data:
            all_maps[i] = data['data']

    # Compute statistics along axis 0
    results = sigma_clipped_stats(all_maps, sigma=3.0, maxiters=5, axis=0) # (3, 1800, 3600, 6)

    # Save statistics array
    dates = start_date + '-' + end_date
    np.savez_compressed(arrays_path/f'stack_stats_{dates}.npz', mean=results[0], median=results[1], stddev=results[2])

    # Save statistic maps
    for stat, stat_name in zip(results, stat_names):
        # Uniform compression of six channels into three colors
        # Each color accounts for one-third of the total flow
        rgb = np.einsum('ij, klj -> kli', rgb_matrix, stat) / br_max_preview
        # Compress into standard color depth
        rgb = np.round(gamma_correction(np.clip(rgb, 0, 1)) * 255).astype(np.uint8)
        # Save map image
        Image.fromarray(rgb).save(stacks_path/f'stack_{stat_name}_{dates}.png')


# === Program Entry Point ===

if __name__ == '__main__':

    # Basic data existence checks
    if not arrays_path.is_dir():
        raise FileNotFoundError(f'{arrays_path} directory does not exist!')
    elif not any(arrays_path.iterdir()):
        raise FileNotFoundError(f'{arrays_path} directory has no data to process!')
    else:
        # Check / create path for results
        stacks_path.mkdir(parents=True, exist_ok=True)
        #diffs_path.mkdir(parents=True, exist_ok=True)
        main_process(args.date1, args.date2)
