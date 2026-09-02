# System
from pathlib import Path
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
# Interface and code
import re
from typing import cast
from argparse import ArgumentParser
from tqdm import tqdm
# Math and arrays
import numpy as np
import numpy.typing as npt
# Image processing
from PIL import Image


# === Console Input Processing ===

parser = ArgumentParser(
    prog='LPA maps animation script',
    description='Combining maps or their fragments into GIF',
    epilog='Askaniy Anpilogov, aaskaniy@gmail.com'
)

parser.add_argument('mjd1', type=float, help='Start date of observation interval in MJD')
parser.add_argument('mjd2', type=float, help='End date in MJD')
parser.add_argument('--workers', type=int, default=12,
                    help='Number of parallel processes for reading map arrays')
parser.add_argument('--duration', type=int, default=10, help='Animation time in seconds')
parser.add_argument('--crop', type=int, nargs=4,
    help='Left, upper, right, and lower pixel coordinate of rectangular region'
)


# === Data Locations ===

save_path = Path(__file__).resolve().parent

# - Data loading path
arrays_path = save_path/'arrays'

# - Result saving paths
animations_path = save_path/'animations'


# === Script Settings ===

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

def load_single_map(args_tuple: tuple):
    """
    Worker function to load a single .npz file into a shared memory buffer.
    args_tuple: (file_path, index, shared_array_memory_name, shared_array_shape, crop_box)
    """
    file_path, idx, memory_name, shape, crop_box = args_tuple
    left, upper, right, lower = crop_box
    # Connect to existing shared memory by name
    shared_array_memory = SharedMemory(name=memory_name)
    # Reconstruct the numpy array from shared memory
    shared_array = np.ndarray(shape, dtype=np.float32, buffer=shared_array_memory.buf)
    with np.load(file_path) as data:
        shared_array[idx] = data['data'][upper:lower, left:right]
    return idx

def main_process(mjd0: str, mjd1: str, n_workers: int, duration: int|float, crop_box: tuple):

    # Date validation
    if mjd0 > mjd1:
        raise ValueError(f'Observation interval cannot start ({start_date}) later than it ends ({end_date})')

    # File names contain MJD epoch at the end of the map
    # Regex to find the float value after 'map_' and before '.npz'
    mjd_pattern = re.compile(r'map_([\d.]+)\.npz')

    # Collecting the list of maps within the specified time range
    files = []
    mjds = []
    for file in arrays_path.glob('map_*.npz'):
        match = mjd_pattern.search(file.name)
        if match:
            mjd = match.group(1)
            if mjd0 < float(mjd) - 0.5 <= mjd1:
                files.append(file)
                mjds.append(mjd)

    n_maps = len(files)
    if n_maps == 0:
        raise FileNotFoundError(f'No maps found within the time range of {mjd0:.5f}—{mjd1:.5f} MJD')

    # Calculate array size
    left, upper, right, lower = crop_box
    shared_array_shape = (n_maps, lower-upper, right-left, channels)
    shared_array_size = cast(int, np.prod(shared_array_shape) * np.dtype(np.float32).itemsize) # in bytes

    # Allocate the shared array
    shared_array_memory = SharedMemory(create=True, size=shared_array_size)
    try:
        shared_array = np.ndarray(shared_array_shape, dtype=np.float32, buffer=shared_array_memory.buf)

        # Prepare arguments for workers: (path, index, shape, base_pointer)
        worker_args = [
            (file, i, shared_array_memory.name, shared_array_shape, crop_box)
            for i, file in enumerate(files)
        ]

        print(f'Loading {n_maps} maps using {n_workers} threads...')
        with mp.Pool(processes=n_workers) as pool:
            # Using imap/imap_unordered with tqdm to track progress
            list(tqdm(pool.imap_unordered(load_single_map, worker_args), total=n_maps, desc='Loading maps'))

        # Post-processing
        shared_array = np.einsum('ij, nklj -> nkli', rgb_matrix, shared_array) / br_max_preview
        shared_array = np.round(gamma_correction(np.clip(shared_array, 0, 1)) * 255).astype(np.uint8)

        print('Converting arrays to images...')
        frames = []
        for i in range(n_maps):
            frames.append(Image.fromarray(shared_array[i]))

        output_name = f'animation_{mjd0:.3f}-{mjd1:.3f}.gif'

        fps = n_maps / duration
        print(f'Saving animation with FPS={fps:.3f}')

        frame_duration = 100 / fps # to ms

        frames[0].save(
            animations_path/output_name,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=frame_duration,
            loop=0
        )

    except Exception as e:
        print(f'Critical error: {e}')
        raise # Re-raise exception

    finally:

        # Cleanup shared array memory
        shared_array_memory.close()
        try:
            shared_array_memory.unlink()
        except FileNotFoundError:
            pass


# === Program Entry Point ===

if __name__ == '__main__':

    # Basic data existence checks
    if not arrays_path.is_dir():
        raise FileNotFoundError(f'{arrays_path} directory does not exist!')
    elif not any(arrays_path.iterdir()):
        raise FileNotFoundError(f'{arrays_path} directory has no data to process!')
    else:
        args = parser.parse_args()
        # Check / create path for results
        animations_path.mkdir(parents=True, exist_ok=True)
        main_process(args.mjd1, args.mjd2, args.workers, args.duration, args.crop)
