# System
import os
import queue
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from time import sleep
from pathlib import Path
# Interface and code
import warnings
from typing import Generator, cast
from argparse import ArgumentParser
from traceback import format_exc
from tqdm import tqdm
# Math and arrays
import numpy as np
import numpy.typing as npt
from math import floor
# Astronomic calculations
import erfa
import datetime as dt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import Observer, FixedTarget
# Image processing
from PIL import Image
from scipy.interpolate import CubicSpline
from scipy.ndimage import map_coordinates


# === Console Input Processing ===

parser = ArgumentParser(
    prog='LPA mapping script',
    description='Sky mapping using archival LPA data',
    epilog='Askaniy Anpilogov, aaskaniy@gmail.com'
)

parser.add_argument('date1', type=str, help='Start date of observation interval in YYYY-MM-DD format')
parser.add_argument('date2', type=str, help='End date in YYYY-MM-DD format')
parser.add_argument('--data_path', type=str, default='/bsa_b/',
                    help='Root directory path for LPA data (default: /bsa_b/)')
parser.add_argument('--workers', type=int, default=6,
                    help='Number of parallel processes for reading recorder data (from 1 to 12)')

args = parser.parse_args()


# === Data Locations ===

# - LPA data path
data_path = Path(args.data_path)

# - Result saving paths
save_path = Path(__file__).resolve().parent
arrays_path = save_path/'arrays'
images_path = save_path/'images'
calibs_path = save_path/'calibs'


# === Script Settings ===

# - Maximum brightness in calibration step units
br_max_preview = 2   # almost all sources are dimmer
br_max_typical = 128 # only the Sun during flares is brighter

# - Map parameters
final_map_width = 3600
final_map_height = 645
prefinal_map_width = 2 * final_map_width
prefinal_map_height = 2 * final_map_height
prefinal_map_x = 0.5 + np.arange(prefinal_map_width)
prefinal_map_y = 0.5 + np.arange(prefinal_map_height)
prefinal_map_x_edges = np.arange(0, prefinal_map_width+1)

# - Data parameters
hour_width0 = 36018 # reference number of points per file
hour_width1 = 1334 # number of points after preliminary compression
hour_factor = 27 # compression factor
hour_width1_step = 1 / hour_width1 / 24 # new grid step in days

# - Calibration step parameters
calib_hours = (1, 5, 9, 13, 17, 21) # hours with calibration steps
calib_shift = 308 / (24 * 60 * 60) # shift of the step center relative to file start in days
t_calib = 2400 # Brightness temperature of a calibration step (K)

# - LPA parameters
lpa_longitude = 37.631363 * u.deg
lpa_latitude = 54.820710 * u.deg
lpa_elevation = 206 * u.m
lpa = Observer(latitude=lpa_latitude, longitude=lpa_longitude, elevation=lpa_elevation, timezone='UTC')
target = FixedTarget(SkyCoord(ra=0*u.deg, dec=0*u.deg, frame='icrs'), name='Zero Meridian')

# - Beam parameters
n_beams = 128
beam_slices = (
    (0, 32),
    (32, 80),
    (80, 128)
)
n_regs = len(beam_slices)
range_regs = range(n_regs)
n_channels = 6
range_channels = range(n_channels)

# - Color procession
# Matrix to compress 6 channels into 3 colors
rgb_matrix = np.array((
    (3, 2, 1, 0, 0, 0),
    (0, 1, 2, 2, 1, 0),
    (0, 0, 0, 1, 2, 3)
)) / 6

# - Multiprocessing parameters
n_reg_workers = args.workers # from 1 to 12
range_reg_workers = range(n_reg_workers) # worker pool
hours_per_chunk = 4
n_reg_works = hours_per_chunk * n_regs
range_reg_works = range(n_reg_works)
n_map_workers = 2
range_map_workers = range(n_map_workers)

# - Shared memory parameters
# Maximum map buffer size — 36 hours:
# 24h map + 8h calibration step + 4h buffer
shared_map_width = hour_width1 * 36
shared_map_shape = (shared_map_width, n_beams, n_channels)
f32size = np.dtype(np.float32).itemsize
shared_map_size = cast(int, np.prod(shared_map_shape) * f32size) # in bytes

# - Math
tau = 2 * np.pi


# === Helper Functions ===

def RA_to_x(ra, width):
    """ Right ascension (deg) -> Horizontal map coordinate (px center) """
    return width * (1 - ra / 360)

def x_to_RA(x, width):
    """ Horizontal map coordinate (px center) -> Right ascension (deg) """
    return (1 - x / width) * 360

def dec_to_y(dec, width):
    """ Declination (deg) -> Vertical map coordinate in the LPA's field of view (px center) """
    return (55.25 - dec) / 360 * width

def y_to_dec(y, width):
    """ Vertical map coordinate in the LPA's field of view (px center) -> Declination (deg) """
    return 55.25 - y / width * 360

def dec_to_Y(dec, width):
    """ Declination (deg) -> Vertical map coordinate (px center) """
    return (90 - dec) / 360 * width

def Y_to_dec(Y, width):
    """ Vertical map coordinate (px center) -> Declination (deg) """
    return 90 - Y / width * 360

def read_pntr(file: Path|str) -> npt.NDArray:
    """
    LPA data file parser. Returns an array of shape [time, beam, channel].
    Ensures the number of measurements is exactly hour_width0=36018 points!
    """
    head = {}
    with open(file, 'rb') as f:
        for _ in range(16):
            line = f.readline()
            a, *b = line.decode('utf-8').strip('\n').split()
            head[a] = b
        npoints = int(head['npoints'][0])
        # Should be approximately 36018 points per hour.
        # However, for example, on June 14, 2024, there were 25796 points per hour.
        data = np.fromfile(f, dtype=np.float32)
        data = data.reshape(npoints, len(head['modulus'])*8, len(head['fbands'])+1)
        if npoints == hour_width0:
            # Correct length of data
            return data
        elif npoints > hour_width0:
            # Truncate data
            return data[:hour_width0]
        elif npoints < hour_width0:
            # Fill missing data points with NaN values
            filled = np.full((hour_width0,) + data.shape[1:], np.nan, dtype=np.float32)
            filled[:npoints] = data
            return filled
        raise ValueError('Undefined data length') # for type checker

def stretch(arr: npt.NDArray, times: int | tuple, copy=False):
    """
    Adds dimensions to the end of the array and repeats it there.
    Uses broadcast by default for memory-efficiency. Uses np.tile() for copying.
    """
    if isinstance(times, int):
        times = (times,)
    new_axes = len(times) * (np.newaxis,)
    if copy:
        return np.tile(arr[..., *new_axes], (*((1,) * len(arr.shape)), *times))
    else:
        return np.broadcast_to(arr[..., *new_axes], (*arr.shape, *times))

def linear_interp(x0: npt.NDArray, y0: npt.NDArray, x1: npt.NDArray):
    """
    Analog of `np.interp(x1, x0, y0)`, working for multidimensional data along the first axis.
    `x0` must be sorted!
    """
    idx = np.searchsorted(x0, x1).clip(0, x0.size-1)
    x_left = x0[idx - 1]
    y_left = y0[idx - 1]
    delta_x = x0[idx] - x_left
    delta_y = y0[idx] - y_left
    slopes = delta_y.T / delta_x
    y1 = y_left + (slopes * (x1 - x_left)).T
    return y1

def gamma_correction(arr0: npt.NDArray) -> npt.NDArray:
    """ Applies gamma correction in sRGB implementation to the array """
    arr1 = np.copy(arr0)
    mask = arr0 < 0.0031308
    arr1[mask] *= 12.92
    arr1[~mask] = 1.055 * np.power(arr1[~mask], 1./2.4) - 0.055
    return arr1


# === Class Declarations ===

class Epoch:
    """ Stores observation time information """

    def __init__(self, date_iso: str, hour: int, date_obj: dt.date | None = None):
        self.date_iso: str = date_iso
        self.date_obj: dt.date = dt.date.fromisoformat(date_iso) if date_obj is None else date_obj
        self.hour = hour

    def to_Time(self) -> Time:
        """
        Calculates epoch from observation date and hour. LPA uses UTC+5 timezone, returns time in UTC.
        Tests showed inaccuracy in coordinate determination by observation time, requiring additional shift.
        """
        return Time(self.date_iso) + (self.hour - 5) * u.hour # + 360 * u.s

    @staticmethod
    def generator(start_obj: dt.datetime, end_obj: dt.datetime) -> Generator:
        """
        Generator of intermediate hours between two dates in ISO format.
        """
        date_obj = start_obj
        while date_obj <= end_obj:
            # Format date once per cycle (24 times fewer strftime calls)
            date_iso = date_obj.strftime('%Y-%m-%d')
            for hour in range(24):
                yield Epoch(date_iso, hour, date_obj)
            # Move to the next day
            date_obj += dt.timedelta(days=1)

class DailyBuffer:
    """
    Accumulates calibration step data and determines when it is time to generate a map.
    """

    def __init__(self, epoch0_mjd: float, calib_mjds: npt.NDArray):
        # List of calibration step epochs
        self.calib_mjds = calib_mjds
        # Counter for measured calibration steps
        self.last_step_idx = -1
        # Declaration of the first map boundaries:
        epoch0 = Time(epoch0_mjd, format='mjd')
        self.last_transit = lpa.target_meridian_transit_time(epoch0, target, which='previous')
        self.next_transit = lpa.target_meridian_transit_time(epoch0, target, which='next')
        # Calculation of index reference points and map time
        self.start_idx = floor((cast(float, self.last_transit.mjd) - epoch0_mjd) / hour_width1_step) + 1
        self.start_time = epoch0_mjd + self.start_idx * hour_width1_step

    def get_next_transit(self, anchor_time: Time) -> Time:
        """ Calculates the next map boundary split by meridian RA=0 """
        # An arbitrary shift is added to ensure distinction between culminations
        return lpa.target_meridian_transit_time(anchor_time + 1 * u.hour, target, which='next')

    def update_last_step(self):
        """ Updates index of the last read calibration step """
        self.last_step_idx += 1

    def try_map_collecting(self):
        """
        Checks if conditions for map generation are met.
        If yes, returns indices of the global map array and the global calibration step array.
        """
        # Two calibration steps after the map range step must be known for high-quality extrapolation
        # (We ignore that the very first map will not be perfectly calibrated)
        if self.last_step_idx >= 1:
            # Define MJD boundaries for potential map
            mjd0 = cast(float, self.last_transit.mjd)
            mjd1 = cast(float, self.next_transit.mjd)
            # Penultimate calibration step must be a future one
            if self.calib_mjds[self.last_step_idx - 1] >= mjd1:
                # Define interval included in the map
                map_idx0 = self.start_idx + floor((mjd0 - self.start_time) / hour_width1_step)
                map_idx1 = self.start_idx + floor((mjd1 - self.start_time) / hour_width1_step)
                # Update next map boundary
                self.last_transit = self.next_transit
                self.next_transit = self.get_next_transit(anchor_time=self.next_transit)
                # Define interval of calibration steps
                # Need 10 = 2 steps before the map + 6 on the map + 2 after
                calib_idx1 = self.last_step_idx + 1
                calib_idx0 = max(0, calib_idx1 - 10)
                return mjd0, mjd1, map_idx0, map_idx1, calib_idx0, calib_idx1


# === Main functions for parallel processes ===

def map_worker(
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        shared_map_memory_name: str,
        shared_calib_memory_name: str,
        shared_calib_shape: tuple,
        epoch0_mjd: float,
        calib_mjds: npt.NDArray
    ):
    """
    Process performing data resampling: binning the accumulated interval into a map,
    declination interpolation, and saving.
    """
    # Connect to existing shared memory by name
    shared_map_memory = SharedMemory(name=shared_map_memory_name)
    shared_calib_memory = SharedMemory(name=shared_calib_memory_name)
    shared_map = np.ndarray(shared_map_shape, dtype=np.float32, buffer=shared_map_memory.buf)
    shared_calib = np.ndarray(shared_calib_shape, dtype=np.float32, buffer=shared_calib_memory.buf)
    # Calculate pixel coordinates of the beams
    beam_decs = np.loadtxt('beam_declinations.tsv').T # (6, 128)
    beam_pixels = dec_to_y(beam_decs, prefinal_map_width)
    # Precompute J2000 coordinate grid for interpolation
    ra = np.radians(x_to_RA(prefinal_map_x, prefinal_map_width))
    dec = np.radians(y_to_dec(prefinal_map_y, prefinal_map_width))
    rra, ddec = np.meshgrid(ra, dec)
    cos_ddec = np.cos(ddec)
    xyz_J2000 = np.stack(
        (
            cos_ddec * np.cos(rra),
            -cos_ddec * np.sin(rra), # `rot_matrix` works right only with the sign
            np.sin(ddec)
        ),
        axis=0
    )
    while True:
        # Wait for task
        task = task_queue.get()
        if task is None:  # Sentinel signal
            break
        # Unpacking: epoch range, indices of the shared arrays
        mjd0, mjd1, map_idx0, map_idx1, calib_idx0, calib_idx1 = task
        try:
            # Retrieve a slice of the map from the buffer
            # Change coordinate time direction: when looking from inside the celestial sphere, RA decreases
            # Performed with high precision to avoid rounding errors in np.cumsum()
            map_indices = np.arange(map_idx0, map_idx1)
            y0 = np.flip(np.take(shared_map, map_indices, mode='wrap', axis=0), axis=0).astype(dtype=np.float64)

            # Define time coordinates on the map
            map_times = epoch0_mjd + (0.5 + map_indices) * hour_width1_step
            map_scale = prefinal_map_width / (mjd1 - mjd0) # map scale
            x0 = map_scale * (mjd1 - np.flip(map_times, axis=0))
            step_coords = map_scale * (mjd1 - np.flip(calib_mjds[calib_idx0:calib_idx1], axis=0))

            # If all values for t are NaN, remove t (usually — remainder of a calibration step)
            # If there are NaNs but not all are NaN, replace with zeros
            # Without NaNs, cumulative sum won't work
            nan_mask = np.all(np.isnan(y0), axis=(1, 2))
            x0 = x0[~nan_mask]
            y0 = np.nan_to_num(y0[~nan_mask])

            # --- Binning ---
            # Auxiliary array of time derivative
            x0_diff = stretch(np.diff(x0), y0.shape[1:]).astype(dtype=np.float64)
            # Cumulative integral
            y_cdf = np.zeros(y0.shape, dtype=np.float64)
            y_cdf[1:] = np.cumsum(0.5 * (y0[:-1] + y0[1:]) * x0_diff, axis=0) # Riemann sum
            # Binning the cumulative distribution
            arr = np.diff(linear_interp(x0, y_cdf, prefinal_map_x_edges), axis=0).astype(np.float32)

            # Interpolation and calibration by steps
            calib_light = np.flip(shared_calib[calib_idx0:calib_idx1, ..., 0], axis=0)
            calib_dark = np.flip(shared_calib[calib_idx0:calib_idx1, ..., 1], axis=0)
            calib_lights = CubicSpline(step_coords, calib_light, bc_type='natural')(prefinal_map_x).astype(np.float32)
            calib_darks = CubicSpline(step_coords, calib_dark, bc_type='natural')(prefinal_map_x).astype(np.float32)
            calib_lights -= calib_darks
            arr = (arr - calib_darks) / calib_lights

            # Swap axes: for interpolation and Pillow, height must come first, then width
            arr = arr.swapaxes(0, 1)

            # Round MJD to slightly excessive precision ~1s (grid step ~24s)
            mjd1_str = str(round(mjd1, 5))

            # Interpolate map channels in height across all pixel values
            # Gamma correction as a temporary solution to help the spline
            # Should try ridge regression (Tikhonov regularization with identity matrix)
            arr = np.clip(np.nan_to_num(arr) / br_max_typical, 0, 1) ** (1/3)
            final_map = np.empty((prefinal_map_height, prefinal_map_width, n_channels), dtype=np.float32)
            for channel in range_channels:
                final_map[..., channel] = CubicSpline(beam_pixels[channel], arr[..., channel], bc_type='natural')(prefinal_map_y)
            final_map = br_max_typical * final_map**3 # fast inverse gamma correction

            # Compensate for precession and nutation
            # A rotation matrix converts J2000 coordinates to the current mean epoch
            mean_epoch = Time((mjd0 + mjd1) / 2, format='mjd')
            rot_matrix = erfa.pnm06a(mean_epoch.tt.jd1, mean_epoch.tt.jd2)
            xyz_on_epoch = np.einsum('ij, jkl -> ikl', rot_matrix, xyz_J2000)
            rra_on_epoch = np.arctan2(-xyz_on_epoch[1], xyz_on_epoch[0]) % tau # do not remove %!
            ddec_on_epoch = np.arcsin(xyz_on_epoch[2])

            # There is a RA shift, which increases with the declination of the sources
            # I hypothesize a model in which the observation plane is slightly tilted relative to the celestial meridian
            # The coordinates are shifted along the trajectory of the great circle's projection onto a cylindrical map
            polar_angle = np.radians(0.32)
            rra_on_epoch = rra_on_epoch - np.arcsin(np.tan(ddec) * np.tan(polar_angle))

            # Warped reprojection for each channel
            xx_on_epoch = RA_to_x(np.degrees(rra_on_epoch), prefinal_map_width)
            yy_on_epoch = dec_to_y(np.degrees(ddec_on_epoch), prefinal_map_width)
            for channel in range_channels:
                final_map[..., channel] = map_coordinates(
                    final_map[..., channel],
                    (yy_on_epoch, xx_on_epoch),
                    order=2,
                    mode='wrap',
                    prefilter=False,
                )

            # Median filtering, 4x compression
            # After reprojection, the map becomes less clear
            # Therefore, it is performed at a higher resolution so that the map can then be compressed again
            final_map = np.mean(final_map.reshape(final_map_height, 2, final_map_width, 2, n_channels), axis=(1, 3))

            # Save map array
            np.savez_compressed(arrays_path/f'map_{mjd1_str}.npz', data=final_map)

            # Uniform compression of six channels into three colors
            # Each color accounts for one-third of the total flow
            final_map = np.einsum('ij, klj -> kli', rgb_matrix, final_map) / br_max_preview

            # Compress into standard color depth
            final_map = np.round(gamma_correction(np.clip(final_map, 0, 1)) * 255).astype(np.uint8)

            # Save map image
            Image.fromarray(final_map).save(images_path/f'map_{mjd1_str}.png')

            # Inform main process that worker finished the map
            result_queue.put((True, mjd1_str))

        except Exception:
            result_queue.put((False, format_exc()))

def reg_worker(
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        shared_map_memory_name: str,
        shared_calib_memory_name: str,
        shared_calib_shape: tuple
    ):
    """ Universal worker for a single recorder data file. """
    # Connect to existing shared memory by name
    shared_map_memory = SharedMemory(name=shared_map_memory_name)
    shared_calib_memory = SharedMemory(name=shared_calib_memory_name)
    shared_map = np.ndarray(shared_map_shape, dtype=np.float32, buffer=shared_map_memory.buf)
    shared_calib = np.ndarray(shared_calib_shape, dtype=np.float32, buffer=shared_calib_memory.buf)

    while True:
        # Wait for task
        task = task_queue.get()
        if task is None:  # Sentinel signal
            break

        # Unpack task
        i_epoch: int = task[0]
        epoch: Epoch = task[1]
        n_reg: int = task[2]

        # Define height interval on the map
        idx_y0, idx_y1 = beam_slices[n_reg]

        # Data indicator
        data = None

        try:
            # --- Process raw data from recorder ---
            day = epoch.date_obj # shortcut
            # Determine folder
            folder = str(data_path/f'N{n_reg}L')
            # Form file path
            base_name = folder + day.strftime('/%Y/%m/%d') + day.strftime('/%d%m%y') + f'_{epoch.hour:0>2}_N{n_reg}'
            file = None
            for i in range(10):
                # Iterate through file versions, select the last one
                file_name = base_name + f'_{i:0>2}.pnt'
                if os.path.isfile(file_name):
                    file = file_name
            if file is None:
                raise FileNotFoundError(f'Files of type {base_name}_**.pnt not found!')

            # Read recorder data, skip the last (aggregate) channel
            data = read_pntr(file)[..., :-1] # shape [time, beam, channel]

            # Measure calibration steps
            if epoch.hour in calib_hours:
                # Define index of the step in global array
                calib_idx = floor(i_epoch / 4)
                # 0 = calibration signal (light), 1 = shutter closed level (dark)
                shared_calib[calib_idx, idx_y0:idx_y1, ..., 0] = np.median(data[3060:3105], axis=0).astype(np.float32)
                shared_calib[calib_idx, idx_y0:idx_y1, ..., 1] = (np.median(data[3010:3055], axis=0) + np.median(data[3110:3155], axis=0)).astype(np.float32) / 2
                # Remove calibration step data
                data[3004:3161] = np.nan

            # Data compression: 0.1s resolution is too high for internal processing
            # In one hour, map width / 24 = ~150 px
            # For future binning, we want roughly 10 data points per pixel
            # This corresponds to 1500 data points, but we usually have 36018.
            # So we adjust data to 36018 and bin to 36018 / 27 = 1334 (~9 points per pixel)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                data = np.nanmedian(data.reshape(hour_width1, hour_factor, *data.shape[1:]), axis=1)

            # Report success
            result_queue.put((True, None))

        except Exception:
            # Report error
            result_queue.put((False, format_exc()))

        finally:
            # Write to shared_array at calculated indices
            # Even if an error occurs, we must overwrite the previous layer of the cyclic array with a template
            idx_x0 = (hour_width1 * i_epoch) % shared_map_width
            idx_x1 = (idx_x0 + hour_width1) % shared_map_width

            if idx_x1 < idx_x0:
                # Emulate cyclicity
                first_part_len = shared_map_width - idx_x0
                if data is None:
                    shared_map[idx_x0:, idx_y0:idx_y1].fill(0)
                    shared_map[:idx_x1, idx_y0:idx_y1].fill(0)
                else:
                    shared_map[idx_x0:, idx_y0:idx_y1] = data[:first_part_len]
                    shared_map[:idx_x1, idx_y0:idx_y1] = data[first_part_len:]
            else:
                if data is None:
                    shared_map[idx_x0:idx_x1, idx_y0:idx_y1].fill(0)
                else:
                    shared_map[idx_x0:idx_x1, idx_y0:idx_y1] = data

def main_process(start_date: str, end_date: str):
    """ Main process orchestrator """

    # Date validation
    if start_date == end_date:
        raise ValueError('Specified observation interval is empty!')
    try:
        start_date_obj = dt.datetime.fromisoformat(start_date)
        end_date_obj = dt.datetime.fromisoformat(end_date).replace(hour=23, minute=0, second=0, microsecond=0)
    except ValueError as e:
        raise ValueError(f'Date format error: {e}')
    if start_date_obj > end_date_obj:
        raise ValueError(f'Observation interval cannot start ({start_date}) later than it ends ({end_date})')

    # Convert epoch generator to a list
    all_epoches: tuple[Epoch, ...] = tuple(Epoch.generator(start_date_obj, end_date_obj))
    epoch0_mjd = cast(float, all_epoches[0].to_Time().mjd)
    n_hours = len(all_epoches)

    # Define parameters for all calibration steps
    # Epochs start at hour 0; hours with steps are at hour 1 every 4 hours
    calib_epoches: tuple[Epoch, ...] = all_epoches[1::4]
    calib_mjds = np.array([obj.to_Time().mjd for obj in calib_epoches]) + calib_shift
    calib_dates = np.array([obj.date_iso for obj in calib_epoches])
    calib_hours = np.array([obj.hour for obj in calib_epoches])

    # Create array of all calibration steps
    # [step number, beam, channel, light/dark]
    shared_calib_shape = (calib_mjds.size, n_beams, n_channels, 2)
    shared_calib_size = cast(int, np.prod(shared_calib_shape) * f32size) # in bytes
    shared_calib_memory = SharedMemory(create=True, size=shared_calib_size)
    shared_calib_array = np.ndarray(shared_calib_shape, dtype=np.float32, buffer=shared_calib_memory.buf)
    shared_calib_array.fill(0)

    # Create buffer to track map generation conditions
    daily_buffer = DailyBuffer(epoch0_mjd, calib_mjds)

    # Create unbinned map array shared across all processes
    shared_map_memory = SharedMemory(create=True, size=shared_map_size)
    shared_map_array = np.ndarray(shared_map_shape, dtype=np.float32, buffer=shared_map_memory.buf)
    shared_map_array.fill(0)

    # Create queues for recorder data readers
    reg_task_queue = mp.Queue()
    reg_result_queue = mp.Queue()

    # Create queues for map generators
    map_task_queue = mp.Queue()
    map_result_queue = mp.Queue()

    # Start map worker pool
    map_processes = []
    for i in range_map_workers:
        p = mp.Process(
            target=map_worker,
            args=(
                map_task_queue,
                map_result_queue,
                shared_map_memory.name,
                shared_calib_memory.name,
                shared_calib_shape,
                epoch0_mjd,
                calib_mjds
            ),
            name=f'MapWorker_{i}'
        )
        p.start()
        map_processes.append(p)

    # Start recorder worker pool
    reg_processes = []
    for i in range_reg_workers:
        p = mp.Process(
            target=reg_worker,
            args=(
                reg_task_queue,
                reg_result_queue,
                shared_map_memory.name,
                shared_calib_memory.name,
                shared_calib_shape,
            ),
            name=f'RegWorker_{i}'
        )
        p.start()
        reg_processes.append(p)

    all_processes = (*map_processes, *reg_processes)

    try:
        # Process in 4-hour chunks
        with tqdm(total=n_hours, desc='Number of processed hours') as pbar:
            for n_start in range(0, n_hours, hours_per_chunk):
                n_end = n_start + hours_per_chunk

                # Fill task queue for the current 4-hour chunk
                for i_epoch in range(n_start, n_end):
                    epoch = all_epoches[i_epoch]
                    for n_reg in range_regs:
                        reg_task_queue.put((i_epoch, epoch, n_reg))

                # Collect results for this chunk
                completed_works = 0

                while completed_works < n_reg_works:
                    # Check if any worker died
                    for i, p in enumerate(map_processes):
                        if not p.is_alive():
                            raise RuntimeError(f'Map process {i} crashed!')
                    for i, p in enumerate(reg_processes):
                        if not p.is_alive():
                            raise RuntimeError(f'Recorder process {i} crashed!')

                    # Poll queues
                    for _ in range_reg_works:
                        try:
                            success, results = reg_result_queue.get(timeout=0.01)
                            completed_works += 1
                            if not success:
                                pbar.write(f'! Process error: {results}')
                        except queue.Empty:
                            break

                    # Check map generation and saving status
                    for _ in range_map_workers:
                        try:
                            success, results = map_result_queue.get(timeout=0.01)
                            if success:
                                # results = mjd1_str
                                pbar.write(f'- Sky map up to MJD={results} successfully saved!')
                            else:
                                pbar.write(f'! Map generation process error: {results}')
                        except queue.Empty:
                            break

                    # Small pause to avoid overloading CPU in empty loop
                    sleep(0.01)

                # Update progress bar
                pbar.update(hours_per_chunk)

                # Update number of collected calibration steps
                daily_buffer.update_last_step()

                # Check requirements for map generation via buffer
                map_results = daily_buffer.try_map_collecting()

                if map_results is not None:
                    iso0 = Time(map_results[0], format='mjd').iso
                    iso1 = Time(map_results[1], format='mjd').iso
                    pbar.write(f'- Building sky map from {iso0} to {iso1}')

                    # Process and save the map
                    map_task_queue.put(map_results)

        # Send Sentinel signal as many times as there are workers in the pool
        for _ in range_map_workers:
            map_task_queue.put(None)
        for _ in range_reg_workers:
            reg_task_queue.put(None)

        print('Checking that map generation is complete...')

        # Wait for the MapWorkers
        while True:
            # Check if workers are still alive
            if all(not map_p.is_alive() for map_p in map_processes):
                break

            # Get the last messages
            for _ in range_map_workers:
                try:
                    success, results = map_result_queue.get(timeout=0.01)
                    if success:
                        # results = mjd1_str
                        print(f'- Sky map up to MJD={results} successfully saved!')
                    else:
                        print(f'! Map generation process error: {results}')
                except queue.Empty:
                    break

            # Small pause to avoid overloading CPU in empty loop
            sleep(0.01)

    except Exception as e:
        print(f'Critical error: {e}')
        raise # Re-raise exception

    finally:
        print('Shutting down processes...')

        # Send Sentinel signal as many times as there are workers in the pool
        for _ in range_map_workers:
            map_task_queue.put(None)
        for _ in range_reg_workers:
            reg_task_queue.put(None)

        # Save collected calibration information
        np.savez(
            calibs_path/f'calib_{start_date}-{end_date}.npz',
            dates=calib_dates, hours=calib_hours, calib=shared_calib_array.copy()
        )

        # If processes are still hanging, kill them forcibly
        for p in all_processes:
            if p.is_alive():
                print(f'Process {p.name} did not respond. Forcing termination...')
                p.terminate() # Fast kill
                p.join()      # Clean up OS resources

        # Cleanup calibration step memory
        shared_calib_memory.close()
        try:
            shared_calib_memory.unlink()
        except FileNotFoundError:
            pass

        # Cleanup map memory
        shared_map_memory.close()
        try:
            shared_map_memory.unlink()
        except FileNotFoundError:
            pass


# === Program Entry Point ===

if __name__ == '__main__':

    # Check / create paths for results
    arrays_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)
    calibs_path.mkdir(parents=True, exist_ok=True)

    main_process(args.date1, args.date2)
