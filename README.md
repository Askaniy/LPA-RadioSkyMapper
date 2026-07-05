# LPA-RadioSkyMapper

This repository contains scripts for processing archival data from the Large Phased Array (LPA) radio telescope. It performs observations at a frequency of 110 MHz in transit instrument mode, scanning the sky along the celestial meridian.

- `mapping.py` constructs sky maps for a specified date range. It performs calibration of raw time series, resampling, and transforms the coordinate grid to the J2000 epoch.
- `stacking.py` performs statistical analysis of the maps and saves "anomaly maps" for transient searching.

Execution is intended for the Pushchino Radio Astronomy Observatory (AKTs FIAN) server.
Results are saved as color PNG images for visual analysis and as NumPy arrays for further processing.


## Usage (CLI)

Tested on Linux. It is recommended to install the `uv` package manager.

**Installation:**
```bash
git clone https://github.com/Askaniy/LPA-RadioSkyMapper.git
cd LPA-RadioSkyMapper
uv sync
```

**Basic Syntax:**
```bash
uv run mapping.py <DATE1> <DATE2> [OPTIONS]
uv run stacking.py <DATE1> <DATE2> [OPTIONS]
```

**Arguments:**
* `DATE1` (required): Start date of the observation interval in `YYYY-MM-DD` format.
* `DATE2` (required): End date in `YYYY-MM-DD` format.
* `--data_path` (optional): Path to archival data (default: `/bsa_b/`)
* `--workers` (optional): Number of worker processes for data reading (default: 12)

**Example Launch:**
```bash
uv run mapping.py 2013-10-11 2026-07-01
uv run stacking.py 2013-10-11 2016-05-26
```


## Technical Details

### Cartographic System

A simple cylindrical projection (3600×1800 pixels) is used.
Declination coverage: from −9.25° to +55.25° (total 64.5°).
Right Ascension coverage: from 360° to 0°.

**Coordinate to Pixel:**

$$x = 3599.5 - \alpha \cdot 10$$

$$Y = (90^\circ - \delta) \cdot 10 - 0.5$$

$$y = (55.3^\circ - \delta) \cdot 10 - 0.5$$

**Pixel to Coordinate:**

$$\alpha = \frac{3599.5 - x}{10}$$

$$\delta = 90^\circ - \frac{Y + 0.5}{10} = 55.3^\circ - \frac{y + 0.5}{10}$$

**Longitude Mapping ($\alpha$):**
| $x$ | $\alpha$ |
| :--- | :--- |
| 0 | 359.95 |
| 3599 | 0.05 |

**Latitude Mapping ($\delta$):**
| $Y$ | $y$ | $\delta$ |
| :--- | :--- | :--- |
| 0 | | $89.95^\circ$ |
| 347 | 0 | $55.25^\circ$ |
| 992 | 644 | $-9.25^\circ$ |
| 1799 | | $-89.95^\circ$ |


### Time Processing Architecture

LPA saves observations hourly in PNTR format files, separate for each of the three recorders.
The script uses a multiprocessing architecture with shared memory (`SharedMemory`), which is filled and read cyclically:
1. **Recorder worker processes**: Up to 12 processes; their task is to process a specific hour for a specific recorder and write the result into a common buffer. Before writing to the ring buffer, the read data is compressed by 27x using median filtering. Calibration steps that interrupt the data stream every 4 hours are measured in these processes and recorded in their own shared memory block. Calibration steps are removed from the data after reading.
2. **Mapping worker processes**: 4 processes that collect accumulated data from the common buffer, binning them into a pixel grid (with ~5x compression), performing declination interpolation (Cubic Spline), applying coordinate rotation to the J2000 epoch, compressing by 4x, and generating the final map image. The trigger for starting map generation is the presence of two calibration steps—one before and one after the observation interval—ensuring high-quality interpolation of calibration data.
3. **Main process**: An orchestrator that distributes tasks in 4-hour chunks to ensure maximum CPU utilization.

Map boundaries are determined by the time of culmination of a point ($\alpha=0$, $\delta=0$), calculated using the `astroplan` library taking Earth's rotation model into account.
When saving, the filename includes a modified Modified Julian Date (MJD) of the map end interval, which corresponds to the left edge.
Thus, the time on the map is calculated as $T = MJD + \alpha_{source}$.
Five decimal places provide redundant precision (1 s), while the grid step is approximately 24 s.


### Spatial-Spectral Processing

LPA performs continuous sky monitoring in 128 beams, non-uniformly distributed across declinations; each beam is split into bundles of 6 frequency channels from 109.21 to 111.29 MHz.
Near the edges of the field of view, the splitting is significant enough that frequency channels of different beams overlap in declination.
Therefore, during mapping, spectral decomposition within beams cannot be neglected.
On the contrary, this effect is used to increase resolution along the declination axis: maps for different channels are processed individually.
In this manner, they are saved as NumPy arrays, and for visualization, six channels are compressed into three (RGB):

```math
\mathbf{v}_{\text{RGB}}
=
\frac{1}{6} 
\begin{bmatrix}
    3 & 2 & 1 & 0 & 0 & 0 \\
    0 & 1 & 2 & 2 & 1 & 0 \\
    0 & 0 & 0 & 1 & 2 & 3
\end{bmatrix}
\cdot
\mathbf{v}_{6\text{ channels}}
```

This technique is called "false color processing" and helps highlight artifacts and other data errors.
On the resulting map, real radio sources most often have a neutral gray color due to the narrow spectral range of LPA.
In turn, many observational artifacts are "colored," as non-genuine flux often does not repeat across different channels.


### Anomaly Maps

To facilitate visual searching for transients, "anomaly maps" were constructed, representing the ratio of observed flux to reference flux: $r = F_{obs} / F_{ref}$.
To distinguish between flux increases and decreases, gray color ($c = 255 / 2$) corresponds to an unchanged flux.
Setting a logarithmic scale and boundaries for distinguishable $r$: $r_{max} = 5$, $r_{min} = 1 / r_{max}$. Then the formula linking anomaly map brightness $c$ and the flux ratio $r$ is:
```math
c = \frac{255}{2} \left(1 + \log_{5}{r} \right)
```

Accordingly, $r$ can be estimated from the map:
```math
r = 5^{\frac{2 \, c}{255} - 1}
```


### Flux Calibration

Every 4 hours, a noise signal (calibration step) with a fixed brightness temperature of 2400 K (accuracy 2–3% (L)) and a "closed shutter" brightness level (D) is recorded into the LPA observation time series.
Readings are interpolated using a cubic spline to calibrate the time series via the formula:
```math
F_{obs} = \frac{F_{raw} - D}{L - D}
```

The profile of a step in the time series point indices looks as follows:
- From 3006 to 3008, the shutter closes
- From 3008 to 3056, the shutter is closed
- From 3056 to 3058, the calibration source turns on
- From 3058 to 3106, the calibration source is on
- From 3106 to 3108, the calibration source turns off
- From 3108 to 3156, the shutter is closed
- From 3156 to 3158, the shutter opens
