"""Tools for RFI flagging.

This module contains the CHORD-specific tools for finding and removing Radio
Frequency Interference (RFI). The general, telescope-agnostic flagging
algorithms are in `draco` (`draco.util.rfi`, `draco.analysis.flagging`); the
pipeline tasks that apply them to CHORD data are in
`chord_pipeline.analysis.flagging`.

Masks are True where the data contain RFI (or are to be excluded) and False
elsewhere.

Static mask

- :py:data:`BAD_FREQUENCIES`
- :py:meth:`frequency_mask`

Times to treat with care

- :py:meth:`guess_fwhm`
- :py:meth:`daytime_flag`
- :py:meth:`transit_flag`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from caput.astro import time as ctime
from caput.astro.skyfield import unix_to_skyfield_time

if TYPE_CHECKING:
    from caput.astro.observer import Observer

# Speed of light in m/s
_C = 299792458.0

# Ranges of bad frequencies, given by their start and end times (UNIX time, None
# for no limit) and their start and end frequencies (MHz).
#
# The "chord" list is the static mask made from the first CHORD data at DRAO,
# starting from CHIME's (ch_util.rfi.BAD_FREQUENCIES["chime"]). The bands
# commented out are CHIME bands not seen in the CHORD data.
BAD_FREQUENCIES = {
    "chord": [
        # Mobile services
        [[None, None], [356.0, 390.0]],
        # CHIME: bad bands at first light
        [[None, None], [449.41, 450.98]],
        [[None, None], [454.88, 456.05]],
        [[None, None], [457.62, 459.18]],
        [[None, None], [483.01, 485.35]],
        # UHF TV channel 17, broadcast from Omak
        [[None, None], [487.70, 494.34]],
        # UHF TV channel 19, broadcast from Omak
        # [[None, None], [497.85, 506.05]],
        # UHF TV channel 24, broadcast from Omak
        [[None, None], [529.10, 536.52]],
        # CHIME: narrow, high power bands, and sporadic RFI between them
        # [[None, None], [460.15, 460.55]],
        # [[None, None], [464.00, 470.32]],
        # CHIME: 6 MHz band
        # [[None, None], [505.85, 511.71]],
        # Bright band, present since CHIME's early days
        [[None, None], [517.97, 525.00]],
        # UHF TV channels 30 and 31, broadcast from Omak
        [[None, None], [564.65, 578.00]],
        # UHF TV channel 33
        [[None, None], [584.00, 590.00]],
        # UHF TV channel 35
        [[None, None], [596.00, 602.00]],
        # CHIME: low power band visible in long baselines
        # [[None, None], [602.00, 607.82]],
        # 600 MHz mobile bands: Rogers, Bell, Freedom
        [[None, None], [617.00, 627.00]],
        [[None, None], [627.00, 637.00]],
        [[None, None], [637.00, 642.00]],
        # Seen in the first CHORD data
        [[None, None], [642.00, 652.00]],
        # 700 and 800 MHz cellular (LTE / 5G NR) bands
        [[None, None], [717.0, 728.0]],  # supplemental downlink, n29
        [[None, None], [729.0, 745.12]],  # downlink, n12 / n85
        [[None, None], [746.0, 756.0]],  # downlink, n13
        [[None, None], [758.0, 768.0]],  # downlink, n14
        [[None, None], [824.0, 849.0]],  # uplink, n5
        [[None, None], [869.0, 894.0]],  # downlink, n5
        # Aircraft secondary surveillance radar / Mode S: 1030 MHz
        # interrogations and 1090 MHz replies (ADS-B)
        [[None, None], [1030.0, 1031.0]],
        [[None, None], [1089.0, 1092.0]],
        # Navigation and GNSS, and the 23 cm amateur band
        [[None, None], [1135.0, 1315.0]],
    ],
}


def frequency_mask(
    freq_centre: np.ndarray,
    freq_width: np.ndarray | float | None = None,
    timestamp: np.ndarray | float | None = None,
    instrument: str = "chord",
) -> np.ndarray:
    """Flag known bad frequencies.

    Parameters
    ----------
    freq_centre
        Centre of each frequency channel in MHz.
    freq_width
        Width of each frequency channel in MHz. If `None` (default), calculate
        the width from the separation of the channel centres. If an array, it
        must be broadcastable against `freq_centre`.
    timestamp
        UNIX observing time. If `None` (default), mask all the bands whatever
        their start and end times, otherwise mask only the bands in use at these
        times. If an array, it must be broadcastable against `freq_centre`.
    instrument
        Name of the instrument in :py:data:`BAD_FREQUENCIES`. Default "chord".

    Returns
    -------
    mask
        True for the bad frequency channels. The shape is the result of
        broadcasting `freq_centre` and `timestamp` together.
    """
    freq_centre = np.asarray(freq_centre)
    if freq_width is None:
        freq_width = np.abs(np.median(np.diff(freq_centre)))

    freq_start = freq_centre - freq_width / 2
    freq_end = freq_centre + freq_width / 2

    mask = np.zeros(np.broadcast(freq_centre, timestamp).shape, dtype=bool)

    try:
        bad_freq = BAD_FREQUENCIES[instrument]
    except KeyError as e:
        raise ValueError(f"No RFI flags defined for {instrument}") from e

    for (start_time, end_time), (fs, fe) in bad_freq:
        # A channel is bad if it overlaps the band
        fmask = (freq_end > fs) & (freq_start < fe)

        if timestamp is None:
            tmask = True
        else:
            tmask = np.ones_like(timestamp, dtype=bool)
            if start_time is not None:
                tmask &= timestamp >= start_time
            if end_time is not None:
                tmask &= timestamp <= end_time

        mask |= tmask & fmask

    return mask


def guess_fwhm(
    freq: float | np.ndarray,
    dec: float | None = None,
    sigma: bool = False,
    voltage: bool = False,
    seconds: bool = False,
    diameter: float = 6.0,
) -> float | np.ndarray:
    """Estimate the width of the primary beam of a CHORD dish.

    Assumes a diffraction-limited circular dish, whose power beam has a full
    width at half maximum of about 1.02 lambda / D.

    Parameters
    ----------
    freq
        Frequency in MHz.
    dec
        Declination of the source in radians. If given, divide by cos(dec) to
        convert from an angle on the sky to an hour angle.
    sigma
        Return the standard deviation of a Gaussian beam instead of the FWHM.
    voltage
        Return the width of the voltage beam instead of the power beam.
    seconds
        Return the time taken to drift through the width, in seconds, instead
        of degrees.
    diameter
        Dish diameter in metres. Default 6 m.

    Returns
    -------
    width
        Beam width in degrees, or seconds if `seconds` is True.
    """
    wavelength = _C / (np.asarray(freq, dtype=np.float64) * 1e6)
    width = np.degrees(1.02 * wavelength / diameter)

    if dec is not None:
        width = width / np.cos(dec)

    if seconds:
        width = width * 240.0 * ctime.SIDEREAL_S

    if voltage:
        width = width * np.sqrt(2.0)

    if sigma:
        width = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    return width


def _observer(obs: Observer | None) -> Observer:
    if obs is None:
        from chord_ephem.observers import chord as obs
    return obs


def daytime_flag(time: np.ndarray, obs: Observer | None = None) -> np.ndarray:
    """Flag the times between sunrise and sunset.

    Parameters
    ----------
    time : np.ndarray[ntime]
        UNIX times.
    obs : Observer, optional
        The observer. Default CHORD.

    Returns
    -------
    flag : np.ndarray[ntime]
        True for times during the day.
    """
    obs = _observer(obs)
    time = np.atleast_1d(time)
    flag = np.zeros(time.size, dtype=bool)

    for rise in obs.solar_rising(time.min() - 24.0 * 3600.0, time.max()):
        sets = obs.solar_setting(rise)[0]
        flag |= (time >= rise) & (time <= sets)

    return flag


def transit_flag(
    body,
    time: np.ndarray,
    nsigma: float = 2.0,
    freq: float | np.ndarray = 400.0,
    obs: Observer | None = None,
) -> np.ndarray:
    """Flag the times near the transit of a celestial body.

    Parameters
    ----------
    body : skyfield body
        Skyfield representation of a celestial body, e.g. from
        `chord_ephem.sources.source_dictionary`, or the Sun from
        `caput.astro.skyfield.skyfield_wrapper.ephemeris["sun"]`.
    time : np.ndarray[ntime]
        UNIX times.
    nsigma : float
        Number of beam sigmas to flag on either side of transit.
    freq : float or np.ndarray[nfreq]
        Frequency in MHz at which to evaluate the beam width.
    obs : Observer, optional
        The observer. Default CHORD.

    Returns
    -------
    flag : np.ndarray[ntime] or np.ndarray[nfreq, ntime]
        True for times within `nsigma` of a transit. Two dimensional if `freq`
        is an array.
    """
    obs = _observer(obs)
    time = np.atleast_1d(time)

    if np.isscalar(freq):
        flag = np.zeros(time.size, dtype=bool)
    else:
        freq = np.asarray(freq)
        flag = np.zeros((freq.size, time.size), dtype=bool)
        freq = freq[:, np.newaxis]

    transits = obs.transit_times(
        body, time[0] - 24.0 * 3600.0, time[-1] + 24.0 * 3600.0
    )

    for ttrans in transits:
        sf_time = unix_to_skyfield_time(ttrans)
        pos = obs.skyfield_obs().at(sf_time).observe(body)

        alt = pos.apparent().altaz()[0]
        dec = pos.cirs_radec(sf_time)[1]

        # Only if the body is above the horizon
        if alt.radians > 0.0:
            window = nsigma * guess_fwhm(
                freq, dec=dec.radians, sigma=True, seconds=True
            )
            flag |= (time >= ttrans - window) & (time <= ttrans + window)

    return flag
