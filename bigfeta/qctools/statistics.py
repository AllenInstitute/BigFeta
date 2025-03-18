import numpy


def get_sided_masked_array(d, side=None, axis=None, mask_stat="median"):
    """mask array to include values greater than ("right") or less than ("left") a statistic of the set

    Parameters
    ----------
    d : :class:`numpy.ndarray`
        input data array
    side : str or None
        "l", "r", or None will choose which values to mask relative to the mask_stat
            -- less than, greater than, or no masking, respectively
    axis : int
        axis on which mask_stat should be calculated
    mask_stat : str
        string "mean" or "median" representing numpy method to determine central value

    Returns
    -------
    m : :class:`numpy.ma.masked_array`
        masked array with "right" or "left" values unmasked        
    """
    mask_stat_switch = {
        "median": numpy.median,
        "mean": numpy.mean
    }

    mask_stat = mask_stat_switch[mask_stat]
    
    if side == "l":
        m = d <= mask_stat(d, axis=axis)
    elif side == "r":
        m = d >= mask_stat(d, axis=axis)
    elif side is None:
        return d
    else:
        raise ValueError("ERROR: please enter 'l', 'r', or None")
    return numpy.ma.masked_array(d, m)


def mad(data, axis=None, side=None, **kwargs):
    """Median Absolute Distance function with left and right side option.

    """
    d = get_sided_masked_array(data, side, axis)
    res = numpy.ma.median(numpy.absolute(d - numpy.ma.median(d, axis)), axis)
    return (res.data if isinstance(res, numpy.ma.masked_array) else res)


def get_mad(data, *args, lr_sum=False, **kwargs):
    """wrapper to get MAD statistic, including option of summing left and right-sided

    """
    if lr_sum:
        return (mad(data, *args, side="l", **kwargs) +
                mad(data, *args, side="r", **kwargs))
    return mad(data, *args, **kwargs)


def distance_filter_2d(data, std_filter=3, dist_filter=None, **kwargs):
    """filter array to exclude outliers by distance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   

    """
    dist_filter = (
        (std_filter * data.std(axis=0))
        if dist_filter is None else dist_filter)
    return data[
        numpy.all(numpy.abs(data.mean(axis=0) - data) < dist_filter, axis=1)
    ]


def get_2d_filtered_mad(data, *args, **kwargs):
    """get MAD on data filtered for outliers

    """
    filtered_data = distance_filter_2d(data, **kwargs)
    return get_mad(filtered_data, **kwargs)
