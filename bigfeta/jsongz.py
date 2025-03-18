import gzip
import json
import os


def _convert_ext(filepath, compress):
    b, e = os.path.splitext(filepath)
    filepath = b + '.json'
    if compress:
        filepath += '.gz'
    return filepath


def _check_ext(filepath):
    compress = False
    b, e = os.path.splitext(filepath)
    if e == '.gz':
        compress = True
    return compress


def dump(obj, filepath, compress=None, encoding='utf-8', *args, **kwargs):
    """json or json.gz dump

    Parameters
    ----------
    obj : obj
        object to dump
    filepath : str
        path for destination of dump
    compress : bool or None
        if None, file compressed or not according to filepath extension
    encoding : str
        encoding of :py:func:`json.dumps` before writing to .gz file.
        not passed into :py:func:`json.dump`
    *args
        :py:func:`json.dump` args
    **kwargs
        :py:func:`json.dump` kwargs

    Returns
    -------
    filepath : str
        potentially modified filepath of dumped object
        uncompressed are forced to '.json' and compressed to '.gz'
    """

    if not compress:
        compress = _check_ext(filepath)
    else:
        filepath = _convert_ext(filepath, compress)
    if compress:
        with gzip.GzipFile(filepath, 'w') as f:
            f.write(json.dumps(obj, *args, **kwargs).encode(encoding))
    else:
        with open(filepath, 'w') as f:
            json.dump(obj, f, *args, **kwargs)
    return filepath


def load(filepath, encoding='utf-8', *args, **kwargs):
    """json or json.gz load

    Parameters
    ----------
    filepath : str
        path for source of load
    encoding : str
        encoding for decoding of :py:func:`json.dumps` after .gz read
        not passed into :py:func:`json.load`
    *args
        :py:func:`json.load` args
    **kwargs
        :py:func:`json.load` kwargs

    Returns
    -------
    obj : dict
        loaded object
    """

    compressed = _check_ext(filepath)
    if compressed:
        with gzip.GzipFile(filepath, 'r') as f:
            obj = json.loads(f.read().decode(encoding), *args, **kwargs)
    else:
        with open(filepath, 'r') as f:
            obj = json.load(f, *args, **kwargs)
    return obj
