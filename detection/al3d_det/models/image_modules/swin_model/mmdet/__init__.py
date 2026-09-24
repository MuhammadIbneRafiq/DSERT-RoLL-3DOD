import mmcv

from .version import __version__, short_version


def digit_version(version_str):
    digit_version = []
    for x in version_str.split('.'):
        if x.isdigit():
            digit_version.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            digit_version.append(int(patch_version[0]) - 1)
            digit_version.append(int(patch_version[1]))
    return digit_version


mmcv_minimum_version = '1.2.4'
mmcv_maximum_version = '1.4.0'
mmcv_version = digit_version(mmcv.__version__)


# DSERT only uses the Swin backbone + FPN neck from this vendored mmdet, which
# also work with newer mmcv 1.x (e.g. 1.7.2 on PyTorch 2.x), so warn instead of assert.
if not (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)):
    import warnings
    warnings.warn(f'MMCV=={mmcv.__version__} is outside the tested range '
                  f'[{mmcv_minimum_version}, {mmcv_maximum_version}].')

__all__ = ['__version__', 'short_version']
