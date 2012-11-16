import numpy as np

from nipy.labs.datasets import VolumeImg


def resample(source, target, interpolation='continuous', return_affine=False):
    source, source_affine = source
    target, target_affine = target

    input_image = VolumeImg(source[:],
                            source_affine,
                            'arbitrary',
                            interpolation=interpolation
                            )

    resampled_image = input_image.as_volume_img(target_affine, target.shape)

    if return_affine:
        return resampled_image.get_data(), resampled_image.get_affine()
    else:
        return resampled_image.get_data()


def atlas_mean(R_voxels, weights):
    return np.mean(np.multiply(R_voxels, weights), axis=1)


def check_float_approximation(P, mask):
    lmap = {}
    for label in np.unique(P[mask]):
        lmap[label] = np.round(label)

    for label in lmap:
        P[P == label] = lmap[label]
