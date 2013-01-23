from collections import namedtuple

import numpy as np
import pylab as pl
import nibabel as nb

import nipy.labs.viz as viz

from .utils import resample, atlas_mean, check_float_approximation


class Atlas(object):

    def __init__(self, label_image, affine, label_map=None):
        self.label_image = label_image
        self.affine = affine.astype('float32')
        self.label_map = label_map
        if len(label_image.shape) == 3:   # single region in atlas
            self.label_image = label_image[..., None]
        self.mask = (label_image != 0.).sum(3).astype('bool')
        self.shape = self.label_image.shape[:-1]
        self.size = self.label_image.shape[-1]

        if self.label_image.dtype.name != 'bool':
            self.label_image = self.label_image.astype('float32')

    def resample(self, voxels_size):
        affine_3x3 = np.array([[-1., 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]) * voxels_size
        mock_vol = namedtuple('Volume', ['shape'])(None)

        self.label_image, self.affine = resample(
            (self.label_image, self.affine),
            (mock_vol, affine_3x3),
            'nearest', True)

        self.shape = self.label_image.shape[:-1]
        self.mask = (self.label_image != 0.).sum(3).astype('bool')

    def labels(self):
        return range(self.size)

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def fit(self, mask, affine):
        if self.label_image.dtype.name == 'bool':
            self.label_image_ = resample((self.label_image, self.affine),
                                         (mask, affine), 'nearest')
        else:
            self.label_image_ = resample((self.label_image, self.affine),
                                         (mask, affine))
        self.label_image_ = np.rollaxis(self.label_image_, 3)
        self.mask_ = mask
        self.affine_ = affine
        self.r_masks_ = [
            np.logical_and((region != 0.).astype('bool'), self.mask_)
            for region in self.label_image_]

    def transform(self, X, pooling_func=atlas_mean):
        nX = np.zeros((X.shape[0], self.size))
        for i, (r_mask, region) in enumerate(
                zip(self.r_masks_, self.label_image_)):
            nX[:, i] += pooling_func(X[:, r_mask[self.mask_]], region[r_mask])
        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)
        for i, (r_mask, region) in enumerate(
                zip(self.r_masks_, self.label_image_)):
            nX[:, r_mask[self.mask_]] += X[:, i, None]
        return nX

    def apply(self, X, pooling_func=atlas_mean):
        is_array = False
        if len(X.shape) == 3:
            is_array = True

        if is_array:
            nX = self.transform(X[self.mask_][np.newaxis, :], pooling_func)
        else:
            nX = self.transform(X, pooling_func)

        nX = self.inverse_transform(nX)

        if is_array:
            new_array = np.zeros(self.mask_.shape)
            new_array[self.mask_] = nX[0, :]
            return new_array
        else:
            return nX

    def iter_extract(self, X, weight_func=np.multiply):
        for i, (r_mask, region) in enumerate(
                zip(self.r_masks_, self.label_image_)):
            label = i + 1
            r_val = np.zeros((X.shape[0], region.size))
            self.r_mask_ = r_mask
            if weight_func is not None:
                r_val = weight_func(region[r_mask], X[:, r_mask[self.mask_]])
            else:
                r_val = X[:, r_mask[self.mask_]]
            yield label, r_val

    def extract(self, X, weight_func=np.multiply):
        return dict(self.iter_extract(X))

    def overlapping(self):
        regions = {}

        for i, region in enumerate(np.rollaxis(self.label_image, 3)):
            overlaps = self.label_image[region.astype('bool'), :].sum(axis=0)
            overlaps = (np.where(overlaps != 0)[0]).tolist()
            if overlaps != []:
                overlaps.remove(i)
            regions.setdefault(i, overlaps)

        return regions

    def discard(self, *labels):
        labels = list(labels)
        mask = np.ones(self.size, dtype='bool')
        mask[labels] = False
        self.label_image = self.label_image[..., mask]

        if self.label_map is not None:
            label_map = {}

            for label in labels:
                del self.label_map[label]

            for i, (k, v) in enumerate(self.label_map.iteritems()):
                if not k in labels:
                    label_map[i] = v

            self.label_map = label_map

        self.mask = (self.label_image != self.null_label).sum(3).astype('bool')
        self.shape = self.label_image.shape[:-1]
        self.size = self.label_image.shape[-1]

    def update(self, regions, names=None):
        self.label_image = np.concatenate((self.label_image, regions), axis=3)

        if self.label_map is not None and names is not None:
            self.label_map.update(
                dict(zip(range(self.size, self.size + len(names)), names)))

        self.mask = (self.label_image != 0.).sum(3).astype('bool')
        self.shape = self.label_image.shape[:-1]
        self.size = self.label_image.shape[-1]

    def add(self, region, name=None):
        if name is None:
            self.update(region)
        else:
            self.update(region, [name])

    def difference(self, label1, label2, discard_operands=False):
        mask = self.label_image[..., label2].astype('bool')

        if self.label_image.dtype.name == 'bool':
            diff = np.copy(self.label_image[..., label1])
            diff[mask] = False
        else:
            diff = np.copy(self.label_image[..., label1])
            diff[mask] -= self.label_image[mask, label2]
            diff[diff < 0] = 0

        if self.label_map is not None:
            name = '%s - %s' % (self.label_map[label1],
                                self.label_map[label2])
            self.add(diff, name)
        else:
            self.add(diff)

        if discard_operands:
            self.discard(label1, label2)

    def union(self, label1, label2, discard_operands=False):
        mask = np.logical_or(self.label_image[..., label1].astype('bool'),
                             self.label_image[..., label2].astype('bool'))

        if self.label_image.dtype.name == 'bool':
            union = np.zeros(self.shape, dtype='bool')
            union[mask] = True
        else:
            union = np.zeros(self.shape, dtype='float32')
            union[mask] = self.label_image[mask, label1] + \
                self.label_image[mask, label2]

        if self.label_map is not None:
            name = '%s + %s' % (self.label_map[label1],
                                self.label_map[label2])
            self.add(union, name)
        else:
            self.add(union)

        if discard_operands:
            self.discard(label1, label2)

    def intersection(self, label1, label2, discard_operands=False):
        mask = np.logical_and(self.label_image[..., label1].astype('bool'),
                              self.label_image[..., label2].astype('bool'))

        if self.label_image.dtype.name == 'bool':
            inter = np.zeros(self.shape, dtype='bool')
            inter[mask] = True
        else:
            inter = np.zeros(self.shape, dtype='float32')
            inter[mask] = self.label_image[mask, label1] + \
                self.label_image[mask, label2]

        if self.label_map is not None:
            name = '%s = %s' % (self.label_map[label1],
                                self.label_map[label2])
            self.add(inter.astype(self.label_image.dtype), name)
        else:
            self.add(inter.astype(self.label_image.dtype))

        if discard_operands:
            self.discard(label1, label2)

    def subset(self, labels):
        label_map = None
        if self.label_map is not None:
            label_map = {}
            [label_map.setdefault(i, self.label_map[i]) for i in labels]

        return self.__class__(self.label_image[..., labels],
                              self.affine,
                              label_map)

    def to_parcellation(self, **options):
        label_map = options.get('label_map')
        parc_image = np.zeros(self.shape, dtype='float32')

        if self.label_image.dtype.name != 'bool':
            threshold = options.get('threshold', .25)
            label_image_max = np.max(self.label_image, axis=3)

            for label in self.labels():
                mask = np.logical_and(
                    self.label_image[..., label] > threshold,
                    self.label_image[..., label] == label_image_max)
                parc_image[mask] = label + 1

            return Parcellation(parc_image, self.affine, label_map, 0)
        else:
            parc_image = np.zeros(self.shape, dtype='float32')

            for label in self.labels():
                R_mask = self.label_image[..., label]
                parc_image[R_mask] = label + 1

            return Parcellation(parc_image, self.affine, label_map, 0.)

    def show(self, label=None, rcmap=None, **options):
        self.label_image = np.array(self.label_image)
        if label is not None:
            color = rcmap or 'black'
            slicer = viz.plot_map(self.label_image[..., label],
                                  self.affine, **options)
            slicer.contour_map(self.mask, self.affine,
                               levels=[0], colors=(color, ))
            return slicer
        else:
            slicer = viz.plot_map(self.mask, self.affine, **options)
            for i, label in enumerate(self.labels()):
                color = rcmap(1. * i / self.size) if rcmap is not None \
                    else pl.cm.gist_rainbow(1. * i / self.size)
                slicer.contour_map(
                    self.label_image[..., label],
                    self.affine, levels=[0],
                    colors=(color, ))
            return slicer

    def save(self, location):
        img = nb.Nifti1Image(self.label_image.astype('float32'), self.affine)
        nb.save(img, location)


class Parcellation(object):

    def __init__(self, label_image, affine, label_map=None, null_label=0):
        self.label_image = label_image.astype('float32')
        self.affine = affine.astype('float32')
        self.null_label = null_label
        self.label_map = label_map
        self.mask = (label_image != null_label).astype('bool')
        self.shape = label_image.shape
        self.size = np.unique(label_image[self.mask]).size
        check_float_approximation(self.label_image, self.mask)

    def labels(self):
        return np.unique(self.label_image[self.mask]).astype('int').tolist()

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def fit(self, mask, affine):
        self.label_image_ = resample((self.label_image, self.affine),
                               (mask, affine), 'nearest')
        self.mask_ = mask
        self.affine_ = affine
        self.r_masks_ = [
            np.logical_and(self.label_image_ == label, self.mask_)
            for label in self.labels()]

    def transform(self, X, pooling_func=np.mean):
        nX = np.zeros((X.shape[0], self.size))
        for i, (r_mask, label) in enumerate(zip(self.r_masks_, self.labels())):
            nX[:, i] = pooling_func(X[:, r_mask[self.mask_]], axis=1)
        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)
        for i in range(len(self.r_masks_)):
            nX[:, self.r_masks_[i][self.mask_]] = X[:, i, np.newaxis]
        return nX

    def iter_extract(self, X):
        for label, r_mask in zip(self.labels(), self.r_masks_):
            yield label, X[:, r_mask[self.mask_]]

    def extract(self, X):
        return dict(self.iter_extract(X))

    def apply(self, X, pooling_func=np.mean):
        is_array = False
        if len(X.shape) == 3:
            is_array = True

        if is_array:
            nX = self.transform(X[self.mask_][np.newaxis, :], pooling_func)
        else:
            nX = self.transform(X, pooling_func)

        nX = self.inverse_transform(nX)

        if is_array:
            new_array = np.zeros(self.mask_.shape)
            new_array[self.mask_] = nX[0, :]
            return new_array
        else:
            return nX

    def union(self, label1, label2):
        self.label_image[self.label_image == label1] = label2
        self.size = np.unique(self.label_image[self.mask]).size

        if self.label_map is not None:
            self.label_map[label2] = '%s + %s' % (self.label_map[label1],
                                                  self.label_map[label2])
            del self.label_map[label1]

    def subset(self, labels):
        label_image = np.ones(self.label_image.size) * self.null_label
        ind = np.where(np.in1d(self.label_image, labels))[0]
        label_image[ind] = self.label_image.ravel()[ind]
        return self.__class__(
            label_image.reshape(self.shape), self.affine, self.null_label)

    def to_atlas(self, *parcellations, **options):
        label_map = options.get('label_map')

        if parcellations is None:
            parcellations = []

        atlas_image = np.zeros(self.shape + (self.size, ), dtype='bool')
        atlas_image = np.rollaxis(atlas_image, 3)

        for i, label in enumerate(self.labels()):
            atlas_image[i, :] = self.label_image == label

        atlas_image = np.rollaxis(atlas_image, 0, 4)

        for parc in parcellations:
            if parc.shape != self.shape:
                parc.label_image = resample((parc.label_image, parc.affine),
                                  (self.label_image, self.affine), 'nearest')

            atlas_image = np.concatenate(
                (atlas_image, parc.to_atlas().atlas_image), axis=3)

        return Atlas(atlas_image, self.affine, label_map)

    def show(self, label=None, rcmap=None, **options):
        self.label_image = np.array(self.label_image)
        if label is None:
            return viz.plot_map(self.label_image, self.affine, **options)
        else:
            color = rcmap or 'black'
            slicer = viz.plot_map(self.label_image == label,
                                  self.affine, **options)
            slicer.contour_map(self.mask, self.affine,
                               levels=[0], colors=(color, ))
            return slicer

    def save(self, location):
        img = nb.Nifti1Image(self.label_image, self.affine)
        nb.save(img, location)
