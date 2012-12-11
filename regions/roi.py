from collections import namedtuple

import numpy as np
import pylab as pl
import nibabel as nb

import nipy.labs.viz as viz

from .utils import resample, atlas_mean, check_float_approximation


class Atlas(object):

    def __init__(self, A, affine, label_map=None):
        self.A = A
        self.affine = affine.astype('float32')
        self.label_map = label_map
        if len(A.shape) == 3:   # single region in atlas
            self.A = A[..., None]
        self.mask = (A != 0.).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]

        if self.A.dtype.name != 'bool':
            self.A = self.A.astype('float32')

    def resample(self, voxels_size):
        affine_3x3 = np.array([[-1., 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]]) * voxels_size
        fake_volume = namedtuple('Volume', ['shape'])(None)

        self.A, self.affine = resample((self.A, self.affine),
                                       (fake_volume, affine_3x3),
                                       'nearest', True)

        self.shape = self.A.shape[:-1]
        self.mask = (self.A != 0.).sum(3).astype('bool')

    def labels(self):
        return range(self.size)

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def transform(self, X, affine, mask, pooling_func=atlas_mean):
        A = resample((self.A, self.affine), (mask, affine), 'nearest')
        A = np.rollaxis(A, 3)

        nX = np.zeros((X.shape[0], self.size))

        for i, region in enumerate(A):
            R_mask = np.logical_and((region != 0.).astype('bool'), mask)
            nX[:, i] += pooling_func(X[:, R_mask[mask]], region[R_mask])

        self.A_ = A
        self.mask_ = mask
        self.affine_ = affine

        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)

        for i, region in enumerate(self.A_):
            R_mask = np.logical_and(
                (region != 0.).astype('bool'), self.mask_)
            nX[:, R_mask[self.mask_]] += X[:, i, None]

        return nX

    def project(self, X, affine, mask, pooling_func=atlas_mean):
        nX = self.transform(X, affine, mask, pooling_func)
        return self.inverse_transform(nX)

    def project_array(self, array, affine, mask=None,
                      pooling_func=atlas_mean):
        if mask is None:
            if np.any(np.isnan(array)):
                mask = array != np.nan
            else:
                mask = array != 0

        X = self.project(array[mask][None, :], affine, mask, pooling_func)

        new_array = np.zeros(self.mask_.shape)
        new_array[self.mask_] = X[0, :]

        return new_array

    def iter_extract(self, X, affine, mask, weight_func=np.multiply):
        if self.A.dtype.name == 'bool':
            A = resample((self.A, self.affine), (mask, affine), 'nearest')
        else:
            A = resample((self.A, self.affine), (mask, affine))

        A = np.rollaxis(A, 3)

        self.A_ = A
        self.mask_ = mask
        self.affine_ = affine

        for i, region in enumerate(A):
            label = i + 1
            R_val = np.zeros((X.shape[0], region.size))

            R_mask = np.logical_and((region != 0.).astype('bool'), mask)
            self.R_mask_ = R_mask
            if weight_func is not None:
                R_val = weight_func(region[R_mask], X[:, R_mask[mask]])
            else:
                R_val = X[:, R_mask[mask]]

            yield label, R_val

    def extract(self, X, affine, mask, weight_func=np.multiply):
        regions = {}

        for label, R_val in self.iter_extract(X, affine, mask, weight_func):
            regions.setdefault(label, R_val)

        return regions

    def overlapping(self):
        regions = {}

        for i, region in enumerate(np.rollaxis(self.A, 3)):
            overlaps = np.sum(self.A[region.astype('bool'), :], axis=0)
            overlaps = (np.where(overlaps != 0)[0]).tolist()
            if overlaps != []:
                overlaps.remove(i)
            regions.setdefault(i, overlaps)

        return regions

    def discard(self, *labels):
        labels = list(labels)
        mask = np.ones(self.size, dtype='bool')
        mask[labels] = False
        self.A = self.A[..., mask]

        if self.label_map is not None:
            label_map = {}

            for label in labels:
                del self.label_map[label]

            for i, (k, v) in enumerate(self.label_map.iteritems()):
                if not k in labels:
                    label_map[i] = v

            self.label_map = label_map

        self.mask = (self.A != 0.).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]

    def update(self, regions, names=None):
        self.A = np.concatenate((self.A, regions), axis=3)

        if self.label_map is not None and names is not None:
            self.label_map.update(
                dict(zip(range(self.size, self.size + len(names)), names)))

        self.mask = (self.A != 0.).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]

    def add(self, region, name=None):
        if name is None:
            self.update(region[..., None])
        else:
            self.update(region[..., None], [name])

    def difference(self, label1, label2, discard_operands=False):
        mask = self.A[..., label2].astype('bool')

        if self.A.dtype.name == 'bool':
            diff = np.copy(self.A[..., label1])
            diff[mask] = False
        else:
            diff = np.copy(self.A[..., label1])
            diff[mask] -= self.A[mask, label2]
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
        mask = np.logical_or(self.A[..., label1].astype('bool'),
                             self.A[..., label2].astype('bool'))

        if self.A.dtype.name == 'bool':
            union = np.zeros(self.shape, dtype='bool')
            union[mask] = True
        else:
            union = np.zeros(self.shape, dtype='float32')
            union[mask] = self.A[mask, label1] + \
                self.A[mask, label2]

        if self.label_map is not None:
            name = '%s + %s' % (self.label_map[label1],
                                self.label_map[label2])
            self.add(union, name)
        else:
            self.add(union)

        if discard_operands:
            self.discard(label1, label2)

    def intersection(self, label1, label2, discard_operands=False):
        mask = np.logical_and(self.A[..., label1].astype('bool'),
                              self.A[..., label2].astype('bool'))

        if self.A.dtype.name == 'bool':
            inter = np.zeros(self.shape, dtype='bool')
            inter[mask] = True
        else:
            inter = np.zeros(self.shape, dtype='float32')
            inter[mask] = self.A[mask, label1] + \
                self.A[mask, label2]

        if self.label_map is not None:
            name = '%s = %s' % (self.label_map[label1],
                                self.label_map[label2])
            self.add(inter.astype(self.A.dtype), name)
        else:
            self.add(inter.astype(self.A.dtype))

        if discard_operands:
            self.discard(label1, label2)

    def subset(self, labels):
        label_map = None
        if self.label_map is not None:
            label_map = {}
            [label_map.setdefault(i, self.label_map[i]) for i in labels]

        return self.__class__(self.A[..., labels],
                              self.affine,
                              label_map)

    def to_parcellation(self, **options):
        label_map = options.get('label_map')
        P = np.zeros(self.shape, dtype='float32')

        if self.A.dtype.name != 'bool':
            threshold = options.get('threshold', .25)
            A_max = np.max(self.A, axis=3)

            for label in self.labels():
                mask = np.logical_and(self.A[..., label] > threshold,
                                      self.A[..., label] == A_max)
                P[mask] = label

            return Parcellation(P, self.affine, label_map, 0)
        else:
            P = np.zeros(self.shape, dtype='float32')

            for label in self.labels():
                R_mask = self.A[..., label]
                P[R_mask] = label

            return Parcellation(P, self.affine, label_map, 0.)

    def show(self, label=None, rcmap=None, **options):
        self.A = np.array(self.A)
        if label is not None:
            color = rcmap or 'black'
            slicer = viz.plot_map(self.A[..., label],
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
                    self.A[..., label],
                    self.affine, levels=[0],
                    colors=(color, ))
            return slicer

    def save(self, location):
        img = nb.Nifti1Image(self.A.astype('float32'), self.affine)
        nb.save(img, location)


class Parcellation(object):

    def __init__(self, P, affine, label_map=None, null_label=0):
        self.P = P.astype('float32')
        self.affine = affine.astype('float32')
        self.null_label = null_label
        self.label_map = label_map
        self.mask = (P != null_label).astype('bool')
        self.shape = P.shape
        self.size = np.unique(P[self.mask]).size
        check_float_approximation(self.P, self.mask)

    def labels(self):
        return np.unique(self.P[self.mask]).astype('int').tolist()

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def fit(self, mask, affine):
        P = resample((self.P, self.affine), (mask, affine), 'nearest')
        self.P_ = P
        self.mask_ = mask
        self.affine_ = affine
        self.r_masks_ = [
            np.logical_and(self.P_ == label, self.mask_)[self.mask_]
            for label in self.labels()]

    def transform(self, X, pooling_func=np.mean):
        nX = np.zeros((X.shape[0], self.size))
        for i, (r_mask, label) in enumerate(zip(self.r_masks_, self.labels())):
            nX[:, i] = pooling_func(X[:, r_mask], axis=1)
        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)
        for i in range(len(self.r_masks_)):
            nX[:, self.r_masks_[i]] = X[:, i, np.newaxis]
        return nX

    def iter_extract(self, X):
        for label, r_mask in zip(self.labels(), self.r_masks_):
            yield label, X[:, self.r_masks]

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
        self.P[self.P == label1] = label2
        self.size = np.unique(self.P[self.mask]).size

        if self.label_map is not None:
            self.label_map[label2] = '%s + %s' % (self.label_map[label1],
                                                  self.label_map[label2])
            del self.label_map[label1]

    def subset(self, labels):
        P = np.ones(self.P.size) * self.null_label
        ind = np.where(np.in1d(self.P, labels))[0]
        P[ind] = self.P.ravel()[ind]
        return self.__class__(
            P.reshape(self.shape), self.affine, self.null_label)

    def to_atlas(self, *parcellations, **options):
        label_map = options.get('label_map')

        if parcellations is None:
            parcellations = []

        A = np.zeros(self.shape + (self.size, ), dtype='bool')
        A = np.rollaxis(A, 3)

        for i, label in enumerate(self.labels()):
            A[i, :] = self.P == label

        A = np.rollaxis(A, 0, 4)

        for parc in parcellations:
            if parc.shape != self.shape:
                parc.P = resample((parc.P, parc.affine),
                                  (self.P, self.affine), 'nearest')

            A = np.concatenate((A, parc.to_atlas().A), axis=3)

        return Atlas(A, self.affine, label_map)

    def show(self, label=None, rcmap=None, **options):
        self.P = np.array(self.P)
        if label is None:
            return viz.plot_map(self.P, self.affine, **options)
        else:
            color = rcmap or 'black'
            slicer = viz.plot_map(self.P == label,
                                  self.affine, **options)
            slicer.contour_map(self.mask, self.affine,
                               levels=[0], colors=(color, ))
            return slicer

    def save(self, location):
        img = nb.Nifti1Image(self.P, self.affine)
        nb.save(img, location)
