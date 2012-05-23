import numpy as np
import pylab as pl
import nibabel as nb

import nipy.labs.viz as viz
from nipy.labs.datasets import VolumeImg


def resample(source, target, interpolation='continuous'):
    source, source_affine = source
    target, target_affine = target

    input_image = VolumeImg(source[:],
                            source_affine,
                            'arbitrary',
                            interpolation=interpolation
                            )

    resampled_image = input_image.as_volume_img(target_affine, target.shape)

    return resampled_image.get_data()


class Atlas(object):

    def __init__(self, A, affine, label_map=None, null_label=0):
        self.A = A
        self.affine = affine
        self.null_label = null_label
        self.label_map = label_map
        if len(A.shape) == 3:   # single region in atlas
            self.A = A[..., None]
        self.mask = (A != null_label).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]
        
    def labels(self):
        return range(self.size + 1)[1:]

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def transform(self, X, affine, mask, 
                  pooling_func=np.mean, weight_func=np.multiply):

        not_nan = ~np.any(np.isnan(X), 0) # deal with nans
        X = X[:, not_nan]
        mask_ = np.zeros(mask.shape, dtype='bool')
        mask_[mask] = mask[mask] == not_nan

        A = resample((self.A, self.affine), (mask, affine), 'nearest')
        A = np.rollaxis(A, 3)
        
        nX = np.zeros((X.shape[0], self.size))

        for i, region in enumerate(A):
            R_mask = np.logical_and(
                (region != self.null_label).astype('bool'), mask_)
            if weight_func is not None:
                R_val = weight_func(region[R_mask], X[:, R_mask[mask_]])
            else:
                R_val = X[:, R_mask[mask_]]

            nX[:, i] =+ pooling_func(R_val, axis=1)

        self.A_ = A
        self.mask_ = mask_
        self.affine_ = affine

        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)

        for i, region in enumerate(self.A_):
            R_mask = np.logical_and(
                (region != self.null_label).astype('bool'), self.mask_)
            nX[:, R_mask[self.mask_]] += X[:, i, None]

        return nX

    def project(self, X, affine, mask, 
                pooling_func=np.mean, weight_func=np.multiply):

        nX = self.transform(X, affine, mask, pooling_func, weight_func)
        return self.inverse_transform(nX)

    def project_array(self, array, affine, mask=None, 
                      pooling_func=np.mean, weight_func=np.multiply):
        if mask is None:
            if np.any(np.isnan(array)):
                mask = array != np.nan
            else:
                mask = array != self.null_label

        X = self.project(array[mask][None, :],
                         affine, mask, pooling_func, weight_func)

        new_array = np.zeros(self.mask_.shape)
        new_array[self.mask_] = X[0, :]

        return new_array

    def iter_extract(self, X, affine, mask, weight_func=np.multiply):
        not_nan = ~np.any(np.isnan(X), 0) # deal with nans
        X = X[:, not_nan]
        mask_ = np.zeros(mask.shape, dtype='bool')
        mask_[mask] = mask[mask] == not_nan

        A = resample((self.A, self.affine), (mask, affine), 'nearest')
        A = np.rollaxis(A, 3)

        self.A_ = A
        self.mask_ = mask_
        self.affine_ = affine

        for i, region in enumerate(A):
            label = i + 1
            R_val = np.zeros((X.shape[0], region.size))

            R_mask = np.logical_and(
                (region != self.null_label).astype('bool'), mask_)
            if weight_func is not None:
                R_val = weight_func(region[R_mask], X[:, R_mask[mask_]])
            else:
                R_val = X[:, R_mask[mask_]]

            yield label, R_val

    def extract(self, X, affine, mask, weight_func=np.multiply):
        regions = {}

        for label, R_val in self.iter_extract(X, affine, mask, weight_func):
            regions.setdefault(label, R_val)

        return regions

    def overlaps(self):
        regions = {}

        A = np.rollaxis(self.A, 3)

        for i, region in enumerate(A):
            label = i + 1
            overlaps = np.sum(A[:, region.astype('bool')], axis=1)
            overlaps = (np.where(overlaps != 0)[0] + 1).tolist()
            if overlaps != []:
                overlaps.remove(label)
            regions.setdefault(label, overlaps)

        return regions

    def _discard(self, labels):
        mask = np.ones(self.size, dtype='bool')
        mask[labels] = False
        self.A = self.A[..., mask]

        if self.label_map is not None:
            label_map = {}

            for label in labels:
                del self.label_map[label - 1]
            for i, (k, v) in enumerate(self.label_map.iteritems()):
                if not k in np.array(labels) - 1:
                    label_map[i + 1] = v

            self.label_map = label_map

        self.mask = (self.A != self.null_label).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]

    def discard(self, label):
        self._discard([label])

    def _add(self, regions, names=None):
        self.A = np.concatenate((self.A, regions), axis=3)

        if self.label_map is not None and names is not None:
            self.label_map.update(
                dict(zip(range(self.size + 1, 
                               self.size + 1 + len(names)), names)))

        self.mask = (self.A != self.null_label).sum(3).astype('bool')
        self.shape = self.A.shape[:-1]
        self.size = self.A.shape[-1]

    def add(self, region, name=None):
        if name is None:
            self._add(region[..., None])
        else:
            self._add(region[..., None], [name])

    def difference(self, label1, label2, discard_operands=False):
        label1 = label1 - 1
        label2 = label2 - 1

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
            self._discard([label1, label2])

    def union(self, label1, label2, discard_operands=False):
        mask = np.logical_or(self.A[..., label1 - 1].astype('bool'), 
                             self.A[..., label2 - 1].astype('bool'))

        if self.A.dtype.name == 'bool':
            union = np.zeros(self.shape, dtype='bool')
            union[mask] = True
        else:
            union = np.zeros(self.shape, dtype='float')
            union[mask] = self.A[mask, label1 - 1] + \
                self.A[mask, label2 - 1]

        if self.label_map is not None:
            name = '%s + %s' % (self.label_map[label1 - 1], 
                                self.label_map[label2 - 1])
            self.add(union, name)
        else:
            self.add(union)
                 
        if discard_operands:
            self._discard([label1, label2])

    def intersection(self, label1, label2, discard_operands=False):
        mask = np.logical_and(self.A[..., label1 - 1].astype('bool'), 
                              self.A[..., label2 - 1].astype('bool'))
        
        if self.A.dtype.name == 'bool':
            inter = np.zeros(self.shape, dtype='bool')
            inter[mask] = True
        else:
            inter = np.zeros(self.shape, dtype='float')
            inter[mask] = self.A[mask, label1 - 1] + \
                self.A[mask, label2 - 1]

        if self.label_map is not None:
            name = '%s = %s' % (self.label_map[label1 - 1], 
                                self.label_map[label2 - 1])
            self.add(inter.astype(self.A.dtype), name)
        else:
            self.add(inter.astype(self.A.dtype))
                 
        if discard_operands:
            self._discard([label1, label2])

    def subset(self, labels):
        label_map = None
        if self.label_map is not None:
            label_map = {}
            [label_map.setdefault(i, self.label_map[i]) for i in labels]

        return self.__class__(self.A[..., np.array(labels) - 1], 
                              self.affine, 
                              label_map, 
                              self.null_label)

    def to_parcellation(self, **options):
        label_map = options.get('label_map')
        P = np.ones(self.shape, dtype='float') * self.null_label

        if self.A.dtype.name != 'bool':
            threshold = options.get('threshold', .25)
            A_max = np.max(self.A, axis=3)

            for label in self.labels():
                mask = np.logical_and(self.A[..., label - 1] > threshold, 
                                      self.A[..., label - 1] == A_max)
                P[mask] = label

            return Parcellation(P, self.affine, 
                                label_map, self.null_label)
        else:
            P = np.ones(self.shape, dtype='float') * self.null_label

            for label in self.labels():
                R_mask = self.A[..., label - 1]
                P[R_mask] = label

            return Parcellation(P, self.affine, 
                                label_map, self.null_label)

    def show(self, label=None, rcmap=None, **options):
        if label is not None:
            color = rcmap or 'black'
            slicer = viz.plot_map(self.A[..., label - 1],
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
                    self.A[..., label - 1],
                    self.affine, levels=[0],
                    colors=(color, ))
            return slicer

    def save(self, location):
        img = nb.Nifti1Image(self.A.astype('float'), self.affine)
        nb.save(img, location)


class Parcellation(object):

    def __init__(self, P, affine, label_map=None, null_label=0):
        self.P = P
        self.affine = affine
        self.null_label = null_label
        self.label_map = label_map
        self.mask = (P != null_label).astype('bool')
        self.shape = P.shape
        self.size = np.unique(P[self.mask]).size
        
    def labels(self):
        return np.unique(self.P[self.mask]).astype('int').tolist()

    def names(self):
        if self.label_map is None:
            return self.labels()
        return self.label_map.values()

    def transform(self, X, affine, mask, pooling_func=np.mean):
        not_nan = ~np.any(np.isnan(X), 0) # deal with nans
        X = X[:, not_nan]
        mask_ = np.zeros(mask.shape, dtype='bool')
        mask_[mask] = mask[mask] == not_nan

        P = resample((self.P, self.affine), (mask, affine), 'nearest')
        nX = np.zeros((X.shape[0], self.size))
        
        for i, label in enumerate(self.labels()):
            R_mask = np.logical_and(P == label, mask_)
            nX[:, i] = pooling_func(X[:, R_mask[mask_]], axis=1)

        self.P_ = P
        self.mask_ = mask_
        self.affine_ = affine

        return nX

    def inverse_transform(self, X):
        nX = np.zeros((X.shape[0], self.mask_.sum()), dtype=X.dtype)

        for i, label in enumerate(self.labels()):
            R_mask = np.logical_and(self.P_ == label, self.mask_)
            nX[:, R_mask[self.mask_]] = X[:, i, None]

        return nX

    def iter_extract(self, X, affine, mask):
        not_nan = ~np.any(np.isnan(X), 0) # deal with nans
        X = X[:, not_nan]
        mask_ = np.zeros(mask.shape, dtype='bool')
        mask_[mask] = mask[mask] == not_nan

        P = resample((self.P, self.affine), (mask, affine), 'nearest')
        
        for i, label in enumerate(self.labels()):
            R_mask = np.logical_and(P == label, mask_)
            yield label, X[:, R_mask[mask_]]

    def extract(self, X, affine, mask):
        regions = {}

        for label, R_val in self.iter_extract(X, affine, mask):
            regions.setdefault(label, R_val)

        return regions

    def project(self, X, affine, mask, pooling_func=np.mean):
        nX = self.transform(X, affine, mask, pooling_func)
        return self.inverse_transform(nX)

    def project_array(self, array, affine, mask=None, pooling_func=np.mean):
        if mask is None:
            if np.any(np.isnan(array)):
                mask = array != np.nan
            else:
                mask = array != self.null_label

        X = self.transform(array[mask][None, :], affine, mask, pooling_func)
        nX = self.inverse_transform(X)

        new_array = np.zeros(self.mask_.shape)
        new_array[self.mask_] = nX[0, :]

        return new_array

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

        return Atlas(A, self.affine, label_map, null_label=0)

    def show(self, label=None, rcmap=None, **options):
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
