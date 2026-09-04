import numpy as np
import torchio as tio
import inspect
from .segmentation_functions import *
from skimage.transform import resize


def reshape(x, orgShape):

    orgDtype = x.dtype
    x = resize(x.astype('float32'), output_shape=orgShape, order=1)
    return x.astype(orgDtype)


def normalize_func(X, params):
    func = return_normalization_func(normalizeParams=params)
    initialDtype = X.dtype

    dimsToAdd = 4 - X.ndim
    if params['per_example']:
        dimsToRemove = 1 + dimsToAdd
        if dimsToAdd > 0:
            X = X[tuple([None] * dimsToAdd)]
            X = np.transpose(X, (dimsToAdd,
                                 *tuple(range(dimsToAdd)),
                                 *tuple(range(-2, 0, 1))))
        X = np.array([func(x[None]) for x in X])
        for _ in range(dimsToRemove):
            X = X[:, 0]
    else:
        if dimsToAdd > 0:
            axis = list(range(dimsToAdd))
            X = np.expand_dims(X, axis=axis)
        X = func(X)
        if dimsToAdd > 0:
            for i in range(dimsToAdd):
                X = np.squeeze(X, axis=0)
    X = X.astype(initialDtype)
    if 'multiplier' in params:
        X = X * float(params['multiplier'])
    if 'clamp' in params:
        X = np.clip(X, *params['clamp'])

    return X


def return_normalization_func(normalizeParams):
    func = None

    if normalizeParams['method'] == 'normal':
        func = tio.transforms.ZNormalization
        inputArgs = inspect.getfullargspec(func).args
        args = {key: normalizeParams[key] for key in inputArgs
                if key in normalizeParams}
        func = tio.transforms.ZNormalization(**args)

    elif normalizeParams['method'] == 'minmax':
        func = tio.transforms.RescaleIntensity
        if 'out_min_max' not in normalizeParams:
            normalizeParams['out_min_max'] = (0, 1)
        inputArgs = inspect.getfullargspec(func).args
        args = {key: normalizeParams[key] for key in inputArgs if key in normalizeParams}
        func = tio.transforms.RescaleIntensity(**args)

    return func
