from skimage import morphology, measure
import numpy as np


def largestCC(out, structureSize, largestCCStructure='cube'):
    out = out.astype('int')
    selem = getattr(morphology, largestCCStructure)(structureSize)
    out2 = binary_opening(out, selem)
    ccs = measure.label(out2)
    bb = np.bincount(ccs.flatten())
    largestCC = np.argmax(bb[1:]) + 1
    out[ccs != largestCC] = 0
    out[out > 0] = 1

    return out


def binary_opening(out, selem):
    if selem.ndim < out.ndim:
        for idx, Slice in enumerate(out):
            out[idx] = morphology.binary_opening(Slice,
                                                 selem)
    else:
        out = morphology.binary_opening(out,
                                        selem)

    return out
