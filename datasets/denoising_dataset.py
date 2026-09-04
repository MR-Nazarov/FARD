import joblib
from image.my_image import myImage
from pathlib2 import Path
from .dataset import *
import numpy as np
import inspect


class testDataset(datasetClass):

    def __init__(self, args):

        super(testDataset, self).__init__(args)

    def pad2log2(self):

        self.processed = np.zeros((self.input.imShape[0],
                                   self.input.imShape[-2],
                                   self.input.imShape[-1]))
        nearestPadRows = np.ceil(np.log2(self.input.I.shape[-1]))
        nearestPadCols = np.ceil(np.log2(self.input.I.shape[-2]))
        padCols = (
            (2 ** nearestPadCols - self.input.I.shape[-1]) / 2,
            (2 ** nearestPadCols - self.input.I.shape[-1]) / 2)
        padRows = (
            (2 ** nearestPadRows - self.input.I.shape[-2]) / 2,
            (2 ** nearestPadRows - self.input.I.shape[-2]) / 2)
        if np.round(padCols[0]) != padCols[0]:
            padCols = (int(padCols[0]), int(padCols[1]) + 1)
        else:
            padCols = tuple([int(x) for x in padCols])
        if np.round(padRows[0]) != padRows[0]:
            padRows = (int(padRows[0]), int(padRows[1]) + 1)
        else:
            padRows = tuple([int(x) for x in padRows])
        if self.input.I.ndim == 3:
            self.input.set_image(np.pad(self.input.I,
                                        ((0, 0), padRows, padCols),
                                        mode='constant'))
        else:
            self.input.set_image(np.pad(self.input.I,
                                        ((0, 0), (0, 0), padRows, padCols),
                                        mode='constant'))
        self.padCols = padCols
        self.padRows = padRows

    def read_target(self):

        if len(self.args.inputDirs) <= self.inputDirIdx:
            return None

        target = None
        if self.args.baseTargetPath is not None:

            target = myImage()
            targetPath = self.args.inputDirs[self.inputDirIdx]
            niiPath = Path(self.args.baseTargetPath). \
                joinpath(targetPath.name).with_suffix('.nii.gz')
            if niiPath.exists():
                targetPath = niiPath

            target.read_im(targetPath, slices=self.args.slices)
            if target.I.ndim == 2:
                target.set_image(target.I[None])
            target.I = target.I.astype('float32')

        return target

    def read_input(self):

        input = myImage()
        if len(self.args.inputDirs) > self.inputDirIdx:
            path = self.args.inputDirs[self.inputDirIdx]
        else:
            return None

        inputArgs = inspect.getfullargspec(input.read_im).args
        argsReadIm = {key: self.args[key] for key in inputArgs
                      if key in self.args}
        input.read_im(path, **argsReadIm)

        if isinstance(input.I, list):
            if input.I[0].ndim == 2:
                input.I = [input.set_image(x[None, :, :])
                           for x in self.input.I]
            self.dtype = input.I[0].dtype
            input.I = [x.astype('float32')
                       for x in self.input.I]
        else:
            if input.I.ndim == 2:
                input.set_image(input.I[None, :, :])
            self.dtype = input.I.dtype
            input.I = input.I.astype(np.float32)

        self.image.meta = input.meta
        self.image.sourcePath = input.sourcePath
        return input

    def post_set_data(self):

        super(testDataset, self).post_set_data()
        self.processed = np.zeros_like(self.input)

    def get_return_dict_read_data(self, input, targets):
        dataDict = {'targets': targets,
                    'input': input,
                    'patchSize': None,
                    'patchSizeTarget': None}
        if targets is not None:
            dataDict['targets'] = targets.I
            dataDict['patchSizeTarget'] = targets.shape[1:]
        if input is not None:
            dataDict['input'] = input.I
            dataDict['patchSize'] = input.shape[1:]

        return dataDict

    def read_data(self):

        super(testDataset, self).read_data()
        targets = self.read_target()
        input = self.read_input()

        # check if threeD and add 1/2 dimensions based on the exsiting number of dimensions
        if input is not None \
                and input.ndim > 3:
            input.I = input.I.reshape((*tuple([1] *
                                              (5 - input.I.ndim)),
                                       *input.I.shape))
            if self.args.targetPath is not None:
                targets.I = targets.I.reshape((*tuple([1] *
                                                      (5 - targets.I.ndim)),
                                               *targets.I.shape))
        dataDict = self.get_return_dict_read_data(input=input,
                                                  targets=targets)

        return dataDict

    def init(self):

        super(testDataset, self).init()

        if self.args.padToLog2:
            self.pad2log2()
        self.image = myImage()

    def normalize_func(self):
        self.input = normalize_func(self.input,
                                    params=self.args.normalizeParams)

    def post_process_pad_size(self, padSize):

        if padSize % 2 != 0:
            padSize = int(padSize / 2)
            padSize = [padSize, padSize + 1]
        else:
            padSize = int(padSize / 2)
            padSize = [padSize, padSize]

        return padSize

    def add_result(self, idx, result):

        if not hasattr(self, 'processed'):
            self.processed = result[0]
            if self.processed.ndim == 4 and not self.args.multi_modal:
                self.processed = self.processed[0]
            # untouchableDims = self.processed.ndim - len(self.args.imDimShape)
            # # self.args.ImDimShape = untouchableDims + np.array(self.args.imDimShape, dtype=np.int64)
            # untouchables = [-1] * untouchableDims
            self.processed = np.transpose(self.processed, self.args.imDimShape)
            # self.processed = np.transpose(self.processed, transposed_shape)
            return 'denoised all slices'

        self.processed[idx * self.args.batchSize:idx * self.args.batchSize + self.args.batchSize] = \
            result[:, 0]
        stringToReturn = f'denoised slice 0/' \
                         f'{self.processed.shape[0]}'

        return stringToReturn

    def pre_write_functions(self):
        funcs = super(testDataset, self).pre_write_functions()
        for func in funcs:
            inputArgs = inspect.getfullargspec(func).args
            args = {key: self.args[key] for
                    key in inputArgs if key in self.args}


    def write_result(self, resultDir=None,
                     description='',
                     sourcePath=None, mod=None):

        # self.processed = np.zeros((self.input.imShape[0],
        #                            self.input.imShape[-2],
        #                            self.input.imShape[-1]))
        if resultDir is None:
            inputDir = self.args.inputDirs[self.inputDirIdx].name
            resultDir = self.args.outputPathTest.joinpath(inputDir)
        description = f'processed by {self.args.conf}'
        super(testDataset, self).write_result(resultDir, description)
        if sourcePath is None:
            sourcePath = self.args.inputDirs[self.inputDirIdx]
        self.input.dtype = self.args.dtype
        if mod is None:
            self.image.write_image_(im=self.processed,
                                    path=resultDir,
                                    fixZPositions=self.args.fixZPositions,
                                    description=description,
                                    source_path=sourcePath)
        else:
            self.image.write_image_(im=self.processed[mod],
                                    path=resultDir,
                                    fixZPositions=self.args.fixZPositions,
                                    description=description,
                                    source_path=sourcePath)
        if hasattr(self, 'inputDirIdx'):
            self.inputDirIdx += 1


class trainDataset(datasetClass):

    def __init__(self, args):

        super(trainDataset, self).__init__(args)

    def read_data(self):

        super(trainDataset, self).read_data()
        try:
            load = joblib.load(self.args.filename)
        except Exception as e:
            import sys
            sys.modules['sklearn.externals.joblib'] = joblib
            load = joblib.load(self.args.filename)

        if 'patchSize' not in load:
            sqrt = np.sqrt(load['train_db'].shape[-1])
            if sqrt % 1 == 0:
                load['patchSize'] = int(sqrt)
                load['patchSize'] = (load['patchSize'],
                                     load['patchSize'])
            else:
                load['patchSize'] = load['train_db'].shape[1:]
        return {'targets': load['train_labels'],
                'input': load['train_db'],
                'patchSize': load['patchSize']}

    def init(self):

        super(trainDataset, self).init()
        if self.args.filenameVal is not None:
            self.testSize = 0
            if not self.args.trainPhase:
                self.filename = self.args.filenameVal
                self.trainPhase = True

    def normalize_func(self):
        self.input = normalize_func(self.input,
                                    self.args.normalizeParams)

    def post_set_data(self):
        super(trainDataset, self).post_set_data()
