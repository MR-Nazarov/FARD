import importlib
import munch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from skimage.transform import resize
from .functions import *
from pathlib2 import Path
from tqdm import tqdm
import copy
from monai import transforms
from monai.transforms import apply_transform
import datasets.env_transforms as env_transforms
import ast
# transforms.ConcatItemsd1 = transforms.ConcatItemsd

class datasetClass(Dataset):

    def __init__(self, args):

        self.deepcopy_args(args)
        # just for initialization
        self.input = np.array([])
        self.targets = np.array([])

        # read data
        self.init()

        # reshape if needed

    def deepcopy_args(self, args):
        problemKeys = ['collate_batch']
        _args = {k: args[k] for k in args
                 if k not in problemKeys}
        self.args = copy.deepcopy(_args)
        for k in problemKeys:
            self.args[k] = args[k]
        if not isinstance(self.args, munch.Munch):
            self.args = munch.munchify(self.args)

    def check_data_validity(self, data):
        return data is not None \
            and data.input is not None

    def add_tenosor_to_dataset(self, tensor, name):
        pass

    def visualize(self):
        pass

    def add_tensor_to_dataset(self, tensor,
                              name,
                              tag=None,
                              lastIter=False,
                              isBatch=True):
        pass

    def set_data_base(self, data):

        valid = self.check_data_validity(data)
        if valid:
            trainInds, testInds = self.calc_inds(data)
            indsToTake = trainInds if self.args.trainPhase else testInds
            if isinstance(data.input, list):
                self.input = [data.input[x] for x in indsToTake]
            else:
                self.input = data.input[indsToTake]

            if data.targets is None:
                self.targets = data.targets
            else:
                if isinstance(data.targets, list):
                    self.targets = [data.targets[x] for x in indsToTake]
                else:
                    self.targets = data.targets[indsToTake]

            self.post_set_data()

        return valid

    def write_results_batch(self):
        """
        In case each batch is a different case - we want to write the results for each batch
        :return:
        """
        self.pre_write_results()
        if self.args.preWriteFunctions is not None:
            self.pre_write_functions()

    def write_result(self, resultDir, description=''):

        """
        Only for testDataset
        :param resultDir:
        :param description:
        :return:
        """
        self.pre_write_results()
        if self.args.preWriteFunctions is not None:
            self.pre_write_functions()

    def pre_write_functions(self):

        """
        Here we perform some operations on the data before writing it to the disk
        This is different than pre_write_results - this uses other functions found in functions python file
        :return:
        """

        glbs = globals()
        if not isinstance(self.args.preWriteFunctions, list):
            self.args.preWriteFunctions = [self.args.preWriteFunctions]

        funcs = []
        for func in self.args.preWriteFunctions:
            if func in glbs:
                FuncPtr = glbs[func]
                funcs.append(FuncPtr)

        return funcs

    def pre_write_results(self):

        """
        function for processing before writing to disk
        :return:
        """

    def post_set_data(self):

        # normalize if needed
        if self.args.trainPhase:
            print(f'Training - number of examples is {len(self.input)}')
        else:
            print(f'Validation - number of examples is {len(self.input)}')
        if self.args.normalize:
            self.normalize_func()

    def init(self):

        self.init_transforms()
        if self.args.filename is not None:
            self.args.filename = Path(self.args.basePathDataset,
                                      self.args.filename)
            if self.args.filenameVal is not None:
                self.args.filenameVal = Path(self.args.basePathDataset,
                                             self.args.filenameVal)
            self.name_ = self.args.filename.name
        self.batchIndices = []
        self.epochIndices = []

        if self.args.test \
                and self.args.jsonData is None:
            if self.args.inputDirs is None:
                self.args.inputDirs = list(Path(self.args.baseInputPath).iterdir())
            elif not isinstance(self.args.inputDirs, list):
                self.args.inputDirs = str(self.args.inputDirs).split(',')
                self.args.inputDirs = [Path(self.args.baseInputPath, p)
                                       for p in self.args.inputDirs]
            self.inputDirIdx = 0

    def check_transform_func_args(self, args):

        for arg in args:
            if isinstance(args[arg], str) \
                    and 'lambda' in args[arg]:
                args[arg] = eval(args[arg])

    def init_transforms(self):
        self._init_transforms('transforms')
        self._init_transforms('valTransforms')
        self._init_transforms('post_transforms')

    def _init_transforms(self, _T):

        T = self.args.get(_T, None)
        if T is not None:
            T = munch.Munch({**self.args.defaultTransforms, **T})
            Transforms = []
            for transform in T:
                if hasattr(transforms, transform):
                    transformFunc = getattr(transforms, transform)
                elif hasattr(env_transforms, transform):
                    transformFunc = getattr(env_transforms, transform)
                else:
                    raise Exception(f"Error: Transform function '{transform}' does not exist.")
                _transform = transformFunc(**T[transform])
                Transforms.append(_transform)
            setattr(self, _T, transforms.Compose(Transforms))

    def calc_inds(self, data):

        if self.args.test:
            testInds = np.arange(len(data.input))
            trainInds = np.empty(0)
        else:
            testInds = np.arange(self.args.testSize)
            trainInds = np.arange(len(data.input))[self.args.testSize:]

        return trainInds, testInds

    def update_indices(self):
        self.epochIndices += self.batchIndices
        self.batchIndices = []

    def init_epoch_indices(self):
        self.epochIndices = []

    def normalize_func(self):

        pass

    def read_data(self):
        """

        :return:
        return {'targets': targets,
                'input': input}
        """
        print('reading data...')

    def permute_data(self, data):
        if self.args.permuteData:
            dataIdxs = np.arange(len(data.input))
            permutation = np.random.permutation(dataIdxs)
            data.input = data.input[permutation]
            if data.targets is not None:
                data.targets = data.targets[permutation]
        return data

    def reshape_to_org_size(self, data):

        if self.check_if_need_to_reshape(data,
                                         keyX='input',
                                         keyPS='patchSize'):
            data.input = np.reshape(data.input,
                                    (data.input.shape[0],
                                     *data.patchSize))

        patchSizeTargetKey = 'patchSize' if 'patchSizeTarget' \
                                            not in data else 'patchSizeTarget'
        if self.check_if_need_to_reshape(data,
                                         keyX='targets',
                                         keyPS=patchSizeTargetKey):
            data.targets = np.reshape(data.targets,
                                      (data.targets.shape[0],
                                       *data[patchSizeTargetKey]))

        if not self.args.trainPhase \
                and self.args.threeD:
            for k in ['input', 'targets']:
                if not data[k] is None:
                    data[k] = data[k][None]

        return data

    def check_if_need_to_reshape(self, x, keyX, keyPS):
        return isinstance(x, np.ndarray) \
            and x[keyX] is not None \
            and x[keyX].shape[1:] != x[keyPS]

    def _change_dtype(self, x, dtype):

        if x is not None:
            if isinstance(x, list):
                x = [xx.astype(dtype) for xx in x]
            else:
                x = x.astype(dtype)
        return x

    def change_dtype(self, data):

        dtype = 'float16' if self.args.singlePrecision else 'float32'
        data.targets = self._change_dtype(data.targets, dtype)
        data.input = self._change_dtype(data.input, dtype)

        return data

    def post_read_data(self, data):

        if data is not None \
                and data.input is not None:
            data = self.permute_data(data)
            data = self.reshape_to_org_size(data)
            data = self.change_dtype(data)
            data = self.resize_input_and_targets(data)

        return data

    def apply_transforms(self, itemDict):

        if not self.args.trainPhase \
                and hasattr(self, 'valTransforms'):
            itemDict = apply_transform(transform=self.valTransforms,
                                       data=itemDict)
        elif hasattr(self, 'transforms'):
            itemDict = apply_transform(transform=self.transforms,
                                       data=itemDict)
        return itemDict

    def prepare_item(self, idx):

        itemDict = {'input': self.input[idx]}
        if self.targets is not None:
            itemDict['target'] = self.targets[idx]
        itemDict = self.apply_transforms(itemDict)
        return itemDict

    def resize(self, data, order):
        if data is not None:
            if len(self.args.outShape) == 2:
                data = self.resize_2d(data, order)
            else:
                data = self.resize_3d(data, order)
        return data

    def resize_2d(self, data, order):

        dataShape = data.shape
        newData = np.zeros((*dataShape[:2],
                            *self.args.outShape),
                           dtype=data.dtype)
        # looks ugly, but saves menewnewData
        loader = tqdm(range(dataShape[0]))
        for exampleIdx in loader:
            loader.set_description(f'resizing exampleIdx = {exampleIdx + 1}')
            for sliceIdx in range(dataShape[1]):
                newData[exampleIdx, sliceIdx] = resize(data[exampleIdx, sliceIdx],
                                                       output_shape=self.args.outShape,
                                                       order=order)

        return data

    def resize_3d(self, data, order):

        dataShape = data.shape
        newData = np.zeros((*dataShape[:1],
                            *self.args.outShape),
                           dtype=data.dtype)
        # looks ugly, but saves memory
        loader = tqdm(range(dataShape[0]))
        for exampleIdx in loader:
            loader.set_description(f'resizing exampleIdx = {exampleIdx + 1}')
            newData[exampleIdx] = resize(data[exampleIdx],
                                         output_shape=self.args.outShape,
                                         order=order)

        return newData

    def resize_by_slicing(self, data):
        pixelsToSlice = np.array(data.shape[-2:]) - np.array(self.args.outShape)
        data = data[..., :-pixelsToSlice[0], :-pixelsToSlice[1]]
        return data

    def resize_input_and_targets(self, data):

        if self.args.outShape is None:
            return data

        if not self.args.trainPhase:
            self.args.orgShape = data.input.shape
            if data.input.ndim == 4:
                self.args.orgShape = self.args.orgShape[1:]
            if self.args.preWriteFunctions is not None:
                if not isinstance(self.args.preWriteFunctions, list):
                    self.args.preWriteFunctions = [self.args.preWriteFunctions]
                self.args.preWriteFunctions += ['reshape']
            else:
                self.args.preWriteFunctions = ['reshape']

        if data.input is not None:
            print('resizing input...')
        if self.args.resizeBySlicing:
            data.input = self.resize_by_slicing(data.input)
        else:
            data.input = self.resize(data.input, order=1)
        if data.targets is not None:
            print('resizing targets...')
        if self.args.resizeBySlicing:
            data.targets = self.resize_by_slicing(data.targets)
        else:
            data.targets = self.resize(data.targets, order=1)
        if data.targets is not None:
            data.targets[data.targets > 0] = 1

        return data

    @property
    def name(self):
        return self.name_

    @property
    def shape(self):
        if isinstance(self.input, list):
            return (len(self.input),
                    *self.input[0].shape)
        return self.input.shape

    def __len__(self):

        return len(self.input)

    def __getitem__(self, idx):

        self.batchIndices.append(idx)
        return self.prepare_item(idx)

    def input_size(self):

        return self.input[0].shape[1]


class OverlapSampler(Sampler):
    r""" add overlap between batches of examples

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, dataset, numOverlap, batchSize):
        super().__init__()
        self.lenDatset = len(dataset)
        self.numOverlaps = numOverlap
        self.indices = []

        i = numOverlap - batchSize
        while i < self.lenDatset - batchSize + numOverlap:
            indsToAdd = np.arange(i, i + batchSize)
            indsToAdd[indsToAdd < 0] = 0
            indsToAdd[indsToAdd >= self.lenDatset] = self.lenDatset - 1
            i += batchSize - numOverlap
            self.indices += indsToAdd.tolist()

    def __iter__(self):
        return (self.indices[i] for i in np.arange(0, len(self.indices) - 1))

    def __len__(self):
        return len(self.indices)
