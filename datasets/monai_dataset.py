# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

import json
from pathlib import Path
import yaml

from .dataset import *
from monai.data import Dataset as MonaiDataset
from monai.data import CacheDataset as CachedMonaiDataset
from monai.data import load_decathlon_datalist


def _resolve_base_dir(json_path: str, split: str = None) -> Path | None:
    """Return the local base_dir for a JSON dataset file.

    Reads the 'base_dir' field stored in the JSON, then looks it up in
    local.yaml's base_dir_map to get the machine-local path.
    Falls back to the stored base_dir if local.yaml is missing or has no match.
    Returns None if the JSON has no base_dir (old-format absolute paths).
    """
    try:
        data = json.loads(Path(json_path).read_text())
    except Exception:
        return None

    stored_base = data.get("base_dir")
    if not stored_base:
        return None

    local_yaml = Path("local.yaml")
    if local_yaml.exists():
        mapping = yaml.safe_load(local_yaml.read_text()).get("base_dir_map", {})
        if stored_base in mapping:
            return Path(mapping[stored_base])

    return Path(stored_base)


class _testDataset(datasetClass):
    def __init__(self, args):
        super().__init__(args)
        self.datalist = load_decathlon_datalist(data_list_file_path=args.jsonData,
                                                is_segmentation=self.args.segmentation,
                                                data_list_key=self.args.keyTest,
                                                base_dir=_resolve_base_dir(args.jsonData, split='test'))


class _trainDataset(datasetClass):
    def __init__(self, args):
        super().__init__(args)
        key = 'training' if \
            self.args.trainPhase \
            else 'validation'
        self.datalist = load_decathlon_datalist(data_list_file_path=args.jsonData,
                                                is_segmentation=self.args.segmentation,
                                                data_list_key=key,
                                                base_dir=_resolve_base_dir(args.jsonData, split=key))


class testDataset(MonaiDataset, _testDataset):

    def __init__(self, args):
        self.dataIdx = []
        _testDataset.__init__(self, args)
        transforms = getattr(self, 'valTransforms',
                             getattr(self, 'transforms'))
        super(testDataset, self).__init__(data=self.datalist,
                                          transform=transforms)

    def __getitem__(self, idx):
        self.dataIdx.append(idx)
        return super(testDataset, self).__getitem__(idx)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """


class trainDataset(MonaiDataset, _trainDataset):

    def __init__(self, args):
        # create transforms and read json file using decathlon - monai.data.load_decathlon_datalist()
        _trainDataset.__init__(self, args)
        transformName = 'transforms'
        if not self.args.trainPhase:
            transformName = 'valTransforms'
        transforms = getattr(self, transformName, getattr(self, 'transforms'))
        super(trainDataset, self).__init__(data=self.datalist,
                                           transform=transforms)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """


class testDatasetCached(CachedMonaiDataset, _testDataset):

    def __init__(self, args):
        _testDataset.__init__(self, args)
        transforms = getattr(self, 'valTransforms',
                             getattr(self, 'transforms'))
        super(testDatasetCached, self).__init__(data=self.datalist,
                                                transform=transforms,
                                                cache_num=self.args.cache_num,
                                                cache_rate=self.args.cache_rate,
                                                num_workers=args.workers)


class trainDatasetCached(CachedMonaiDataset, _trainDataset):

    def __init__(self, args):
        _trainDataset.__init__(self, args)
        transformName = 'transforms'
        if not self.args.trainPhase:
            transformName = 'valTransforms'
        transforms = getattr(self, transformName, getattr(self, 'transforms'))
        super(trainDatasetCached, self).__init__(data=self.datalist,
                                                 transform=transforms,
                                                 cache_num=self.args.cache_num,
                                                 cache_rate=self.args.cache_rate,
                                                 num_workers=args.workers)

    def read_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_set_data(self):
        """
        Only to override the read_data of the base classes
        """

    def post_read_data(self, data):
        """
        Only to override the read_data of the base classes
        """
