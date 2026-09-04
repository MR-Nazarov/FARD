import munch
import datasets as datasets
from torch.utils.data import DataLoader
from monai.data import DataLoader as MonaiDataloader


class dataloaderClass:

    def __init__(self, args):
        self.args = args
        self.init()

    def init(self):
        className = 'testDataset' if \
            self.args.test else 'trainDataset'
        if self.args.useCachedDS:
            className += 'Cached'
        dataset_module = getattr(datasets, self.args.dataset)
        self.datasetClass = getattr(dataset_module, className)
        self.init_datasets()
        self.assign_dataloader_args()
        self.init_data_in_datasets()
        self.assign_data()

    def reload_new_data(self):
        self.read_data(self.evalDataset)
        self.assign_data()

    def init_data_in_datasets(self):

        if not self.args.test:
            self.read_data(self.trainDataset)
        else:
            self.read_data(self.evalDataset)

    @property
    def shape(self):
        if self.args.test:
            return self.evalDataset.shape
        else:
            return self.trainDataset.shape

    def init_datasets(self):

        self.args['trainPhase'] = not self.args.test
        if self.args.trainPhase:
            self.trainDataset = self.datasetClass(self.args)
        self.args['trainPhase'] = False
        self.evalDataset = self.datasetClass(self.args)

    def assign_dataloader_args(self):
        workers = self.args.get('workers', 0)
        self.dataLoaderArgs = dict(
            batch_size=self.args.batchSize,
            shuffle=self.args.shuffle,
            num_workers=workers,
            pin_memory=True,
            collate_fn=self.args.collate_batch,
            drop_last=self.args.dropLast)
        # if 'monai' in self.args.dataset:
        #     del self.dataLoaderArgs['collate_fn']
        if workers > 0:
            if hasattr(DataLoader, 'persistent_workers'):
                self.dataLoaderArgs['persistent_workers'] = self.args.get('persistent_workers', True)

            if hasattr(DataLoader, 'prefetch_factor'):
                self.dataLoaderArgs['prefetch_factor'] = self.args.get('prefetch_factor', 2)

    def read_data(self, ds):
        self.data = ds.read_data()
        self.data = munch.munchify(self.data)
        self.data = ds.post_read_data(self.data)

    def assign_data(self):
        DL_class = MonaiDataloader \
            if 'monai' in self.args.dataset \
            else DataLoader
        for dsType in ['trainDataset', 'evalDataset']:
            dlType = dsType.replace('Dataset', '')
            if hasattr(self, dsType):
                ds = getattr(self, dsType)
                valid = ds.set_data_base(self.data)
                if valid or 'monai' in str(ds):
                    setattr(self,
                            dlType,
                            DL_class(dataset=ds,
                                     **self.dataLoaderArgs))

                else:
                    setattr(self, dlType, None)

        # free memory of self.data after assiging the data to the datasets
        del self.data
