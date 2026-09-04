"""Dataset implementations, selected by the conf's ``dataset:`` field.

``dataloaderClass`` resolves it with ``getattr(datasets, args.dataset)``, so a
dataset must be imported here to be reachable. The predecessor also imported
``denoising_dataset``, ``segmentation_dataset`` and ``segmentation_dataset_monai``;
no conf in this repo references them.
"""

import datasets.denoising_dataset_monai
