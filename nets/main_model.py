# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

import inspect
import nets as Nets
from nets.functions import *
from confs_functions import get_func_args
from monai import transforms
#from M_build.datasets import env_transforms

class MainModel(torch.nn.Module):

    def __init__(self, args):
        super(MainModel, self).__init__()
        self.args = args
        self.config_model()
        # self.init_transforms(name='preProcessModel')
        # self.init_transforms(name='postProcessModel')

    def config_model(self):
        net_module = getattr(Nets, self.args.cnnModule)
        modelObj = getattr(net_module, self.args.modelName)
        inputArgs = inspect.getfullargspec(modelObj.__init__).args
        modelArgs = {k: self.args[k] for k in self.args if k in inputArgs}
        self.model = modelObj(**modelArgs)

    def forward(self, x):
        x = self.pre_process_model(x)
        out = self.model(x)
        out = self.post_process_model(out)

        return out

    def pre_process_model(self, x):
        if hasattr(self, 'preProcessModelTransforms'):
            return self.preProcessModelTransforms(x)
        if 'preProcessModel' in self.args \
                and self.args.preProcessModel is not None:
            x = eval(self.args.preProcessModel)(x)
        return x

    def post_process_model(self, x):
        if hasattr(self, 'postProcessModelTransforms'):
            return self.postProcessModelTransforms(x)
        if 'postProcessModel' in self.args \
                and self.args.postProcessModel is not None:
            func = eval(self.args.postProcessModel)
            funcArgs = get_func_args(self.args, func)
            x = func(x, **funcArgs)
        return x

    def init_transforms(self, name):

        if isinstance(self.args.get(name), dict):
            Transforms = []
            for transform in self.args.get(name):
                if hasattr(transforms, transform):
                    transformFunc = getattr(transforms, transform)
                # elif hasattr(env_transforms, transform):
                #     transformFunc = getattr(env_transforms, transform)
                _transform = transformFunc(**self.args.get(name)[transform])
                Transforms.append(_transform)
            setattr(self, f'{name}Transforms',
                    transforms.Compose(Transforms))
