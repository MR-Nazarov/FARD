import nets as Nets
import yaml
from yaml import CLoader
from pathlib2 import Path
import inspect
import torch.nn as nn
import torchvision.models
import losses.perceptual_nets as lossNets

class PerceptualLoss(nn.Module):

    def __init__(self, args, index):
        super(PerceptualLoss, self).__init__()
        self.args = args
        self.index = index
        self.torchNet = False
        self.config_model()


    def __call__(self, x, **kwargs):

        inputArgs = inspect.getfullargspec(self.inference.forward).args
        kwargs = {k: kwargs[k] for k in kwargs if k in inputArgs}
        out = self.inference(x, **kwargs)

        return out

    def _get_perceptualNet_modules(self):
        netSplt = self.args.perceptualNet.split(',')[self.index].split('.')
        module = '.'.join(netSplt[:-1])
        cnnName = netSplt[-1]
        return module, cnnName

    def config_perceptual_net(self):
        module, cnnName = self._get_perceptualNet_modules()
        net_module = getattr(Nets, module, None)
        if net_module is None:
            net_module = getattr(torchvision.models,module)
            self.torchNet = True

        if self.torchNet:
            modelObj = getattr(net_module, cnnName)(pretrained=True)
            self.inference = getattr(lossNets, f"{module}LossNetwork")(modelObj)
            self.inference.eval()
            return
        modelObj = getattr(net_module, cnnName)
        configPath = Path(self.args.perceptualNetConfig)
        config = yaml.load(configPath.open('r'),
                           Loader=CLoader)
        inputArgs = inspect.getfullargspec(modelObj.__init__).args
        if hasattr(net_module, 'convert_configs'):
            modelArgs = self.convert_configs(net_module=net_module,
                                             inputArgs=inputArgs,
                                             config=config)
        else:
            modelArgs = config
        if self.args.overridePerceptualArgs:
            argsToAdd = [k for k in self.args if k in inputArgs]
        else:
            argsToAdd = [k for k in self.args if k in inputArgs
                         and k not in modelArgs]
        modelArgs = {**modelArgs,
                     **{k: self.args[k] for k in argsToAdd}}
        modelArgs['perceptualFeatures'] = True
        modelArgs['addInputToPerceptual'] = self.args.addInputToPerceptual
        self.inference = modelObj(**modelArgs)
        # self.perceptualNet.to(self.args.device)
        self.inference.eval()

    def config_model(self):
        self.config_perceptual_net()
        if not self.torchNet:
            self.load_pretrained()

    def convert_configs(self, inputArgs, net_module, config):
        config, confKey = getattr(net_module, 'convert_configs')(config)
        if confKey is not None:
            modelconfig = self.getattrd(config, confKey)
            modelArgs = {k: modelconfig[k]
                         for k in modelconfig
                         if k in inputArgs}
        else:
            modelArgs = {k: config[k] for k in config if k in inputArgs}

        return modelArgs

    def getattrd(self, obj, name):
        if '.' not in name:
            return getattr(obj, name)
        nameSplt = name.split('.')
        return self.getattrd(getattr(obj, nameSplt[0]),
                             '.'.join(nameSplt[1:]))

    def load_pretrained(self):
        module, _ = self._get_perceptualNet_modules()
        module = getattr(Nets, module)
        configPath = Path(self.args.perceptualNetConfig)
        config = yaml.load(configPath.open('r'),
                           Loader=CLoader)
        if hasattr(module, 'load_pretrained'):
            module.load_pretrained(config=config,
                                   model=self.inference)
