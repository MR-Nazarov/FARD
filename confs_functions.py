# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

import copy
import yaml
from pathlib import Path
import numpy as np
from yaml import CLoader
from sklearn.model_selection import ParameterGrid
import inspect


def getGrid(args, gridSearch):
    for key in args:
        if not isinstance(args[key], list):
            args[key] = [args[key]]

    if gridSearch:
        grid = list(ParameterGrid({**args}))
    else:

        maxLen = np.max([len(args[x]) for x in args])
        grid = []

        for gridInd in range(maxLen):

            a_ = {}
            for arg in args:
                if len(args[arg]) > gridInd:
                    a_[arg] = args[arg][gridInd]
                else:
                    a_[arg] = args[arg][0]

            grid.append(a_)

    return grid


def add_to_default(args):
    defaultsPath = Path('confs/defaults.yaml')
    defaults = yaml.load(defaultsPath.open('r'),
                         Loader=CLoader)
    newArgs = [x for x in args if x not in defaults]
    for k in newArgs:
        defaults[k] = args[k]
    yaml.dump(defaults, defaultsPath.open('w'))
    if len(newArgs):
        print(f'adding args {newArgs} to defaults')


def compare_confs(c1, c2):
    if len([c for c in c1 if c not in c2]):
        return False
    if len([c for c in c2 if c not in c1]):
        return False
    if not all([c1[c] == c2[c] for c in c1]):
        return False
    return True


def update_args(argOld, argNew, fieldsToUpdate):
    args = copy.deepcopy(argOld)
    for field in fieldsToUpdate:
        args[field] = argNew[field]

    return argOld


def add_defaults(args,
                 keysToOverWrite=None,
                 path='confs',
                 defaultsFile='defaults'):
    defaultsPath = Path(f'{path}/{defaultsFile}.yaml')
    defaults = yaml.load(defaultsPath.open('r'),
                         Loader=CLoader)
    nonExistArgs = [x for x in defaults if x not in args]
    if keysToOverWrite is not None:
        nonExistArgs += keysToOverWrite
        nonExistArgs = [k for k in nonExistArgs if k in defaults]
    for k in nonExistArgs:
        args[k] = defaults[k]
    return args


def fix_args_if_needed(args):
    for a in args:
        if isinstance(args[a], str) \
                and ('e-' in args[a]):
            args[a] = float(args[a])
        if isinstance(args[a], str) \
                and ('e+' in args[a]):
            args[a] = int(args[a])

    return args


def inherit_from_other_conf(argsOrg):
    if 'inheritFrom' not in argsOrg:
        return argsOrg

    confPath = Path(argsOrg['inheritFrom'])
    if confPath.suffix == '':
        confPath = confPath.with_suffix('.yaml')
    argsInherited = yaml.load(confPath.open('r'),
                              Loader=CLoader)
    argsInherited = process_args(argsInherited)
    nonExistArgs = [k for k in argsInherited
                    if k not in argsOrg]
    argsInherited = {k: argsInherited[k] for
                     k in nonExistArgs}
    inheritedKeys = list(argsInherited.keys())
    argsOrg['inheritedKeys'] = inheritedKeys
    args = {**argsInherited,
            **argsOrg}
    return args


def load_conf_test(conf,
                   path=Path('confs_test'),
                   project='',
                   argsToOverride=None,
                   noDefatuls=False):
    path = path.joinpath(project)
    args = load_conf(conf,
                     path=path,
                     argsToOverride=argsToOverride,
                     noDefaults=True)

    args['test'] = True
    args['test_project'] = project
    if 'inheritedKeys' in args:
        keysToOverwrite = args['inheritedKeys']
    else:
        keysToOverwrite = None
    if not noDefatuls:
        args = add_defaults(args,
                            keysToOverWrite=keysToOverwrite,
                            defaultsFile='defaults_test')
    # add_defaults overwrites every inherited key that also exists in defaults,
    # which silently discarded argsToOverride for those keys -- so a CLI
    # `--set dontLog=true` on a conf using inheritFrom had no effect. Overrides
    # are the most specific source, so they are re-applied last.
    if argsToOverride is not None:
        args = override_args(args, argsToOverride)
    return args


def load_conf(conf,
              path=Path('confs'),
              project='',
              argsToOverride=None,
              noDefaults=False):
    _path = copy.deepcopy(path)
    path = path.joinpath(project)
    confs = [x for x in path.iterdir() if x.suffix == '.yaml']
    # Exact stem first, so a conf can be named for what it is
    # (`--conf brats_fard_t1n`). Falls back to the historical behaviour of
    # matching the last underscore-separated token, which is what makes
    # `--conf 81` find `conf_81.yaml`.
    confFile = [x for x in confs if x.stem == str(conf)]
    if not confFile:
        confFile = [x for x in confs if str(conf) == x.stem.split('_')[-1]]

    if len(confFile):
        args = yaml.load(confFile[0].open('r'), Loader=CLoader)
        args['conf'] = f'conf_{conf}'
        args = inherit_from_other_conf(args)
        args = process_args(args, noDefaults, path=_path)
        if argsToOverride is not None:
            args = override_args(args, argsToOverride)
        return args


def override_args(args, newArgs):
    for arg in newArgs:
        args[arg] = newArgs[arg]

    return args


def process_args(args, noDefaults=False, path='confs'):
    if not noDefaults:
        args = add_defaults(args, path=path)
    args = fix_args_if_needed(args)
    return args


def find_conf(args,
              path=Path('confs')):
    confs = list(path.iterdir())
    confs = [x for x in confs if x.name != 'defaults.yaml']
    confs = [x for x in confs if x.suffix == '.yaml']
    for _conf in confs:
        _args = yaml.load(_conf.open('r'), Loader=CLoader)
        if compare_confs(_args, args):
            return _conf
    return None


def save_conf(args,
              path=Path('confs')):
    add_to_default(args)
    conf = find_conf(args, path)
    if conf is None:
        confNames = list(path.iterdir())
        confNames = [x for x in confNames if x.suffix == '.yaml']
        confNames = [x.with_suffix('').name for x in confNames]
        confNames = [x for x in confNames if 'defaults' not in x]
        confNames = [int(x.split('_')[-1]) for x in confNames]
        if not len(confNames):
            confNames = [0]
        newConf = f'conf_{np.max(confNames) + 1}'
        confPath = path.joinpath(newConf).with_suffix('.yaml')
        yaml.dump(args, confPath.open('w'))
        print(f'saved new conf as {confPath}')
        conf = confPath

    return conf.with_suffix('').name


def get_func_args(kwargs, func):
    _args = inspect.getfullargspec(func).args
    _args = {k: kwargs[k] for k in _args if k in kwargs}

    return _args
