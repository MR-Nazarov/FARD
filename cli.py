# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Single entry point for training and testing.

The predecessor had ``train.py`` and ``test.py``, each with a hardcoded list of
conf numbers at the bottom that you edited before every run -- ``test.py`` carried
three commented-out experiment blocks competing for the same slot. Which
experiment ran was therefore an uncommitted source edit.

    python cli.py train --project revision_BRATS --conf 81
    python cli.py test  --project motion_corrupt --conf 111
    python cli.py test  --project motion_corrupt --conf 111 112 113
    python cli.py test  --project motion_corrupt --conf 111 --set no_clamp=true
"""

import argparse
import sys

import munch

from confs_functions import load_conf, load_conf_test
from trainer import pytorch_trainer


def _parse_overrides(pairs):
    """``key=value`` overrides, parsed as YAML so types survive."""
    import yaml

    out = {}
    for pair in pairs or []:
        if '=' not in pair:
            raise ValueError(f'--set expects key=value, got {pair!r}')
        key, _, value = pair.partition('=')
        out[key.strip()] = yaml.safe_load(value)
    return out


def run_one(mode, project, conf, overrides=None):
    loader = load_conf if mode == 'train' else load_conf_test
    args = loader(conf, project=project, argsToOverride=overrides)
    if args is None:
        raise SystemExit(f'no conf {conf} in project {project!r}')
    args = munch.munchify(args)

    print(f'{mode}: project={project} conf={conf} model={args.modelName}')
    cnn = pytorch_trainer(**args)
    if mode == 'train':
        if not args.Train:
            print(f'  conf_{conf} has Train: false -- nothing to do')
            return
        cnn.train()
    else:
        cnn.test()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n', maxsplit=1)[0])
    p.add_argument('mode', choices=['train', 'test'])
    p.add_argument('--project', required=True, help='conf directory under confs/')
    p.add_argument('--conf', required=True, nargs='+',
                   help='one or more conf numbers')
    p.add_argument('--set', dest='overrides', nargs='*', metavar='KEY=VALUE',
                   help='override conf fields, e.g. --set numEpochs=2 dontLog=true')
    args = p.parse_args(argv)

    overrides = _parse_overrides(args.overrides)
    for conf in args.conf:
        print(f'\n{"=" * 60}\n{args.mode} conf_{conf}\n{"=" * 60}', flush=True)
        run_one(args.mode, args.project, conf, overrides)


if __name__ == '__main__':
    sys.exit(main())
