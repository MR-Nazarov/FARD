# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""How a model's raw output is post-processed before metrics and saving.

In the predecessor codebase this was decided by string-matching the wandb project
name inside ``trainer.py``:

    # eval (trainer.py:1360, 1375)
    if (not no_clamp) and ('BRATS' in project_name
                           or project_name in ('motion_corrupt', 'corrupt_test')):
        clamp(0, 1)
    elif round and not no_clamp:
        round()

    # train (trainer.py:574, 726)
    if project_name != 'BRATS' and round:
        round()

Two problems with that. Renaming a project silently changed the arithmetic; and
the two sites disagree — eval tests ``'BRATS' in name`` (substring) while train
tests ``name != 'BRATS'`` (equality), so a ``BRATS_WT`` run clamps at eval and
rounds during training. 41 confs are in that state.

Here the choice is data: ``output_mode`` and ``train_round`` are conf fields.
``resolve()`` fills them from the legacy predicate when absent, so confs that
predate this behave exactly as before and can be migrated one at a time.
"""

from enum import Enum


class OutputMode(str, Enum):
    """What to do with raw model output at evaluation time."""

    CLAMP = 'clamp'   # clamp to [0, 1]; inputs normalised to unit range
    ROUND = 'round'   # round to integers; raw DICOM intensities
    NONE = 'none'     # leave untouched


#: Projects whose data is [0,1]-normalised, in the legacy predicate's terms.
_LEGACY_CLAMP_PROJECTS = ('motion_corrupt', 'corrupt_test')
_LEGACY_CLAMP_SUBSTRING = 'BRATS'
_LEGACY_TRAIN_NO_ROUND_PROJECT = 'BRATS'


def legacy_output_mode(project_name, round_, no_clamp) -> OutputMode:
    """Reproduce the eval-time branch exactly as trainer.py:1360/1375 had it."""
    project_name = str(project_name)
    if not no_clamp and (_LEGACY_CLAMP_SUBSTRING in project_name
                         or project_name in _LEGACY_CLAMP_PROJECTS):
        return OutputMode.CLAMP
    if round_ and not no_clamp:
        return OutputMode.ROUND
    return OutputMode.NONE


def legacy_train_round(project_name, round_) -> bool:
    """Reproduce the train-time branch exactly as trainer.py:574/726 had it.

    Note the equality test, against eval's substring test. Preserved rather than
    corrected: changing it changes trained-model behaviour.
    """
    return bool(str(project_name) != _LEGACY_TRAIN_NO_ROUND_PROJECT and round_)


def resolve(args):
    """Return ``(output_mode, train_round)`` for a conf.

    Uses the explicit ``output_mode`` / ``train_round`` fields when present,
    otherwise derives them from the legacy project-name predicate. Reading them
    through this function is what lets ``project_name`` go back to being a label.
    """
    get = args.get if hasattr(args, 'get') else (lambda k, d=None: getattr(args, k, d))

    explicit = get('output_mode', None)
    if explicit is not None:
        mode = OutputMode(explicit)
    else:
        mode = legacy_output_mode(get('project_name', ''),
                                  get('round', False),
                                  get('no_clamp', False))

    train_round = get('train_round', None)
    if train_round is None:
        train_round = legacy_train_round(get('project_name', ''), get('round', False))

    return mode, bool(train_round)


def apply(output, mode: OutputMode):
    """Apply ``mode`` to a tensor or list of tensors."""
    import torch

    if mode is OutputMode.NONE:
        return output
    op = ((lambda t: torch.clamp(t, 0, 1)) if mode is OutputMode.CLAMP
          else torch.round)
    if isinstance(output, list):
        return [op(t) for t in output]
    return op(output)
