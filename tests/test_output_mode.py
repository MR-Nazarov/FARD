"""Output-mode resolution must stay behaviour-identical to the legacy predicate.

Run standalone (no pytest needed):
    PYTHONPATH=src python tests/test_output_mode.py

The cases below are drawn from the real conf population: resolving every conf in
the predecessor repo (1440 of them) produced 1051 round / 184 clamp / 205 none,
with zero divergence from the legacy branch. These pin the distinctive corners of
that mapping so a later refactor cannot quietly move one.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fard.output_mode import (  # noqa: E402
    OutputMode,
    legacy_output_mode,
    legacy_train_round,
    resolve,
)

# (project_name, round, no_clamp) -> (eval mode, train_round)
CASES = [
    # BRATS proper: clamps at eval, and is the one project that does NOT round in
    # training -- the train branch tests equality against exactly this name.
    (('BRATS', True, False), (OutputMode.CLAMP, False)),

    # BRATS_WT: clamps at eval via the substring test, but DOES round in training
    # because it is not equal to 'BRATS'. 41 confs sit in this inconsistent state.
    (('BRATS_WT', True, False), (OutputMode.CLAMP, True)),
    (('revision_BRATS', True, False), (OutputMode.CLAMP, True)),

    # Named in the clamp tuple rather than matched by substring.
    (('motion_corrupt', True, False), (OutputMode.CLAMP, True)),
    (('corrupt_test', True, False), (OutputMode.CLAMP, True)),

    # Backbone sweeps: NOT in the tuple and no 'BRATS' substring, so they round.
    # The test confs that produced the published §5 numbers all set no_clamp,
    # which is what keeps that comparison symmetric -- see the no_clamp cases.
    (('motion_corrupt_dncnn', True, False), (OutputMode.ROUND, True)),
    (('motion_corrupt_edsr', True, False), (OutputMode.ROUND, True)),

    # no_clamp wins over everything at eval, but does not affect training.
    (('motion_corrupt', True, True), (OutputMode.NONE, True)),
    (('motion_corrupt_dncnn', True, True), (OutputMode.NONE, True)),
    (('BRATS', True, True), (OutputMode.NONE, False)),

    # DICOM-intensity projects round; round:false leaves output alone.
    (('3_modal_sheba', True, False), (OutputMode.ROUND, True)),
    # round:false leaves output alone at eval AND suppresses the training round --
    # train_round is (name != 'BRATS') AND round, so it needs round to be on.
    (('general_project', False, False), (OutputMode.NONE, False)),
]


def test_legacy_mapping():
    for (pn, rnd, nc), (want_mode, want_tr) in CASES:
        got_mode = legacy_output_mode(pn, rnd, nc)
        got_tr = legacy_train_round(pn, rnd)
        assert got_mode is want_mode, f'{pn!r} round={rnd} no_clamp={nc}: {got_mode} != {want_mode}'
        assert got_tr == want_tr, f'{pn!r} round={rnd}: train_round {got_tr} != {want_tr}'


def test_resolve_matches_legacy_when_unspecified():
    for (pn, rnd, nc), expected in CASES:
        got = resolve({'project_name': pn, 'round': rnd, 'no_clamp': nc})
        assert got == expected, f'{pn!r}: {got} != {expected}'


def test_explicit_fields_override():
    conf = {'project_name': 'BRATS', 'round': True, 'no_clamp': False}
    assert resolve(conf) == (OutputMode.CLAMP, False)          # legacy
    conf2 = {**conf, 'output_mode': 'none', 'train_round': True}
    assert resolve(conf2) == (OutputMode.NONE, True)           # explicit wins


def test_apply():
    import torch
    x = torch.tensor([-0.5, 0.4, 1.7])
    assert torch.equal(fard_apply(x, OutputMode.NONE), x)
    assert torch.equal(fard_apply(x, OutputMode.CLAMP), torch.tensor([0.0, 0.4, 1.0]))
    assert torch.equal(fard_apply(x, OutputMode.ROUND), torch.tensor([-0.0, 0.0, 2.0]))
    # lists are handled elementwise
    out = fard_apply([x, x], OutputMode.CLAMP)
    assert isinstance(out, list) and len(out) == 2


def fard_apply(output, mode):
    from fard.output_mode import apply
    return apply(output, mode)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'  ok  {name}')
    print('\nall output-mode tests passed')
