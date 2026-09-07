# SPDX-License-Identifier: MIT
# Copyright (c) 2026 the FARD authors. See LICENSE.

"""Configuration for the registration pipeline.

Replaces the module-level constants and ``if __name__`` literals in
``data/register_*.py``. Every path, case list and scan name that used to be
edited in source is a field here.
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

_PLACEHOLDER = re.compile(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}')


def expand(raw: dict) -> dict:
    """Resolve ``${NAME}`` in config values.

    Roots are named symbolically so a config is not tied to one machine. A name
    resolves from the environment first, then from the ``paths:`` block of
    ``local.yaml`` at the repository root. An unresolved name raises, rather than
    yielding a path with a literal ``${...}`` that fails confusingly later.
    """
    local = {}
    localPath = Path(__file__).resolve().parent.parent / 'local.yaml'
    if localPath.exists():
        local = (yaml.safe_load(localPath.read_text()) or {}).get('paths', {}) or {}

    def resolve(value):
        if not isinstance(value, str) or '${' not in value:
            return value
        missing = []

        def sub(m):
            got = os.environ.get(m.group(1), local.get(m.group(1)))
            if got is None:
                missing.append(m.group(1))
                return m.group(0)
            return str(got)

        out = _PLACEHOLDER.sub(sub, value)
        if missing:
            raise KeyError(
                f'unresolved path placeholder(s) {sorted(set(missing))} in {value!r}. '
                f'Set them in the environment, or add a `paths:` entry to local.yaml '
                f'(copy local.yaml.example).')
        return out

    return {k: resolve(v) for k, v in (raw or {}).items()}


@dataclass
class RegistrationConfig:
    """One registration stage over one dataset.

    Attributes:
        root: dataset root holding the per-fold directories.
        fold_template: directory name per fold; ``{fold}`` is substituted.
        folds: folds to process.
        input_stage: stage directory read from, under each fold.
        output_stage: stage directory written to, under each fold.
        atlas: DICOM directory of the MNI atlas the transform is estimated against.
        reference_scan: the scan the atlas transform is FITTED to. The README's
            ``high_quality_based_registration`` switch is exactly this choice --
            a high-quality scan (``T1_HR``) for training data, a low-quality one
            (``BICUBIC_T1_LR``) for test data. Naming the scan directly removes
            the boolean and makes the choice visible in the config.
        moving_scans: the other scans the fitted transform is applied to.
        cases: case directories to process; ``None`` means every case found under
            ``input_stage``, sorted numerically.
        sampling_seed: seed for the registration metric's random sampling. The
            original code called ``SetMetricSamplingPercentage(0.01)`` and took the
            default seed, which is ``sitkWallClock`` -- so every run produced a
            slightly different transform and no registration was reproducible.
            An int makes it deterministic; ``None`` restores the old wall-clock
            behaviour. The stored ``registration_C`` predates this and was produced
            with wall-clock seeding, so it cannot be reproduced bit-exact whatever
            is set here.
        anonymize: de-identify each written series. The registration writer copies
            Patient Name/ID/Birth Date and Accession Number from the source series,
            so output is NOT shareable unless this is on or a separate
            anonymization pass is run afterwards.
        skip_missing: skip a case whose reference scan is absent instead of raising.
    """

    root: Path
    atlas: Path
    reference_scan: str
    moving_scans: List[str]
    output_root: Optional[Path] = None
    input_stage: str = 'registration_B'
    output_stage: str = 'registration_C'
    fold_template: str = 'T1_T2_Flair_fixed_{fold}'
    folds: List[int] = field(default_factory=lambda: [1])
    cases: Optional[List[str]] = None
    sampling_seed: Optional[int] = 42
    anonymize: bool = False
    skip_missing: bool = False

    def __post_init__(self):
        self.root = Path(self.root)
        self.atlas = Path(self.atlas)
        self.output_root = Path(self.output_root) if self.output_root else self.root
        if not self.atlas.is_dir():
            raise FileNotFoundError(
                f'atlas directory not found: {self.atlas}\n'
                'Set `atlas:` in the config to the MNI DICOM directory.')
        if not self.root.is_dir():
            raise FileNotFoundError(
                f'dataset root not found: {self.root}\nSet `root:` in the config.')
        if self.reference_scan in self.moving_scans:
            raise ValueError(
                f'reference_scan {self.reference_scan!r} also appears in moving_scans; '
                'it is resampled automatically and must not be listed twice.')

    @classmethod
    def from_yaml(cls, path) -> 'RegistrationConfig':
        path = Path(path)
        raw = expand(yaml.safe_load(path.read_text()))
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(
                f'{path}: unknown config keys {sorted(unknown)}. '
                f'Valid keys: {sorted(cls.__dataclass_fields__)}')
        return cls(**raw)

    def fold_dir(self, fold: int) -> Path:
        return self.root / self.fold_template.format(fold=fold)

    def input_dir(self, fold: int) -> Path:
        return self.fold_dir(fold) / self.input_stage

    def output_dir(self, fold: int) -> Path:
        """Output stage for ``fold``. Rooted at ``output_root`` so derived stages
        can be written to a different disk than the read-only source dataset."""
        return self.output_root / self.fold_template.format(fold=fold) / self.output_stage

    def case_list(self, fold: int) -> List[str]:
        """Cases to process for ``fold`` -- explicit if configured, else discovered."""
        if self.cases is not None:
            return [str(c) for c in self.cases]
        in_dir = self.input_dir(fold)
        if not in_dir.is_dir():
            raise FileNotFoundError(f'input stage not found: {in_dir}')
        found = [p.name for p in in_dir.iterdir() if p.is_dir()]
        return sorted(found, key=lambda c: (not c.isdigit(), int(c) if c.isdigit() else c))


@dataclass
class LR2HRConfig:
    """Stage A / Stage B of the LR-HR and cross-modal registration.

    Replaces the literals in ``data/register_LR2HR.py``'s ``if __name__`` block,
    including ``high_quality_based_registration`` (now ``hq_based``) and the
    hardcoded two-modality scan names.

    Attributes:
        modalities: contrasts present per case, e.g. ``[T1, T2, FLAIR]``.
        reference_modality: the contrast everything else is registered onto.
        hq_based: fit on the high-quality scans (training) or the low-quality
            ones (test). This is the README's
            ``high_quality_based_registration`` switch.
        hq_name/lq_name: scan directory templates; ``{mod}`` is substituted.
        companions: extra scans per modality that the fitted transform is
            replayed onto, as templates.
    """

    root: Path
    modalities: List[str]
    reference_modality: str
    output_root: Optional[Path] = None
    input_stage: str = 'original'
    stage_a_template: str = 'registration_A_{mod}'
    stage_b_stage: str = 'registration_B'
    hq_name: str = '{mod}_HR'
    lq_name: str = 'BICUBIC_{mod}_LR'
    hq_based: bool = True
    companions: List[str] = field(default_factory=lambda: ['BICUBIC_{mod}_LR'])
    # Stage A's stored parameter files use 512 iterations, not elastix's stock 256.
    # Stage B used the stock map unmodified.
    stage_a_parameters: dict = field(
        default_factory=lambda: {'MaximumNumberOfIterations': 512})
    stage_b_parameters: dict = field(default_factory=dict)
    fold_template: str = 'T1_T2_Flair_fixed_{fold}'
    folds: List[int] = field(default_factory=lambda: [1])
    cases: Optional[List[str]] = None
    skip_missing: bool = False

    def __post_init__(self):
        self.root = Path(self.root)
        self.output_root = Path(self.output_root) if self.output_root else self.root
        if not self.root.is_dir():
            raise FileNotFoundError(f'dataset root not found: {self.root}')
        if self.reference_modality not in self.modalities:
            raise ValueError(
                f'reference_modality {self.reference_modality!r} is not in '
                f'modalities {self.modalities}')

    from_yaml = classmethod(RegistrationConfig.from_yaml.__func__)

    def fold_dir(self, fold: int) -> Path:
        return self.root / self.fold_template.format(fold=fold)

    def input_dir(self, fold: int) -> Path:
        return self.fold_dir(fold) / self.input_stage

    def stage_a_out_dir(self, fold: int, mod: str) -> Path:
        """Where stage A WRITES -- under output_root."""
        return (self.output_root / self.fold_template.format(fold=fold)
                / self.stage_a_template.format(mod=mod))

    def stage_a_in_dir(self, fold: int, mod: str) -> Path:
        """Where stage B READS stage-A output from -- under root, so that
        redirecting output_root does not also redirect stage B's inputs."""
        return (self.root / self.fold_template.format(fold=fold)
                / self.stage_a_template.format(mod=mod))

    def stage_b_dir(self, fold: int) -> Path:
        return (self.output_root / self.fold_template.format(fold=fold)
                / self.stage_b_stage)

    def companion_scans(self, mod: str, already_done: str) -> List[str]:
        names = [c.format(mod=mod) for c in self.companions]
        return [n for n in names if n != already_done]

    case_list = RegistrationConfig.case_list
