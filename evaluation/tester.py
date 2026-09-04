import os as _os, sys as _sys
# evaluation/ imports the shared confs_functions and losses from the repo root
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in (_ROOT, _os.path.join(_ROOT, "src"), _os.path.dirname(_os.path.abspath(__file__))):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import munch
from confs_functions import load_conf
from pathlib import Path
from testing_class import MetricsCalculator
import os


def main(project, conf, argsToOverride=None):
    # Get the absolute path to the confs directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    confs_path = current_dir / 'confs'  # Assuming confs is one level up

    # Load configuration
    args = load_conf(conf,
                     project=project,
                     argsToOverride=argsToOverride,
                     path=confs_path)  # Pass the explicit path

    if args is None:
        raise ValueError(f"Failed to load configuration for project '{project}' conf '{conf}'. "
                         f"Checked path: {confs_path}")

    args = munch.munchify(args)

    print('Using device:', args.device if hasattr(args, 'device') else 'cuda')

    calculator = MetricsCalculator(args)
    calculator.calculate()


if __name__ == '__main__':
    # BRATS_WT scoring
    # conf_2: revision_BRATS conf_88 (T1n, 150 test cases, SSIM/PSNR/FSIM/LPIPS)
    # conf_3: revision_BRATS conf_89 (T2w, 150 test cases, SSIM/PSNR/FSIM/LPIPS)
    # conf_4: BRATS_WT conf_32       (T2w, 271 training cases, MAE+WT+Ring)
    # conf_5: BRATS_WT conf_63       (FLAIR, 271 training cases, MAE+WT+Ring)
    project = 'BRATS'
    # conf_23: case 117 slice 113, all models, MMMD target-matched subdir
    for conf in [23]:
        print(f'\n{"="*60}\nScoring 3_modal_sheba conf_{conf}\n{"="*60}', flush=True)
        main(project=project, conf=conf)