"""
Score the EDSR_MHCA motion+ghosting robustness sweep via the Post_processing tester
and merge into motion_corrupt_metrics_all_edsr.csv (BYTE-identical schema to
motion_corrupt_metrics_all.csv, so downstream plotting reads it unchanged).

Matches the FARD motion scoring convention EXACTLY (input_per_conf lists all 9
confs -> guide CorruptInput = corrupted guide), so FARD vs DnCNN differ only by
the model. EDSR's saved pred is transposed+flipped (218x182) vs GT (182x218); the group
confs set skip_volume_alignment:false so the tester permutes GT/input to match.

Optimization (exact, verified against testing_class.calculate_metrics, which
scores output-vs-target only, and the FARD CSV, where CleanInput is constant per
target modality):
  * Pred is input-independent  -> take Pred AND CorruptInput from ONE corrupt run
    per group (input_per_conf active).            [54 conf-scorings]
  * CleanInput depends only on the target modality -> compute it once per target
    from a single clean-input scoring.            [ 3 conf-scorings]
Total 57 vs 108 for the naive double-run.
"""
import io, re, csv, contextlib, sys
from tester import main as tester_main

PROJECT = 'BRATS/motion_corrupt_edsr'
GROUPS = [1, 2, 3, 4, 5, 6]
# target modality -> (group conf, representative conf) for the one clean-input scoring
CLEAN_TARGET_CONF = {'T1n': (1, 111), 'T2w': (2, 211), 'FLAIR': (3, 311)}
METRICS = ['SSIM', 'PSNR', 'FSIM', 'LPIPS']
AVG_RE = re.compile(r'^(\d+)\tAVG\t(.+)$')


def run_and_parse(conf, override=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tester_main(project=PROJECT, conf=conf, argsToOverride=override)
    out = buf.getvalue()
    sys.stderr.write(out)
    rows = {}
    for line in out.splitlines():
        m = AVG_RE.match(line)
        if m:
            vals = [float(x) for x in m.group(2).split('\t')]
            if len(vals) >= 16:
                rows[int(m.group(1))] = vals[:16]
    return rows


def meta(c):
    x, y, z = c // 100, (c % 100) // 10, c % 10
    out_mod = {1: 'T1n', 2: 'T2w', 3: 'FLAIR', 4: 'T1n', 5: 'T2w', 6: 'FLAIR'}
    corr_mod = {1: 'T1n', 2: 'T2w', 3: 'FLAIR'}
    sev = {1: 'mild', 2: 'moderate', 3: 'severe'}
    ctype = 'ghosting' if x <= 3 else 'motion'
    x_norm = x if x <= 3 else x - 3
    role = 'self' if x_norm == y else 'guide'
    return out_mod[x], corr_mod[y], ctype, sev[z], role


def main():
    # 1) CorruptInput run per group -> Pred (input-independent) + CorruptInput per conf
    corr = {}
    for g in GROUPS:
        sys.stderr.write(f'\n===== group {g}: CorruptInput run (input_per_conf active) =====\n')
        corr.update(run_and_parse(g))

    # 2) One clean-input scoring per target -> CleanInput (constant across that target's confs)
    clean_ct = {}
    for tgt, (g, c) in CLEAN_TARGET_CONF.items():
        sys.stderr.write(f'\n===== target {tgt}: CleanInput run (conf {c}, input_per_conf disabled) =====\n')
        rows = run_and_parse(g, override={'input_per_conf': {}, 'confs_to_process': [c]})
        if c not in rows:
            raise RuntimeError(f'CleanInput run for {tgt} produced no AVG row for conf {c}')
        clean_ct[tgt] = rows[c][8:16]

    cols = ['conf', 'output_modality', 'corrupted_contrast', 'corruption_type', 'severity', 'role']
    for grp in ['Pred', 'CleanInput', 'CorruptInput']:
        for mt in METRICS:
            cols += [f'{grp}_{mt}', f'{grp}_{mt}_std']

    confs = sorted(corr)
    out_path = 'motion_corrupt_metrics_all_edsr.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in confs:
            omod, cmod, ctype, sev, role = meta(c)
            pred = corr[c][0:8]
            corrin = corr[c][8:16]
            cleanin = clean_ct[omod]
            w.writerow([c, omod, cmod, ctype, sev, role]
                       + [round(v, 4) for v in pred + cleanin + corrin])

    print(f'\nwrote {out_path}: {len(confs)}/54 confs')
    print(f"{'conf':>5} {'role':<6} {'type':<9} {'Pred_SSIM':>9} {'CleanIn':>8} {'CorrIn':>8} {'Δ(corr-clean)':>14}")
    for c in confs:
        omod, _, ctype, _, role = meta(c)
        pin = corr[c][0]; ci = clean_ct[omod][0]; ki = corr[c][8]
        print(f"{c:>5} {role:<6} {ctype:<9} {pin:>9.4f} {ci:>8.4f} {ki:>8.4f} {ki-ci:>14.4f}")


if __name__ == '__main__':
    main()
