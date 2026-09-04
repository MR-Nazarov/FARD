"""
Score the misregistration sweep via the Post_processing tester and merge into
misregistration_metrics_all.csv (SAME schema as motion_corrupt_metrics_all.csv).

The tester emits Pred + ONE Input group per run, so each of the 3 output-modality
group confs (1=T1n, 2=T2w, 3=FLAIR) is run TWICE:
  * CorruptInput run: input_per_conf active -> self confs use the misregistered
    target input; guide confs fall back to `input` (clean). Input = CorruptInput.
  * CleanInput   run: input_per_conf overridden to {} -> all confs use the clean
    target input. Input = CleanInput.
Pred is identical in both runs (same saved output vs GT); we take Pred+CleanInput
from the clean run and CorruptInput from the corrupt run.

AVG row format (tab-separated): conf, 'AVG', then 16 values =
  Pred[SSIM,SSIM_std,PSNR,PSNR_std,FSIM,FSIM_std,LPIPS,LPIPS_std] + Input[...same...]
"""
import io, re, csv, contextlib, sys
from tester import main as tester_main

GROUPS = [1, 2, 3]
METRICS = ['SSIM', 'PSNR', 'FSIM', 'LPIPS']
AVG_RE = re.compile(r'^(\d+)\tAVG\t(.+)$')


def run_and_parse(conf, override=None):
    """Run one group conf, capture stdout, return {conf_num: [16 floats]}."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tester_main(project='misregistration', conf=conf, argsToOverride=override)
    out = buf.getvalue()
    sys.stderr.write(out)                       # keep the tester log visible on stderr
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
    mod = {1: 'T1n', 2: 'T2w', 3: 'FLAIR'}
    sev = {1: 'mild', 2: 'moderate', 3: 'severe'}
    return mod[x], mod[y], sev[z], ('self' if x == y else 'guide')


def main():
    clean, corr = {}, {}
    for g in GROUPS:
        sys.stderr.write(f'\n===== group {g}: CorruptInput run (input_per_conf active) =====\n')
        corr.update(run_and_parse(g))
        sys.stderr.write(f'\n===== group {g}: CleanInput run (input_per_conf disabled) =====\n')
        clean.update(run_and_parse(g, override={'input_per_conf': {}}))

    cols = ['conf', 'output_modality', 'corrupted_contrast', 'corruption_type', 'severity', 'role']
    for grp in ['Pred', 'CleanInput', 'CorruptInput']:
        for mt in METRICS:
            cols += [f'{grp}_{mt}', f'{grp}_{mt}_std']

    confs = sorted(set(clean) & set(corr))
    missing = sorted((set(clean) | set(corr)) - set(confs))
    out_path = 'misregistration_metrics_all.csv'
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in confs:
            omod, cmod, sev, role = meta(c)
            pred = clean[c][0:8]                 # Pred (from clean run)
            cleanin = clean[c][8:16]             # Input from clean run  = CleanInput
            corrin = corr[c][8:16]               # Input from corrupt run = CorruptInput
            w.writerow([c, omod, cmod, 'misregistration', sev, role]
                       + [round(v, 4) for v in pred + cleanin + corrin])

    print(f'\nwrote {out_path}: {len(confs)}/27 confs   missing={missing}')
    # sanity: guide CorruptInput should equal CleanInput; self should differ (degrade)
    print(f"{'conf':>5} {'role':<6} {'Pred_SSIM':>9} {'CleanIn_SSIM':>12} {'CorrIn_SSIM':>12} {'ΔSSIN(corr-clean)':>18}")
    for c in confs:
        _, _, _, role = meta(c)
        pin = clean[c][0]; ci = clean[c][8]; ki = corr[c][8]
        print(f"{c:>5} {role:<6} {pin:>9.4f} {ci:>12.4f} {ki:>12.4f} {ki-ci:>18.4f}")


if __name__ == '__main__':
    main()
