"""
Score all 15 reg-B-in-MNI confs (441-465, project 3_modal_sheba_mni) via the tester,
collect Pred SSIM/PSNR/FSIM/LPIPS, and print the ablation table alongside the existing
MNI 1-ch / 2-ch numbers from sheba_ablation_scores.csv.

FSIM/LPIPS come off the tester on the [0,1] scale; SSIM/PSNR are directly comparable.
(FSIM@255 / LPIPS@raw for the CSV scale are recomputed separately.)
Writes regb_in_mni_scores.csv.
"""
import io, re, csv, contextlib, sys, os
# The scoring confs use paths relative to the evaluation directory, so run
# from there rather than from wherever the caller happens to be.
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tester import main as tester_main

CONFS = [c for x in (44, 45, 46) for c in range(x * 10 + 1, x * 10 + 6)]
MODN = {44: 'T1', 45: 'T2', 46: 'FLAIR'}
AVG_RE = re.compile(r'^(\d+)\tAVG\t(.+)$')


def score(conf):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tester_main(project='3_modal_sheba_mni', conf=conf, argsToOverride={'dontLog': True})
    out = buf.getvalue(); sys.stderr.write(out)
    for line in out.splitlines():
        m = AVG_RE.match(line)
        if m and int(m.group(1)) == conf:
            v = [float(x) for x in m.group(2).split('\t')]
            return v[:8]   # Pred SSIM,std,PSNR,std,FSIM,std,LPIPS,std
    return None


def main():
    rows = []
    for c in CONFS:
        r = score(c)
        if r is None:
            sys.stderr.write(f"conf {c}: NO AVG\n"); continue
        rows.append(dict(conf=c, modality=MODN[c // 10], fold=c % 10,
                         SSIM=round(r[0], 4), PSNR=round(r[2], 4),
                         FSIM01=round(r[4], 4), LPIPS01=round(r[6], 4)))
        sys.stderr.write(f"conf {c} ({MODN[c//10]} f{c%10}): SSIM={r[0]:.4f} PSNR={r[2]:.2f}\n")
    with open('regb_in_mni_scores.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['conf', 'modality', 'fold', 'SSIM', 'PSNR', 'FSIM01', 'LPIPS01'])
        w.writeheader(); w.writerows(rows)
    # per-modality means
    print("\n=== reg-B 3-ch in MNI (mean over folds) ===")
    print(f"{'mod':>6} {'SSIM':>7} {'PSNR':>7}")
    for mod in ('T1', 'T2', 'FLAIR'):
        sub = [r for r in rows if r['modality'] == mod]
        if sub:
            import statistics as st
            print(f"{mod:>6} {st.mean(r['SSIM'] for r in sub):>7.4f} {st.mean(r['PSNR'] for r in sub):>7.3f}")
    print(f"\nwrote regb_in_mni_scores.csv ({len(rows)} confs)")


if __name__ == '__main__':
    main()
