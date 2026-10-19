#!/usr/bin/env python
"""Plot companion spectrum: truth, direct reconstruction estimate, and/or
joint-regression spectral extraction (with error bars).

Usage:
  python plot_companion_spectrum.py \
      --true    delta_true.txt \
      --est     delta_estimate.txt \
      --spec    companion_spectrum.txt \
      --cov     companion_spectrum_covariance.txt \
      --out     companion_spectrum_comparison.png \
      --mode    both

  All arguments have defaults matching the filenames above.

  --mode selects which estimator(s) are compared against the truth:
      both        truth + direct reconstruction + joint regression (default)
      direct      truth + direct reconstruction only
      regression  truth + joint regression only
  Only the files needed by the selected mode are read, and the fidelity
  metrics, SNR labels, ratio panel and correlation-matrix plot follow the
  same selection.  (The correlation matrix is only produced when the joint
  regression is shown, since it comes from its covariance matrix.)

File formats expected:
  delta_true.txt / delta_estimate.txt:
      Lines of the form  "N.  value"  (1-indexed, one slice per line).
      No header required; lines not matching this pattern are skipped.

  companion_spectrum.txt:
      Output of the companion_spectrum block in Broadband_AI_support.py.
      Lines starting with '#' are comments.  Data lines have columns:
          slice  lambda(nm)  alpha_k  sigma_k  SNR_k
      This file also supplies the wavelength axis.  In --mode direct it is
      still read for the wavelengths; if it is missing, the x-axis falls
      back to the slice index.

  companion_spectrum_covariance.txt:
      Output of the same block: a square matrix of floats (one row per line,
      space-separated, optional leading '#' comment line).  The diagonal
      gives Var(alpha_k); sqrt gives sigma_k (same as companion_spectrum.txt).
"""

import argparse, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── argument parsing ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description='Plot companion spectrum comparison')
ap.add_argument('--true', default='delta_true.txt',
                help='truth delta-function flux per slice')
ap.add_argument('--est',  default='delta_estimate.txt',
                help='direct-reconstruction flux per slice')
ap.add_argument('--spec', default='companion_spectrum.txt',
                help='joint-regression spectrum (companion_spectrum.txt)')
ap.add_argument('--cov',  default='companion_spectrum_covariance.txt',
                help='covariance matrix (companion_spectrum_covariance.txt)')
ap.add_argument('--out',  default='companion_spectrum_comparison.png',
                help='output PNG filename')
ap.add_argument('--title', default='Companion spectrum',
                help='plot title')
ap.add_argument('--mode', default='both',
                choices=['both', 'direct', 'regression'],
                help='which estimator(s) to compare against the truth: '
                     '"both" (default), "direct" (direct reconstruction only), '
                     'or "regression" (joint regression only)')
a = ap.parse_args()

show_direct = a.mode in ('both', 'direct')
show_reg    = a.mode in ('both', 'regression')

# ── loader for "N.  value" files ──────────────────────────────────────────────
def load_indexed(path):
    """Read files whose lines look like  '1.  1234.56'  or  '1.  value'."""
    values = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m = re.match(r'^(\d+)\.\s+([\-\d\.eE\+]+)', line)
            if m:
                idx = int(m.group(1))
                val = float(m.group(2))
                values[idx] = val
    # return as numpy array ordered by index
    keys = sorted(values)
    return np.array([values[k] for k in keys])

# ── load companion_spectrum.txt ───────────────────────────────────────────────
def load_spectrum(path):
    """Returns (lamb, alpha, sigma, snr) arrays."""
    lamb, alpha, sigma, snr = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    # slice  lambda  alpha  sigma  snr
                    lamb.append(float(parts[1]))
                    alpha.append(float(parts[2]))
                    sigma.append(float(parts[3]))
                    snr.append(float(parts[4]))
                except ValueError:
                    continue
    return (np.array(lamb), np.array(alpha),
            np.array(sigma), np.array(snr))

# ── load covariance matrix ────────────────────────────────────────────────────
def load_cov(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(x) for x in line.split()])
    return np.array(rows)

# ── conform an array to the expected number of slices ─────────────────────────
def conform(arr, n, name):
    """Trim or NaN-pad arr to length n, warning if the length differed."""
    if len(arr) == n:
        return arr
    print(f"WARNING: {name} has {len(arr)} entries, spectrum has {n}")
    out = np.full(n, np.nan)
    m = min(len(arr), n)
    out[:m] = arr[:m]
    return out

# ── read data ─────────────────────────────────────────────────────────────────
truth = load_indexed(a.true)

estimate = None
if show_direct:
    estimate = load_indexed(a.est)

lamb = alpha = sigma = snr = None
cov = None

# The spectrum file carries the wavelength axis, so it is read in every mode.
# In --mode direct it is optional: without it we fall back to the slice index.
if os.path.exists(a.spec):
    lamb, alpha, sigma, snr = load_spectrum(a.spec)
elif show_reg:
    ap.error(f"--mode {a.mode} needs the joint-regression spectrum, "
             f"but {a.spec!r} was not found")

have_lamb = lamb is not None and len(lamb) > 0
if not have_lamb:
    print(f"NOTE: no wavelengths available from {a.spec!r}; "
          f"using the slice index as the x-axis")
    lamb = np.arange(1, len(truth) + 1, dtype=float)

nslice   = len(lamb)
xlabel   = 'Wavelength (nm)' if have_lamb else 'Slice index'
xpad     = 20.0 if have_lamb else 0.5

truth = conform(truth, nslice, 'truth')
if show_direct:
    estimate = conform(estimate, nslice, 'estimate')

if show_reg:
    if os.path.exists(a.cov):
        cov = load_cov(a.cov)
        # sigma from covariance diagonal (should match companion_spectrum.txt)
        sigma_cov = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        print(f"NOTE: {a.cov!r} not found; error bars use sigma_k from "
              f"{a.spec!r} and the correlation matrix is skipped")

print(f"Mode     : {a.mode}")
print(f"Loaded {nslice} wavelength slices")
print(f"Truth    : {truth}")
if show_direct:
    print(f"Estimate : {estimate}")
if show_reg:
    print(f"Alpha_k  : {alpha}")
    print(f"Sigma_k  : {sigma}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 9),
                         gridspec_kw={'height_ratios': [3, 1]})

ax = axes[0]

# Truth
ax.plot(lamb, truth, 'o-', color='black', linewidth=2, markersize=8,
        label='Truth', zorder=5)

# Direct reconstruction estimate
if show_direct:
    ax.plot(lamb, estimate, 's--', color='steelblue', linewidth=2,
            markersize=8, label='Direct reconstruction', zorder=4)

# Joint regression with error bars
if show_reg:
    ax.errorbar(lamb, alpha, yerr=sigma, fmt='^:', color='darkorange',
                linewidth=2, markersize=8, capsize=5, capthick=1.5,
                label='Joint regression ± 1σ', zorder=6)

ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
ax.set_ylabel('Companion flux (e⁻)', fontsize=13)
ax.set_title(a.title, fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_xlim(lamb.min() - xpad, lamb.max() + xpad)
# Extend y-range downward so the metrics text box doesn't overlap data
_ymin, _ymax = ax.get_ylim()
ax.set_ylim(_ymin - 0.30 * (_ymax - _ymin), _ymax * 1.05)

# Ratio panel: recovered / truth
ax2 = axes[1]
with np.errstate(divide='ignore', invalid='ignore'):
    if show_direct:
        ratio_est = np.where(truth > 0, estimate / truth, np.nan)
    if show_reg:
        ratio_reg = np.where(truth > 0, alpha / truth, np.nan)
        ratio_err = np.where(truth > 0, sigma / truth, np.nan)

ax2.axhline(1.0, color='black', linewidth=1.2, linestyle='-', label='Unity')
if show_direct:
    ax2.plot(lamb, ratio_est, 's--', color='steelblue',
             linewidth=1.5, markersize=7, label='Direct / truth')
if show_reg:
    ax2.errorbar(lamb, ratio_reg, yerr=ratio_err, fmt='^:', color='darkorange',
                 linewidth=1.5, markersize=7, capsize=4, capthick=1.2,
                 label='Regression / truth')
ax2.axhline(0, color='grey', linewidth=0.5, linestyle=':')
ax2.set_xlabel(xlabel, fontsize=13)
ax2.set_ylabel('Ratio to truth', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
ax2.set_xlim(lamb.min() - xpad, lamb.max() + xpad)
ax2.set_ylim(-0.5, 3.0)

# SNR markers on flux panel (joint regression only)
if show_reg:
    for i, (lam, s) in enumerate(zip(lamb, snr)):
        if abs(s) >= 3:
            ax.annotate(f'{s:.1f}σ', (lam, alpha[i]),
                        textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8, color='darkorange')

# ── spectral fidelity metrics ─────────────────────────────────────────────────
def fidelity_metrics(estimated, truth, label):
    """Compute Pearson r and spectral angle."""
    estimated = np.asarray(estimated, dtype=float)
    truth     = np.asarray(truth,     dtype=float)

    finite = np.isfinite(estimated) & np.isfinite(truth)
    valid  = finite & (truth != 0)

    e, t = estimated[finite], truth[finite]

    if e.size >= 2 and np.std(e) > 0 and np.std(t) > 0:
        pearson = float(np.corrcoef(e, t)[0, 1])
    else:
        pearson = float('nan')

    norms = np.linalg.norm(e) * np.linalg.norm(t)
    if norms > 0:
        cos_theta = np.dot(e, t) / norms
        spectral_angle = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))
    else:
        spectral_angle = float('nan')

    print(f"\n  {label}")
    print(f"    Pearson correlation  : {pearson:.4f}  (1=perfect shape)")
    print(f"    Spectral angle       : {spectral_angle:.2f} deg  (0=perfect shape)")
    return pearson, spectral_angle

print("\nSpectral fidelity metrics:")
print("  " + "=" * 54)
metric_cols = []          # list of (column header, metrics tuple)
if show_direct:
    m_est = fidelity_metrics(estimate, truth, "Direct reconstruction")
    metric_cols.append(('Direct', m_est))
if show_reg:
    m_reg = fidelity_metrics(alpha, truth, "Joint regression")
    metric_cols.append(('Regression', m_reg))
print("  " + "=" * 54)

# Add metrics as text box on the flux panel
_rows = [('Pearson r',            0, '{:>12.3f}'),
         ('Spectral angle (deg)', 1, '{:>12.2f}')]

_lines = ['{:<20}'.format('Fidelity metrics')
          + ''.join('{:>12}'.format(name) for name, _ in metric_cols)]
for _label, _idx, _fmt in _rows:
    _lines.append('{:<20}'.format(_label)
                  + ''.join(_fmt.format(m[_idx]) for _, m in metric_cols))
_metrics_txt = '\n'.join(_lines)

ax.text(0.5, 0.03, _metrics_txt, transform=ax.transAxes,
        fontsize=8, verticalalignment='bottom', ha='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85,
                  edgecolor='grey'))

plt.tight_layout()
plt.savefig(a.out, dpi=150, bbox_inches='tight')
print(f"Saved {a.out}")

# ── correlation matrix plot ───────────────────────────────────────────────────
# Only meaningful for the joint regression: normalise A_inv to get the
# correlation matrix: C_kk' = A_inv_kk' / sqrt(A_inv_kk * A_inv_k'k')
if show_reg and cov is not None:
    _diag = np.sqrt(np.maximum(np.diag(cov), 1e-30))
    corr  = cov / np.outer(_diag, _diag)
    ncorr = corr.shape[0]

    fig2, ax3 = plt.subplots(figsize=(7, 6))
    im = ax3.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    plt.colorbar(im, ax=ax3, label='Correlation coefficient')

    # Annotate each cell with the value
    for i in range(ncorr):
        for j in range(ncorr):
            color = 'white' if abs(corr[i, j]) > 0.6 else 'black'
            ax3.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center',
                     fontsize=7, color=color)

    # Axis labels: wavelength in nm
    _tick_labels = [f'{int(round(l))}' for l in lamb[:ncorr]]
    ax3.set_xticks(range(len(_tick_labels)))
    ax3.set_yticks(range(len(_tick_labels)))
    ax3.set_xticklabels(_tick_labels, rotation=45, ha='right', fontsize=9)
    ax3.set_yticklabels(_tick_labels, fontsize=9)
    ax3.set_xlabel(xlabel, fontsize=12)
    ax3.set_ylabel(xlabel, fontsize=12)
    ax3.set_title('Spectral correlation matrix  ($A^{-1}$ normalised)',
                  fontsize=12)

    plt.tight_layout()
    _corr_out = a.out.replace('.png', '_corrmatrix.png')
    plt.savefig(_corr_out, dpi=150, bbox_inches='tight')
    print(f"Saved {_corr_out}")

# ── text summary ──────────────────────────────────────────────────────────────
_hdr = f"\n{'Slice':>5}  {'lambda':>8}  {'truth':>10}"
if show_direct:
    _hdr += f"  {'estimate':>10}"
if show_reg:
    _hdr += f"  {'alpha_k':>10}  {'sigma_k':>10}  {'SNR':>7}"
_hdr += f"  {'ratio':>7}"
print(_hdr)

for i in range(nslice):
    tr  = truth[i]
    row = f"{i:5d}  {lamb[i]:8.1f}  {tr:10.3e}"
    if show_direct:
        row += f"  {estimate[i]:10.3e}"
    if show_reg:
        row += f"  {alpha[i]:10.3e}  {sigma[i]:10.3e}  {snr[i]:7.2f}"
    # ratio column: regression/truth when shown, otherwise direct/truth
    num = alpha[i] if show_reg else estimate[i]
    ra  = num / tr if tr != 0 else float('nan')
    row += f"  {ra:7.3f}"
    print(row)
