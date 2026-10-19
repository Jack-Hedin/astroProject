#!/usr/bin/env python
"""Blind point-source detection on mf_coadd_hpf.fits.

Merge of the two earlier versions, keeping the correct pieces of each:

  from the ORIGINAL version
    - convergence on the FULL candidate set (not just detections), so the
      mask that feeds sigma(r) is guaranteed stable at exit
    - no sign-flip null (the rot90 trials were near-degenerate because
      sigma(r) is rotationally symmetric, so they carried ~1 trial of
      information, not 4)

  from the APERTURE version
    - aperture photometry statistic and analytic FAP

  FIXED here
    1. Detection decision is made on the APERTURE SNR, as documented.
       The earlier version thresholded the peak SNR and only reported the
       aperture value.
    2. The pixel-covariance kernel rho now includes the high-pass filter.
       The map is mf_coadd - gaussian(mf_coadd, mf_smooth_sigma), and the
       HPF removes most of the long-range positive correlation. Ignoring it
       overestimates the aperture noise by a factor of ~3 at aprad=1 FWHM.
       Pass --smooth to match mf_smooth_sigma in the pipeline.
    3. rho is now mean_l |FFT(h_l)|^2, not |FFT(mean_l h_l)|^2. The coadd is
       (1/N) sum_l r_l (x) h_l with independent per-frame noise, so the
       covariance is the mean of the per-frame autocorrelations.
    4. Radial MEAN is subtracted as well as dividing by the radial sigma.
       Both earlier versions assumed mu(r) = 0.
    5. Aperture sum uses a local background annulus (the detection-side
       analogue of adding a nuisance constant to a design matrix). Disable
       with --no-bg.
    6. Small-sample Student-t FAP (Mawet et al. 2014) instead of erfc, and
       the blind-search resolution-element count is reported alongside the
       annular one. The annular count alone understates the trials factor
       for a search over all radii.
    7. Aperture autocorrelation computed once on a small lag grid via direct
       correlation -- no FFT padding, so no parity/wraparound failure modes.

Usage:
  python detect_sources_merged.py <run_dir_or_fits>
        [--center CY CX] [--fwhm 4] [--smooth 3]
        [--thresh 5] [--cand 3.5] [--binw 4] [--aprad 1.0]
        [--bg-in 1.5] [--bg-out 3.0] [--no-bg]
        [--fap-mode blind|annular] [--stat t|gauss]
        [--rmax R] [--plot] [--null]

  --center    satellite center, numpy (row, col). Default: image center.
              DS9 reports (x, y) 1-based with a y-flip relative to the
              internal frame: row = (dim-1) - (ds9_y - 1), col = ds9_x - 1.
  --fwhm      PSF FWHM in pixels (mask radius = 2*FWHM, min peak sep = FWHM).
  --smooth    Gaussian sigma [px] used for the high-pass filter in the
              pipeline (mf_smooth_sigma). MUST match, or the aperture noise
              is wrong. Set 0 if the input map was not high-pass filtered.
  --thresh    detection threshold, in APERTURE SNR units.
  --cand      provisional PEAK SNR threshold for candidacy and for masking
              during the sigma(r) iteration. Keep below --thresh.
  --binw      annulus width in pixels for the mu(r)/sigma(r) profile.
  --aprad     aperture radius in units of FWHM (default 1.0).
  --bg-in/out local background annulus radii, in units of the aperture
              radius (default 1.5 to 3.0).
  --fap-mode  'blind'   : N_res over the whole searched area (default).
              'annular' : N_res = 2*pi*r/FWHM, the direct-imaging convention,
                          appropriate only when the separation was known
                          a priori.
  --stat      't'     : Student-t with the small-sample penalty (default).
              'gauss' : Gaussian erfc, valid only when the annulus holds
                        many independent resolution elements.
  --rmax      ignore radii beyond this (pixels). Default: largest radius at
              which the annulus is still azimuthally complete, i.e. the
              inscribed radius. Beyond it the annuli are corner fragments.

Outputs: detections.txt, mf_coadd_hpf_wsnr.fits, mf_coadd_hpf_sigma.fits,
         mf_coadd_hpf_mu.fits, optionally sigma_vs_radius.png.
"""
import sys, os, argparse
import numpy as np
from astropy.io import fits
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from scipy.special import erfc
from scipy.stats import t as student_t

MIN_ANN_PIX = 20          # minimum unmasked pixels for an annulus to be used

ap = argparse.ArgumentParser()
ap.add_argument('path')
ap.add_argument('--center', nargs=2, type=float, default=None, metavar=('CY', 'CX'))
ap.add_argument('--fwhm',   type=float, default=4.0)
ap.add_argument('--smooth', type=float, default=3.0,
                help='mf_smooth_sigma used for the HPF (0 = map not filtered)')
ap.add_argument('--thresh', type=float, default=5.0)
ap.add_argument('--cand',   type=float, default=3.0)
ap.add_argument('--at', nargs=2, type=int, default=None, metavar=('Y', 'X'),
                help='evaluate the detection statistics at this KNOWN position '
                     '(numpy row, col) and report the SINGLE-TRIAL p-value '
                     'instead of the blind-search FAP. Use for follow-up or '
                     'injection tests where the position is specified in '
                     'advance; the blind N_res trials penalty does not apply '
                     'to a position you did not search for. Bypasses --cand '
                     'entirely -- the position is evaluated whether or not it '
                     'is a candidate.')
ap.add_argument('--binw',   type=float, default=4.0)
ap.add_argument('--aprad',  type=float, default=1.0)
ap.add_argument('--bg-in',  type=float, default=1.5, dest='bg_in')
ap.add_argument('--bg-out', type=float, default=3.0, dest='bg_out')
ap.add_argument('--no-bg',  action='store_true', dest='no_bg')
ap.add_argument('--fap-mode', choices=['blind', 'annular'], default='blind')
ap.add_argument('--stat',     choices=['t', 'gauss'], default='t')
ap.add_argument('--rmax',   type=float, default=None)
ap.add_argument('--apscan', action='store_true',
                help='report aperture SNR vs aperture radius for each candidate')
ap.add_argument('--plot',   action='store_true')
ap.add_argument('--null',   action='store_true')
a = ap.parse_args()

# ---------------------------------------------------------------- load ----
p = a.path
if os.path.isdir(p):
    outdir   = p
    hpf_path = os.path.join(p, 'mf_coadd_hpf.fits')
    psf_path = os.path.join(p, 'psf_mean.fits')
else:
    outdir   = os.path.dirname(p) or '.'
    hpf_path = p
    psf_path = os.path.join(outdir, 'psf_mean.fits')

img = fits.getdata(hpf_path).astype(float)
if img.ndim != 2:
    sys.exit(f"error: expected a 2D map, got shape {img.shape}")
ny, nx = img.shape
cy, cx = a.center if a.center else (ny / 2.0, nx / 2.0)

yy, xx = np.indices(img.shape)
r    = np.hypot(yy - cy, xx - cx)
rbin = (r / a.binw).astype(int)
nb   = rbin.max() + 1

# Largest radius at which an annulus is still azimuthally complete.
r_inscribed = min(cy, ny - 1 - cy, cx, nx - 1 - cx)
rmax = a.rmax if a.rmax is not None else r_inscribed
if a.rmax is None:
    print(f"search radius capped at the inscribed radius: rmax = {rmax:.1f} px "
          f"(beyond this the annuli are corner fragments; override with --rmax)")
search_mask = (r <= rmax) & (r >= a.fwhm)   # exclude the very centre too

aprad_px = a.aprad * a.fwhm

# ------------------------------------------------- covariance kernel rho ----
# Var[S_aperture] = sigma^2 * sum_{i,j in ap} rho(i-j),  rho = normalised
# autocorrelation of the effective per-frame kernel.
#
# Two corrections relative to the earlier version:
#   (a) the effective kernel is the PSF AFTER the high-pass filter, since the
#       map being searched is mf_coadd - gaussian(mf_coadd, smooth);
#   (b) the coadd averages independent per-frame noise, so the covariance is
#       mean_l |H_l|^2, not |mean_l H_l|^2 (Jensen: these are not equal).
# |FFT|^2 is shift invariant, so the centring convention of psf_mean.fits
# (fftshifted and y-flipped on write) is irrelevant here.
_rho = None
if os.path.exists(psf_path):
    try:
        _psf_cube = fits.getdata(psf_path).astype(float)
        if _psf_cube.ndim == 2:
            _psf_cube = _psf_cube[None, :, :]
        _psd = np.zeros(_psf_cube.shape[1:], dtype=float)
        for _h in _psf_cube:
            _he = _h - gaussian_filter(_h, sigma=a.smooth) if a.smooth > 0 else _h
            _F  = np.fft.fft2(_he)
            _psd += np.abs(_F) ** 2
        _psd /= len(_psf_cube)
        _ac  = np.real(np.fft.ifft2(_psd))          # lag 0 at index [0, 0]
        _rho = np.fft.fftshift(_ac) / _ac[0, 0]     # lag 0 at the array centre
        print(f"pixel covariance kernel from {psf_path}: "
              f"{len(_psf_cube)} frame(s), HPF sigma = {a.smooth:g} px")
    except Exception as _e:
        print(f"warning: could not build covariance kernel ({_e}); "
              f"assuming independent pixels (aperture SNR will be optimistic)")
else:
    print(f"note: {psf_path} not found; assuming independent pixels "
          f"(aperture SNR will be optimistic)")


def _lag_sum(shape_mask, rho):
    """sum_{i,j in mask} rho(i-j), via the mask autocorrelation on a lag grid.

    Direct correlation on a small stamp -- no FFT padding, so no parity or
    wraparound failure modes. Returns n_pix if rho is None.
    """
    n_pix = int(shape_mask.sum())
    if rho is None or n_pix == 0:
        return float(n_pix)
    m = shape_mask.astype(float)
    acm = correlate2d(m, m, mode='full')            # lag 0 at the centre
    ly, lx = acm.shape
    c_ry, c_rx = rho.shape[0] // 2, rho.shape[1] // 2
    hy, hx = ly // 2, lx // 2
    y0, y1 = c_ry - hy, c_ry - hy + ly
    x0, x1 = c_rx - hx, c_rx - hx + lx
    if y0 < 0 or x0 < 0 or y1 > rho.shape[0] or x1 > rho.shape[1]:
        # lag grid larger than the kernel: pad the kernel with zeros
        rho_w = np.zeros((ly, lx))
        sy0, sx0 = max(0, -y0), max(0, -x0)
        gy0, gx0 = max(0, y0), max(0, x0)
        gy1, gx1 = min(rho.shape[0], y1), min(rho.shape[1], x1)
        rho_w[sy0:sy0 + (gy1 - gy0), sx0:sx0 + (gx1 - gx0)] = rho[gy0:gy1, gx0:gx1]
    else:
        rho_w = rho[y0:y1, x0:x1]
    return float((acm * rho_w).sum())


# Aperture and background-annulus geometry are position independent, so the
# correlation sums are computed once on a stamp.
_CORR_CACHE = {}


def corr_factors(rad):
    """(sum_ij rho over the aperture, n_ap, sum_ij rho over the bg annulus,
    n_bg) for an aperture of radius `rad` px. Cached: geometry is position
    independent, so this is computed once per radius, not once per candidate."""
    key = round(float(rad), 6)
    if key not in _CORR_CACHE:
        L = int(np.ceil(a.bg_out * rad)) + 2
        sy, sx = np.indices((2 * L + 1, 2 * L + 1))
        sr = np.hypot(sy - L, sx - L)
        disc = sr <= rad
        bg   = (sr >= a.bg_in * rad) & (sr <= a.bg_out * rad)
        _CORR_CACHE[key] = (_lag_sum(disc, _rho), int(disc.sum()),
                            _lag_sum(bg, _rho), int(bg.sum()))
    return _CORR_CACHE[key]


CORR_AP, N_AP, CORR_BG, N_BG = corr_factors(aprad_px)

print(f"aperture: r = {aprad_px:.1f} px, {N_AP} pixels, "
      f"sum_ij rho = {CORR_AP:.1f}  (independent-pixel value would be {N_AP})")
if not a.no_bg:
    print(f"background annulus: {a.bg_in:g}-{a.bg_out:g} x aperture radius, "
          f"{N_BG} pixels")


# ------------------------------------------------------ radial profiles ----
def radial_profile(mask=None):
    """Per-annulus mu(r) and sigma(r); maps by interpolation.

    The estimator is robust to sources, so no mask is needed and none is used
    by default.  This matters: with mean/std the profile has to be protected by
    masking detected candidates, which makes sigma(r) depend on --cand -- a
    threshold documented as controlling only which peaks are *reported*.  That
    dependence was measured at 17% in the whitened value of a source at r=18 px
    between --cand 3.5 and 3.0, and the resulting sigma(18) varied by 16-28%
    between maps whose noise is identical by construction.

    Sources are positive-only and the noise is symmetric about zero, so the
    lower half of each annulus is uncontaminated by them.  Taking the median as
    mu and 1.4826 x MAD over the sub-median pixels as sigma therefore needs no
    knowledge of where the sources are.  Measured on five real maps: sigma(18)
    spread falls from 16-28% to 7.1% (consistent with the sampling error of an
    annulus holding ~100 independent resolution elements), and the result is
    now *identical* at --cand 3.0, 3.5 and 5.0.

    The estimator is robust to sources, so no mask is needed and none is used
    by default.  With mean/std the profile has to be protected by masking
    detected candidates, which makes sigma(r) depend on --cand -- a threshold
    documented as controlling only which peaks are *reported*.  That dependence
    was measured at 17% in the whitened value of a source at r=18 px, and the
    resulting sigma(18) varied by 16-28% between maps whose noise is identical
    by construction.

    Sources are positive-only and the noise is symmetric about zero, so the
    lower half of each annulus is uncontaminated by them.  Taking the median as
    mu and 1.4826 x MAD over the sub-median pixels as sigma therefore needs no
    knowledge of where the sources are, and the result is identical at --cand
    3.0, 3.5 and 5.0.

    This profile is purely radial.  A 2D (radius x azimuth) version was tried
    to absorb the anisotropy of the elongated satellite halo, but on a
    source-free control field the extra cells raised the maximum noise peak
    (4.47 -> 4.77 sigma) by more than the anisotropy correction was worth: the
    companion's margin over the source-free maximum was WORSE with 2D than 1D.
    The anisotropy is real but is reconstruction error, not photon noise, and
    is better attacked in the reconstruction than absorbed by a noisier
    estimator here.

    `mask` excludes specific pixels (used by --at to exclude a known source
    from its own profile); it is not used by the blind detection loop.
    """
    if mask is None:
        mask = np.zeros(img.shape, bool)
    radii, mus, sigs = [], [], []
    skipped = []
    for i in range(nb):
        v = img[(rbin == i) & ~mask]
        if v.size < MIN_ANN_PIX:
            skipped.append(i)
            continue
        med = np.median(v)
        lo  = v[v < med]
        sig = (1.4826 * np.median(np.abs(lo - med))
               if lo.size >= MIN_ANN_PIX // 2 else v.std(ddof=1))
        if not sig > 0:
            skipped.append(i)
            continue
        radii.append((i + 0.5) * a.binw)
        mus.append(med)
        sigs.append(sig)
    radii = np.array(radii); mus = np.array(mus); sigs = np.array(sigs)
    if radii.size < 2:
        sys.exit("error: fewer than two usable annuli; check --center/--binw")
    return (np.interp(r, radii, mus), np.interp(r, radii, sigs),
            radii, mus, sigs, skipped)


def find_peaks(snr, level):
    """Local maxima above level, brightest first, min separation = FWHM."""
    cand = np.argwhere((snr > level) & search_mask)
    cand = sorted(map(tuple, cand), key=lambda ij: -snr[ij])
    peaks = []
    for (i, j) in cand:
        lo_i, hi_i = max(i - 1, 0), min(i + 2, ny)
        lo_j, hi_j = max(j - 1, 0), min(j + 2, nx)
        if snr[i, j] < snr[lo_i:hi_i, lo_j:hi_j].max():
            continue
        if any(np.hypot(i - pi, j - pj) < a.fwhm for pi, pj, _ in peaks):
            continue
        peaks.append((i, j, snr[i, j]))
    return peaks


# ----------------------------------------------------- aperture statistic ---
def other_source_mask(ci, cj, cand_list):
    """Mask of 2*FWHM disks around every candidate EXCEPT the one at (ci,cj).

    The candidate's own mask must not be applied: it has radius 2*FWHM, which
    swallows the whole background annulus whenever bg_out*aprad < 2*FWHM
    (true for the defaults at aprad <= 1). That silently zeroed bhat.
    """
    m = np.zeros(img.shape, bool)
    for (i, j, _) in cand_list:
        if i == ci and j == cj:
            continue
        m |= np.hypot(yy - i, xx - j) < 2 * a.fwhm
    return m


def aperture_snr(ci, cj, mu_map, sigma_map, src_mask, rad=None):
    """Background-subtracted aperture sum divided by its correlated noise.

    Signal  S = sum_ap (img - mu(r)) - n_ap * bhat
    with bhat the mean of (img - mu(r)) over a local annulus, excluding
    pixels flagged in src_mask. bhat is the detection-side analogue of a
    nuisance constant in a design matrix: it absorbs any local pedestal left
    by imperfect satellite subtraction, at the cost of a little variance.

    Noise  N^2 = sigma_local^2 * [ CORR_AP + n_ap^2 * CORR_BG / n_bg^2 ]
    The first term is the correlated aperture variance; the second is the
    variance injected by bhat. Covariance between the aperture and the
    annulus is neglected -- they are disjoint and the HPF makes rho at those
    lags small and slightly negative, so this is mildly conservative.
    """
    rad = aprad_px if rad is None else rad
    corr_ap, _, corr_bg, _ = corr_factors(rad)
    dy, dx = yy - ci, xx - cj
    d2 = dy * dy + dx * dx
    ap_mask = d2 <= rad ** 2
    n_ap = int(ap_mask.sum())
    if n_ap == 0:
        return 0.0, 0.0, 1.0, 0, 0.0

    resid = img - mu_map
    S = float(resid[ap_mask].sum())

    bhat, var_b_factor = 0.0, 0.0
    if not a.no_bg:
        bg_mask = ((d2 >= (a.bg_in * rad) ** 2) &
                   (d2 <= (a.bg_out * rad) ** 2) & ~src_mask)
        n_bg = int(bg_mask.sum())
        if n_bg >= MIN_ANN_PIX:
            bhat = float(resid[bg_mask].mean())
            S -= n_ap * bhat
            var_b_factor = (n_ap ** 2) * corr_bg / max(n_bg ** 2, 1)

    sigma_local = float(sigma_map[ap_mask].mean())
    noise = sigma_local * np.sqrt(max(corr_ap + var_b_factor, 1e-30))
    return (S / noise if noise > 0 else 0.0), S, noise, n_ap, bhat


# ------------------------------------------------------------------ FAP ----
def n_res_annular(r0):
    return max(2.0 * np.pi * r0 / a.fwhm, 1.0)


def n_res_blind():
    """Independent resolution elements over the whole searched area."""
    area = float(search_mask.sum())
    return max(area / (np.pi * (a.fwhm / 2.0) ** 2), 1.0)


def n_indep_in_annulus(r0):
    """Independent resolution elements in the sigma(r) annulus at r0.

    This sets the effective dof of the noise estimate, which is what the
    small-sample correction needs -- not the raw pixel count, since the
    matched filter correlates neighbouring pixels.
    """
    area = 2.0 * np.pi * r0 * a.binw
    return max(area / (np.pi * (a.fwhm / 2.0) ** 2), 2.0)


def snr_from_p(p1):
    """Equivalent one-sided Gaussian sigma for a single-trial p."""
    from scipy.special import erfcinv
    return float(np.sqrt(2.0) * erfcinv(min(max(2.0 * p1, 1e-300), 2.0)))


def analytic_fap(snr_val, r0):
    """Single-trial p and FAP, with the small-sample penalty when --stat t.

    Gaussian erfc assumes sigma(r) is known exactly. It is estimated from a
    finite number of independent resolution elements in the annulus, so at
    small separations the tail is heavier than Gaussian. Following Mawet
    et al. (2014), use Student-t with n-1 dof and inflate the statistic by
    sqrt(1 + 1/n).
    """
    n_ind = n_indep_in_annulus(r0)
    if a.stat == 't':
        t_stat = snr_val / np.sqrt(1.0 + 1.0 / n_ind)
        p1 = float(student_t.sf(t_stat, df=max(n_ind - 1.0, 1.0)))
    else:
        t_stat = snr_val
        p1 = 0.5 * float(erfc(snr_val / np.sqrt(2.0)))

    n_res = n_res_blind() if a.fap_mode == 'blind' else n_res_annular(r0)
    fap = n_res * p1 if p1 < 1e-10 else 1.0 - (1.0 - p1) ** n_res
    return fap, n_res, p1, n_ind, t_stat


# ------------------------------------------------------------ null mode ----
if a.null:
    print("NULL mode: calibration only valid on source-free (CSO_scale=0) maps.")
    print("           On maps with sources this just reports the brightest feature.")
    mu_map, sigma_map, radii, mus, sigs, skipped = radial_profile(
        np.zeros(img.shape, bool))
    snr = np.where(search_mask, (img - mu_map) / sigma_map, -np.inf)
    i, j = np.unravel_index(np.argmax(snr), snr.shape)
    zero_mask = np.zeros(img.shape, bool)
    best = max(((ii, jj) for (ii, jj, _) in find_peaks(snr, a.cand)),
               key=lambda ij: aperture_snr(ij[0], ij[1], mu_map, sigma_map,
                                           zero_mask)[0],
               default=(i, j))
    ap_s, S, N, n, b = aperture_snr(best[0], best[1], mu_map, sigma_map, zero_mask)
    fap, n_res, p1, n_ind, _ = analytic_fap(ap_s, r[best])
    print(f"max peak SNR     = {snr[i, j]:.3f} at (y,x)=({i},{j}), r={r[i, j]:.1f}")
    print(f"max aperture SNR = {ap_s:.3f} at (y,x)=({best[0]},{best[1]}), "
          f"r={r[best]:.1f}  [signal={S:.4e}, noise={N:.4e}, npix={n}]")
    print(f"implied FAP      = {fap:.3e}  (N_res={n_res:.1f}, {a.fap_mode})")
    sys.exit(0)


# ------------------------------------------------------------- detection ----
# Single pass.  The iterative mask-and-recompute scheme this replaces existed
# only because the mean/std profile had to be protected from the sources; with
# a source-robust estimator the profile is fixed, so iterating would reproduce
# pass 1 exactly.  Removing the loop also removes the last route by which
# --cand could influence sigma(r).
mu_map, sigma_map, radii, mus, sigs, skipped = radial_profile()
snr = np.where(search_mask, (img - mu_map) / sigma_map, -np.inf)
candidates = find_peaks(snr, a.cand)
print(f"{len(candidates)} candidate(s) above {a.cand:g} sigma (peak); "
      f"sigma(r) is source-robust and independent of --cand")

if skipped:
    print(f"warning: {len(skipped)} annulus/annuli had < {MIN_ANN_PIX} unmasked "
          f"pixels and were interpolated across (bins {skipped[:8]}"
          f"{'...' if len(skipped) > 8 else ''})")

# ------------------------------------------ aperture SNR and FAP per candidate
# The detection decision is made on the APERTURE SNR. Peak SNR only governs
# candidacy (--cand), because a source can fall below --thresh on the peak
# statistic while clearing it on the aperture statistic.
rows = []
for (i, j, speak) in candidates:
    _om = other_source_mask(i, j, candidates)
    ap_s, S, N, n, bhat = aperture_snr(i, j, mu_map, sigma_map, _om)
    fap, n_res, p1, n_ind, t_stat = analytic_fap(ap_s, r[i, j])
    rows.append(dict(i=i, j=j, r=r[i, j], speak=speak, sap=ap_s, S=S, N=N,
                     n=n, bhat=bhat, fap=fap, n_res=n_res, p1=p1, n_ind=n_ind))

rows.sort(key=lambda d: -d['sap'])
detections = [d for d in rows if d['sap'] >= a.thresh]
print(f"final: {len(detections)} detection(s) with aperture SNR >= {a.thresh:g}")

# ------------------------------------------------------ known-position mode --
if a.at is not None:
    ai, aj = a.at
    if not (0 <= ai < ny and 0 <= aj < nx):
        sys.exit(f"error: --at ({ai},{aj}) is outside the {ny}x{nx} map")
    # Snap to the local maximum within one FWHM: a specified position is
    # normally the injection/catalogue position, and the response peak can sit
    # a pixel or two away.  Report both if they differ.
    d2at = (yy - ai) ** 2 + (xx - aj) ** 2
    # Recompute the profile with a 2*FWHM disc at the position excluded.  The
    # lower-half sigma estimator is immune to the source, but the median mu is
    # not: a source occupying ~10% of its cell drags the cell median up and
    # subtracts itself (measured: mu inflated by 0.84 sigma at the injected
    # companion, cell median +3.96e-3 with the source in vs +1.89e-3 without).
    # A blind search cannot do this -- it doesn't know where the sources are --
    # but a known position can and must.
    _at_excl = d2at <= (2.0 * a.fwhm) ** 2
    mu_at, sig_at, *_ = radial_profile(_at_excl)
    snr_at = np.where(search_mask, (img - mu_at) / sig_at, -np.inf)
    _near = np.where(d2at <= a.fwhm ** 2, snr_at, -np.inf)
    pi, pj = np.unravel_index(np.argmax(_near), _near.shape)
    _om = other_source_mask(pi, pj, candidates)
    ap_s, S, N, n, bhat = aperture_snr(pi, pj, mu_at, sig_at, _om)
    _, _, p1, n_ind, t_stat = analytic_fap(ap_s, r[pi, pj])
    print(f"\n--at ({ai},{aj}): known-position evaluation "
          f"(single-trial statistics; no blind-search penalty)")
    if (pi, pj) != (ai, aj):
        print(f"  local max within 1 FWHM is at ({pi},{pj}); statistics "
              f"reported there. At ({ai},{aj}) itself the peak SNR is "
          f"{snr_at[ai, aj]:.2f}.")
    print(f"  Peak SNR        : {snr_at[pi, pj]:.2f}   "
          f"(self-excluded profile; blind map reads {snr[pi, pj]:.2f})")
    print(f"  Aperture SNR    : {ap_s:.2f}   "
          f"(signal {S:.4e}, noise {N:.4e}, {n} pix)")
    print(f"  Local background: {bhat:.4e} e-/pix")
    print(f"  sigma dof       : {n_ind:.1f} independent elements in the annulus")
    print(f"  Single-trial p  : {p1:.3e}   "
          f"(equivalent Gaussian {snr_from_p(p1):.2f} sigma)")
    print(f"  NOTE: this p assumes the position was fixed BEFORE looking at "
          f"the map. If the position came from searching this map, the blind "
          f"FAP in the table above is the honest number.")

# --------------------------------------------------------- aperture scan ----
# The optimal linear statistic for correlated noise is t^T C^-1 x / sqrt(t^T C^-1 t).
# Here the source response and the noise covariance share the same kernel
# (both are mean_l h_l (x) h_l), so t is proportional to rho and C^-1 t is
# uniform -- i.e. a flat top-hat is the right FORM, provided the aperture
# encloses the whole response. It does not follow that any particular radius
# is safe: too small truncates the response, too large reaches into the
# negative bowl the HPF digs around every source. Scan and pick.
if a.apscan and rows:
    print("\naperture SNR vs radius (columns: aprad in FWHM):")
    grid = [0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0]
    print(f"{'y':>6} {'x':>6} {'peak':>7} " +
          " ".join(f"{g:>7.2f}" for g in grid))
    for d in rows[:8]:
        _om = other_source_mask(d['i'], d['j'], candidates)
        vals = [aperture_snr(d['i'], d['j'], mu_map, sigma_map, _om,
                             rad=g * a.fwhm)[0] for g in grid]
        print(f"{d['i']:6d} {d['j']:6d} {d['speak']:7.2f} " +
              " ".join(f"{v:7.2f}" for v in vals))
    print("  (pick the radius that maximises SNR on an INJECTED source, not on "
          "a candidate)")

# ------------------------------------------------------------- save maps ----
fits.writeto(os.path.join(outdir, 'mf_coadd_hpf_wsnr.fits'),
             np.where(search_mask, (img - mu_map) / sigma_map, 0
                      ).astype(np.float32), overwrite=True)
fits.writeto(os.path.join(outdir, 'mf_coadd_hpf_sigma.fits'),
             sigma_map.astype(np.float32), overwrite=True)
fits.writeto(os.path.join(outdir, 'mf_coadd_hpf_mu.fits'),
             mu_map.astype(np.float32), overwrite=True)


def _fap_str(f):
    return '< 1e-300' if f <= 0 else f'{f:.3e}'


lines = [
    f"# aperture radius   = {a.aprad:.2f} FWHM = {aprad_px:.1f} px "
    f"({N_AP} pix, sum_ij rho = {CORR_AP:.1f})",
    f"# HPF sigma         = {a.smooth:g} px (must match mf_smooth_sigma)",
    f"# background        = {'off' if a.no_bg else f'{a.bg_in:g}-{a.bg_out:g} x aperture radius'}",
    f"# FAP               = {a.fap_mode}, statistic = {a.stat}",
    f"# search annulus    = {a.fwhm:.1f} to {rmax:.1f} px",
    f"#{'y':>6} {'x':>6} {'r_pix':>8} {'SNRpeak':>8} {'SNRaper':>8} "
    f"{'N_res':>8} {'FAP':>12} {'bkg':>12} {'sigma_r':>12}",
]
for d in detections:
    lines.append(
        f"{d['i']:7d} {d['j']:6d} {d['r']:8.1f} {d['speak']:8.2f} "
        f"{d['sap']:8.2f} {d['n_res']:8.1f} {_fap_str(d['fap']):>12} "
        f"{d['bhat']:12.4e} {sigma_map[d['i'], d['j']]:12.4e}")
report = "\n".join(lines)
with open(os.path.join(outdir, 'detections.txt'), 'w') as f:
    f.write(report + "\n")

if detections:
    print(report)
    print()
    for d in detections:
        alt = (n_res_annular(d['r']) if a.fap_mode == 'blind'
               else n_res_blind())
        alt_name = 'annular' if a.fap_mode == 'blind' else 'blind'
        print(f"  Detection at (y,x) = ({d['i']}, {d['j']}),  r = {d['r']:.1f} px")
        print(f"    Peak SNR        : {d['speak']:.2f}")
        print(f"    Aperture SNR    : {d['sap']:.2f}   "
              f"(signal {d['S']:.4e}, noise {d['N']:.4e}, {d['n']} pix)")
        print(f"    Local background: {d['bhat']:.4e} e-/pix")
        print(f"    N_res ({a.fap_mode:<7s}): {d['n_res']:.1f}   "
              f"[{alt_name} would give {alt:.1f}]")
        print(f"    sigma(r) dof    : {d['n_ind']:.1f} independent elements "
              f"in the annulus")
        print(f"    Single-trial p  : {d['p1']:.3e}")
        print(f"    FAP             : {_fap_str(d['fap'])}")
        print()
else:
    print("no detections")
    if rows:
        print(f"\ntop candidate(s) above {a.cand:g} sigma (peak):")
        print(f"#{'y':>6} {'x':>6} {'r_pix':>8} {'SNRpeak':>8} {'SNRaper':>8} {'FAP':>12}")
        for d in rows[:5]:
            print(f"{d['i']:7d} {d['j']:6d} {d['r']:8.1f} {d['speak']:8.2f} "
                  f"{d['sap']:8.2f} {_fap_str(d['fap']):>12}")

print(f"\nfinal pass noise floor: sigma(r) min/max = "
      f"{sigs.min():.3e} / {sigs.max():.3e}")
print(f"saved detections.txt, mf_coadd_hpf_wsnr.fits, mf_coadd_hpf_sigma.fits, "
      f"mf_coadd_hpf_mu.fits in {outdir}")

if a.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.semilogy(radii, sigs, 'b.-', label='sigma(r), sources masked')
    for d in detections:
        ax1.axvline(d['r'], color='g', ls=':', alpha=0.7,
                    label=f"r={d['r']:.1f} ap={d['sap']:.2f} "
                          f"FAP={_fap_str(d['fap'])}")
    ax1.set_ylabel('noise std [e-]'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title(os.path.basename(hpf_path))
    ax2.plot(radii, mus, 'r.-', label='mu(r), subtracted before whitening')
    ax2.axhline(0, color='grey', lw=0.8, ls=':')
    ax2.set_xlabel('radius from center [pix]'); ax2.set_ylabel('mean [e-]')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'sigma_vs_radius.png'), dpi=120)
    print("saved sigma_vs_radius.png")
