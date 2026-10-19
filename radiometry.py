# =============================================================================
# radiometry.py  --  magnitude calibration + radiometric chain for the PD testbed.
#
# Turns "I want a mag +10 target in Johnson V" into (per-slice spectral weights,
# total detected counts) for the broadband generator, with the correct COLOR
# across 400-1000 nm. Two concerns are kept strictly separate:
#
#   (1) the OBJECT spectrum  S_obj(lam)  = source(lam) x reflectance(lam)
#         - source = solar (blackbody placeholder, T_sun) for SDA
#         - reflectance = 1 for a GRAY object (Stage-1 approximation); later a
#           per-pixel/per-material reflectance multiplies in (Stage 2). The
#           calibration only ever sees the DISK-INTEGRATED S_obj, so the gray->color
#           transition is a spectrum swap, not a redesign.
#
#   (2) the RADIOMETRIC CHAIN  G(lam) = T_atm(lam) x T_opt(lam) x QE(lam)
#         applied to predict DETECTED counts. NOT part of the magnitude definition.
#
# MAGNITUDE is an intrinsic source property, defined through the FILTER RESPONSE
# only (above the atmosphere), photon-counted:
#     m_X = -2.5 log10( Integral S_obj R_X (lam/hc) dlam / ZP_X )
# The chain then predicts detected counts; thus a given magnitude yields both the
# correct intrinsic brightness AND the correct counts after throughput losses.
#
# *** EMPIRICAL CURVES ARE PLACEHOLDERS unless you drop in real data. ***
# Filter responses (Gaussian approx), Zyla 5.5 QE (datasheet-shaped approx),
# atmospheric transmission (Rayleigh+gray-aerosol flavored), and optics throughput
# (smooth) are ALL approximate and clearly flagged. Vega zero-points are standard
# literature values (Bessell 1998) to ~3 s.f. -- VERIFY before quoting absolute mags.
# Sloan/AB zero-points are EXACT (3631 Jy definition). Everything is overridable.
# =============================================================================

import numpy as np

# ---- physical constants (cgs; wavelengths handled in cm where physics needs it) ----
_H   = 6.62607015e-27      # erg s
_C   = 2.99792458e10       # cm / s
_HC  = _H * _C             # erg cm
_JY  = 1.0e-23             # erg s^-1 cm^-2 Hz^-1  per Jansky
_AB_FNU = 3631.0 * _JY     # AB zero-point flux density (EXACT by definition)

def _nm_cm(lam_nm):  return np.asarray(lam_nm, float) * 1e-7      # nm -> cm
def _photon_per_erg(lam_nm):  return _nm_cm(lam_nm) / _HC          # (lam/hc), photons per erg
_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 2.x renamed trapz

# =============================================================================
# Source spectrum
# =============================================================================
def blackbody_flambda(lam_nm, T_kelvin):
    """Planck F_lambda(lam,T) in arbitrary (consistent) energy units; SHAPE is what
    matters (absolute scale is set later by the magnitude). Solar placeholder: T~6000K."""
    lam = _nm_cm(lam_nm)
    x = _HC / (lam * 1.380649e-16 * T_kelvin)        # hc/(lam kB T)
    return lam ** -5 / np.expm1(x)

# =============================================================================
# Filter bandpasses  (Gaussian APPROXIMATIONS; replace with real response tables)
# lam_eff [nm], FWHM [nm].  Johnson-Cousins (Vega system) + Sloan (AB system).
# =============================================================================
JOHNSON = {"B": (442.0, 94.0), "V": (540.0, 88.0), "R": (647.0, 147.0), "I": (786.0, 106.0)}
SLOAN   = {"u": (355.0, 57.0), "g": (468.0, 141.0), "r": (616.0, 137.0),
           "i": (748.0, 153.0), "z": (893.0, 137.0)}

# Vega-system zero-point flux densities f_lambda(mag 0), erg s^-1 cm^-2 A^-1.
# STANDARD LITERATURE VALUES (Bessell 1998, Cousins R/I) -- *** VERIFY ***.
ZP_VEGA_FLAMBDA = {"B": 6.32e-9, "V": 3.63e-9, "R": 2.18e-9, "I": 1.13e-9}  # per Angstrom

def filter_response(band, lam_nm, system="johnson", table=None):
    """Filter response R(lam) in [0,1]. Gaussian approx unless `table`=(lam_nm,R) given."""
    if table is not None:
        tl, tr = table
        return np.interp(lam_nm, tl, tr, left=0.0, right=0.0)
    cat = JOHNSON if system == "johnson" else SLOAN
    if band not in cat:
        raise ValueError(f"band {band!r} not in {system} set {list(cat)}")
    lam0, fwhm = cat[band]
    sig = fwhm / 2.3548200450309493
    return np.exp(-0.5 * ((np.asarray(lam_nm, float) - lam0) / sig) ** 2)

# =============================================================================
# Radiometric chain terms (ALL PLACEHOLDERS; drop in measured curves)
# =============================================================================
def qe_zyla55(lam_nm, table=None):
    """APPROX QE of an Andor Zyla 5.5 sCMOS (datasheet-shaped). Replace with measured QE."""
    if table is not None:
        tl, tq = table
        return np.clip(np.interp(lam_nm, tl, tq, left=0.0, right=0.0), 0, 1)
    anchor_l = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000.])
    anchor_q = np.array([0.30, 0.48, 0.57, 0.60, 0.57, 0.52, 0.45, 0.38, 0.30, 0.22, 0.15, 0.10, 0.06])
    return np.clip(np.interp(lam_nm, anchor_l, anchor_q, left=0.0, right=0.0), 0, 1)

def atm_transmission(lam_nm, airmass=1.0, tau_aerosol=0.05, table=None):
    """APPROX atmospheric transmission: Rayleigh (lam^-4) + gray aerosol, exp(-X*tau).
    Flavored placeholder (no molecular absorption bands). Replace with a real model."""
    if table is not None:
        tl, tt = table
        return np.clip(np.interp(lam_nm, tl, tt, left=0.0, right=0.0), 0, 1)
    lam_um = np.asarray(lam_nm, float) / 1000.0
    tau_ray = 0.0094977 * lam_um ** (-4.05)                 # sea-level Rayleigh optical depth
    return np.exp(-airmass * (tau_ray + tau_aerosol))

def optics_throughput(lam_nm, table=None):
    """APPROX product of all optics passbands (mirrors, lenses, diversity BS, coatings).
    Smooth placeholder ~0.5-0.7. Replace with the instrument's measured product."""
    if table is not None:
        tl, tt = table
        return np.clip(np.interp(lam_nm, tl, tt, left=0.0, right=0.0), 0, 1)
    anchor_l = np.array([400, 500, 600, 700, 800, 900, 1000.])
    anchor_t = np.array([0.50, 0.62, 0.70, 0.68, 0.66, 0.64, 0.60])
    return np.clip(np.interp(lam_nm, anchor_l, anchor_t, left=0.0, right=0.0), 0, 1)

# =============================================================================
# Magnitude calibration
# =============================================================================
def _band_photon_integral(flam, R, lam_nm):
    """Photon-counted in-band flux: Integral f_lambda R (lam/hc) dlam.  (dlam in Angstrom)"""
    dl_A = np.gradient(np.asarray(lam_nm, float)) * 10.0     # nm -> Angstrom widths
    return np.sum(flam * R * _photon_per_erg(lam_nm) * dl_A)

def _zeropoint_photons(band, lam_nm, system, R):
    """Photon count rate [phot s^-1 cm^-2] of a mag-0 source in this band."""
    if system == "johnson":
        flam0 = ZP_VEGA_FLAMBDA[band]                         # ~flat across the narrow band
        return _band_photon_integral(np.full_like(np.asarray(lam_nm, float), flam0), R, lam_nm)
    # AB (Sloan): F_nu = 3631 Jy -> F_lambda = F_nu c / lam^2 ; EXACT
    lam_cm = _nm_cm(lam_nm)
    flam_ab = _AB_FNU * _C / lam_cm ** 2 * 1e-8               # erg s^-1 cm^-2 A^-1 (c/lam^2, /1e8: cm->A)
    return _band_photon_integral(flam_ab, R, lam_nm)

def calibrate(mag, band="V", system="johnson", *, lam_nm,
              T_sun=6000.0, reflectance=None,
              D_tel_m=3.5, obstruction_frac=0.0, t_exp_s=5.0e-3, airmass=1.0,
              filt_table=None, qe_table=None, atm_table=None, opt_table=None,
              verbose=True):
    """Calibrate a broadband observation from a magnitude.

    Returns (weights, flux_total, info):
      weights    : per-slice spectral weights on lam_nm (normalized, sum=1) -> generator `spectrum`
      flux_total : total DETECTED counts [e-] over the band                 -> generator `flux_total`
      info       : dict of diagnostics (per-band counts, chain values, etc.)

    The magnitude fixes the intrinsic in-band photon flux (filter only); the chain
    (atmosphere x optics x QE) x area x exposure then predicts detected counts.
    `reflectance`: None=gray (1); else array on lam_nm = disk-integrated object reflectance.
    """
    lam_nm = np.asarray(lam_nm, float)
    refl = np.ones_like(lam_nm) if reflectance is None else np.asarray(reflectance, float)
    shape = blackbody_flambda(lam_nm, T_sun) * refl          # S_obj shape (energy F_lambda, arb scale)

    R = filter_response(band, lam_nm, system=system, table=filt_table)
    if R.max() <= 0:
        raise ValueError(f"band {band!r} has no response on the given grid")
    # fraction of the band's response that actually lies on the supplied grid
    if filt_table is None:
        lam_full = np.arange(300.0, 1101.0, 1.0)
        R_full = filter_response(band, lam_full, system=system)
        on = (lam_full >= lam_nm[0]) & (lam_full <= lam_nm[-1])
        frac_on_grid = float(_trapz(R_full[on], lam_full[on]) / _trapz(R_full, lam_full))
    else:
        frac_on_grid = 1.0
    # in-band intrinsic photon flux of the (unscaled) source shape:
    src_int = _band_photon_integral(shape, R, lam_nm)
    ZP = _zeropoint_photons(band, lam_nm, system, R)
    # scale alpha so that -2.5 log10(alpha*src_int / ZP) = mag
    alpha = (ZP * 10 ** (-0.4 * mag)) / src_int

    # detected chain
    G = (atm_transmission(lam_nm, airmass, table=atm_table)
         * optics_throughput(lam_nm, table=opt_table)
         * qe_zyla55(lam_nm, table=qe_table))
    area_cm2 = np.pi / 4.0 * (D_tel_m * 100.0) ** 2 * (1.0 - obstruction_frac ** 2)
    dl_A = np.gradient(lam_nm) * 10.0
    counts_per_slice = (alpha * shape * G * _photon_per_erg(lam_nm)
                        * area_cm2 * t_exp_s * dl_A)          # e- per slice per frame
    flux_total = float(counts_per_slice.sum())
    weights = counts_per_slice / counts_per_slice.sum()

    info = dict(alpha=alpha, ZP_photons=ZP, area_cm2=area_cm2, t_exp_s=t_exp_s,
                airmass=airmass, band=band, system=system, mag=mag,
                frac_band_on_grid=frac_on_grid, chain_mean=float(G.mean()))
    if verbose:
        print(f"  radiometry: mag {mag:+.1f} {system}-{band}  ->  flux_total = {flux_total:.3e} e-/frame")
        print(f"    alpha={alpha:.3e}  ZP={ZP:.3e} phot/s/cm2  area={area_cm2:.3e} cm2  texp={t_exp_s*1e3:.1f} ms")
        print(f"    chain<G>={G.mean():.3f}  (atm x optics x QE; ALL PLACEHOLDERS)")
        if info["frac_band_on_grid"] < 0.95:
            print(f"    WARNING: only {info['frac_band_on_grid']*100:.0f}% of the {band} band lies on "
                  f"[{lam_nm[0]:.0f},{lam_nm[-1]:.0f}] nm -> magnitude is biased; widen the grid.")
    return weights, flux_total, info


def summary_plot(lam_nm, path="radiometry_chain.png", system="johnson", T_sun=6000.0):
    """Save a plot of the chain terms + a band, for visual verification."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    lam_nm = np.asarray(lam_nm, float)
    fig, ax = plt.subplots(figsize=(8, 5))
    sh = blackbody_flambda(lam_nm, T_sun); ax.plot(lam_nm, sh / sh.max(), label=f"solar BB {T_sun:.0f}K (shape)")
    ax.plot(lam_nm, qe_zyla55(lam_nm), label="Zyla 5.5 QE (approx)")
    ax.plot(lam_nm, atm_transmission(lam_nm), label="atm transmission (approx)")
    ax.plot(lam_nm, optics_throughput(lam_nm), label="optics throughput (approx)")
    cat = JOHNSON if system == "johnson" else SLOAN
    for b in cat:
        r = filter_response(b, lam_nm, system=system)
        if r.max() > 0.05: ax.fill_between(lam_nm, 0, r, alpha=0.15)
    ax.set_xlabel("wavelength [nm]"); ax.set_ylabel("normalized response")
    ax.set_title("Radiometric chain + bands (PLACEHOLDER curves)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


if __name__ == "__main__":
    lam = np.linspace(400, 1000, 15)
    print("self-test on a 15-slice 400-1000 nm grid:")
    w10, f10, _ = calibrate(10.0, "V", "johnson", lam_nm=lam)
    w15, f15, _ = calibrate(15.0, "V", "johnson", lam_nm=lam)
    print(f"  5-mag check: flux(+10)/flux(+15) = {f10/f15:.1f}  (expect 100.0)")
    print(f"  weights sum = {w10.sum():.6f}  (expect 1.0)")
    calibrate(10.0, "r", "sloan", lam_nm=lam)
    summary_plot(lam)
    print("  wrote radiometry_chain.png")
