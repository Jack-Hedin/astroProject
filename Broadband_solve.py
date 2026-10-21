#!/usr/bin/env python
import numpy as np
import numpy.ma as ma
import math
import scipy
import matplotlib
import numba
import aotools
import matplotlib.pyplot as plt
import skimage
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
from scipy.fftpack import fft2, ifft2
from scipy import signal
from scipy import optimize
from astropy.io import fits
from scipy import ndimage
import time
import sys
import os
import argparse

#Args
ap = argparse.ArgumentParser()
ap.add_argument('--niter', type=int, default=50) #108
ap.add_argument('--whiten', type=int ) #324
ap.add_argument('--error', default='ls') #339
ap.add_argument('--restart', type=int, default=0) #110
ap.add_argument('--filter', type=int, default=0) #139
ap.add_argument('--companion', type=int, default=0) #210
ap.add_argument('--positivity', type=int, default=2) #272
ap.add_argument('--estimate_path', default='default') #380
ap.add_argument('--mf_path', default='') #140
ap.add_argument('--output', default='default')
ap.add_argument('--support', type=int, default=0)
a = ap.parse_args()

# --- Run log: tee all stdout to a timestamped log file ---
class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, logfile):
        self._stdout  = sys.stdout
        self._logfile = logfile
    def write(self, msg):
        self._stdout.write(msg)
        self._logfile.write(msg)
    def flush(self):
        self._stdout.flush()
        self._logfile.flush()
    def fileno(self):
        return self._stdout.fileno()

_log_timestamp = time.strftime('%Y%m%d_%H%M%S')

if a.output == 'default':
    _run_dir = _log_timestamp
else:
    _run_dir = a.output
os.makedirs(_run_dir, exist_ok=True)
_log_path = os.path.join(_run_dir, 'run.log')
_log_fh   = open(_log_path, 'w')
_orig_stdout = sys.stdout          # save original before wrapping
sys.stdout = _Tee(_log_fh)
print(f"Run log: {_log_path}")
_plot_files = []   # accumulates saved plot filenames for the HTML report
_plot_counter = [0]

def _save_plot(label='plot'):
    """Save current figure to a timestamped PNG and record it."""
    _plot_counter[0] += 1
    _path = os.path.join(_run_dir, f'plot{_plot_counter[0]:02d}_{label}.png')
    plt.savefig(_path, dpi=80, bbox_inches='tight')
    plt.close()
    _plot_files.append((_path, label))

"""
Physical parameters
T1 -- reference temperature for the Stage-2 reflectance deconvolution.
  The object's disk-integrated spectrum is divided by B_lambda(T1) below to
  form a reflectance, and radiometry.calibrate() multiplies it back by
  B_lambda(T_sun).  The two MUST match or a spurious colour tilt is imposed on
  every simulated object; the calibrate() call passes T_sun=T1 to guarantee it.
  *** THIS IS NOT A PHYSICAL SOLAR TEMPERATURE AND MUST NOT BE "CORRECTED". ***
  Because the two blackbodies cancel, the VALUE IS ARBITRARY in Stage-2 usage
  (verified: 4000/5778/6000/7500 K all give identical weights to 1 ULP).  It
  only has physical meaning for Stage-1 GRAY objects (reflectance=None), which
  this script never uses.  Kept at 6000 to match radiometry's historical
  default so results stay comparable with all earlier runs.
  Setting T1 = 5778 without matching T_sun tilts per-slice weights by up to
  8.8% across 400-914 nm -- a smooth slope that reads as a plausible
  reflectance rather than as a bug.
"""
T1 = 6000  # reference temperature [K] for the reflectance deconvolution; see above (5778 K for the SUn)
P = 600    # air pressure in mm Hg
T = 7      # air temperature in degrees C  (NOTE: unrelated to T1; Celsius, not Kelvin)
fwv = 8    # water vapor pressure in mm Hg
zen = 0    # Zenith angle in degrees
zenith = (zen/360.0)*2*np.pi

# Telescope parameters
D = 360.0  # Diameter of telescope aperture [cm]
print('Diameter of telescope =', D/100.0, 'meters')
D1 = 2.0*D  # Phase screen diameter

# Wavelength parameters
lamb1 = 400.0  # min wavelength [nm]
lamb2 = 920.0  # max wavelength [nm]
pixscale = 1.22 * 206265.0 * (lamb1 * 1e-9) / (D * 1e-2) / 2.0
print(f'pixel scale = {pixscale:.5f} arcsec  '
      f'(Rayleigh/2 at lambda={lamb1}nm, D={D/100:.1f}m)')

# Simulation parameters
incr = 8
r0_500 = 17
nim = 100
nderiv = 0
display = 0
niter = a.niter
disperse = 1
restart = a.restart
fitsout = 1

# --- Support mask (optional, restart only) ------------------------------------
#'/Users/jackr/OneDrive/Documents/Desktop/Hedin/support.fits'
support_mask_file = ''
if a.support: support_mask_file = '/Users/jackhedin/Projects/astroProject/template.fits'
    # e.g. '/Users/.../satellite_mask.fits'
multiwavelength_object_filename = "multi_wavelength_object.fits" #"multi_wavelength_object.fits"  # Path to pre-generated object
"""
 support_mask_file: path to a FITS file containing a 2D binary (0/1) mask in
   display space (same convention as satellite_mask.fits). When set, pixels
   outside the mask are forced to zero via L-BFGS-B bounds. Works with both
   restart=0 (cold start) and restart=1 (warm start).
 Set to '' to disable (default behaviour, no support constraint).
 
 --- Matched-filter point-source detection -----------------------------------
 do_matched_filter = 1: after reconstruction, mask the satellite, subtract the
   masked model from the data, convolve each residual frame with the
   wavelength-averaged PSF (unit-volume normalised), and co-add frames.

 Workflow for running the matched filter without re-running the recovery:
   1. First run:  restart=0, do_matched_filter=0, niter=N  -> saves estimate.fits
   2. MF run:     restart=1, do_matched_filter=1, niter=0
      -> regenerates data+PSFs (same random_seed), loads estimate.fits,
         skips the optimizer entirely, runs the matched filter.

 mf_estimate_file : estimate FITS to mask ('' = use estimate.fits from this run)
   The support_mask_file is used directly as the model mask -- no threshold
   prompt. If support_mask_file = '' all nonzero estimate pixels are used.
"""     
#'C:/Users/jackr/OneDrive/Documents/Desktop/Hedin/20261005_021013/estimate.fits'
do_matched_filter = a.filter
mf_estimate_file  =  a.mf_path   # '' = use estimate.fits from this run
mf_smooth_sigma   = 3     # Gaussian smoothing sigma [pixels] for mf_coadd_smooth.fits
                           # Set to 0 to skip smoothing.
mf_use_truth      = 0     # 1 = use object.fits (truth without delta function)
                           #     as the model; support_mask_file zeros the delta
                           #     function (which lies outside the support).
"""
 --- Companion spectral extraction -------------------------------------------
 Requires: restart=1, niter=0, support_mask_file != ''
 After the reconstruction, the converged satellite model (with outside-support
 pixels zeroed) is subtracted from the data. The residuals are then used to
 extract the companion spectrum via joint linear regression across all wavelength
 slices. Position is given in DS9 display coordinates (1-based, y-flipped).
 1 = run spectral extraction
 DS9 x coordinate of companion (1-based, from MF detection)
 DS9 y coordinate of companion (1-based, from MF detection)
 Satellite exclusion threshold for the spectral
 aperture, as a fraction of the peak of the
 satellite MODEL footprint.  0 = exclude only the
 object support (previous behaviour).

 SUPPORT_MASK says where the satellite's FLUX is,
 which is the right constraint in object space.
 What contaminates b, though, is the residual left
 by subtracting the satellite MODEL, and that model
 is the support convolved with the PSF -- so its
 errors are smeared over the whole PSF footprint,
 tens of pixels wide, not confined to the support.
 Excluding only the support therefore removes the
 region where the satellite IS while leaving the
 region where its subtraction errors ARE.  This
 shows up as a large, highly significant nuisance
 pedestal (+1.26e3 e-/frame at 9.2 sigma at
 m_v=11.88) and is the leading candidate for the
 Rbar ~ 1.3 amplitude bias.

 The footprint is computed directly rather than
 approximated by a dilation radius: the support is
 convolved with the frame-averaged PSF and
 thresholded, so the excluded region carries the
 true anisotropy of the PSF and the parameter is
 the fraction of peak model amplitude below which
 contamination is accepted.  Typical trial values
 1e-3 to 1e-1; larger excludes more.

 At small separations this cannot be made free:
 removing the primary's halo also removes aperture
 the companion templates need.  Sweep it and watch
 Rbar, the pedestal significance and the per-slice
 SNRs against the printed aperture pixel count.
 aperture radius in pixels for regression sums.
 NOTE: this is NOT ~2x PSF FWHM -- at this pixel
 scale the PSF core is a few px, so 160 px is a
 ~80,000 pixel aperture. Inverse-variance
 weighting means the far pixels contribute almost
 nothing to b, so it is mainly slow; the run
 prints an aperture diagnostic showing the radius
 that encloses 99% of sum(T^2). Shrink to a few
 times that once cs_nuisance is on.
 nuisance terms in the design matrix:
   'none'   = physical slices only (old behaviour)
   'const'  = + local pedestal  [recommended]
   'linear' = + pedestal and x/y gradients
 Absorbs residual satellite-subtraction offset
 under the aperture, which otherwise biases every
 alpha_k in the same direction. Marginalised out
 of the reported covariance automatically.
 1 = use truth object.fits as satellite model (diagnostic only)
     bypasses reconstruction to test regression code in isolation
"""
companion_spectrum = a.companion       
companion_ds9_x   = 270      
companion_ds9_y   = 257      
companion_sat_excl_frac = 0.0
companion_ap_rad  = 160       
cs_nuisance       = 'const'  
cs_use_truth      = 0        

# --- Noise / radiometry parameters (Change 1) --------------------------------
"""
   add_noise = 0 reproduces the previous noise-free behavior EXACTLY.
   add_noise = 1 calibrates the object to `target_mag` with radiometry.calibrate()
   (the object's own disk-integrated spectrum is passed as the reflectance term,
   radiometry's Stage-2 design), converting `object` and hence `data` to
   DETECTED e-/frame, then applies Poisson (photon) + Gaussian read noise.
"""

add_noise = 1
target_mag = 1.0        # apparent magnitude of the target
mag_band = "V"           # radiometry band: Johnson B,V,R,I or Sloan u,g,r,i,z
mag_system = "johnson"   # "johnson" (Vega) or "sloan" (AB)
t_exp_s = 5.0e-3         # exposure time per frame [s]
read_noise = 2.0         # read noise [e- rms / pixel / frame]
gain = 1.0               # e- per count; data kept in electrons throughout
obstruction_frac = 0.0   # central obstruction (linear fraction of D)

# --- Positivity strategy ------------------------------------------------------
"""
 The optimiser works on `vars`; the object is o = phi(vars).  Positivity is
 always additionally enforced by an L-BFGS-B lower bound of 0 on vars.

   positivity = 1 : o = v            bounds-only.  phi = identity.
   positivity = 2 : o = v^2 + 2*c*v  with c = read_noise*sqrt(gain).   DEFAULT

 Equivalently, 2 is v = sqrt(o + c^2) - c: square-root parameterisation of the
 object on a pedestal c^2, with the pedestal subtracted so o = 0 stays
 attainable.  c must be positive and of the right order; its precise value is
 not critical (measured insensitivity over ~3 decades).  c = read*sqrt(gain) is
 a convention: dimensionally correct (units of sqrt(e-)), auto-scaling with the
 detector, and requiring no tuning.

 Both are diffeomorphic on [0, inf) (phi' >= 2c > 0 everywhere, INCLUDING at
 the boundary), so they share the same feasible set, the same KKT points, and
 the same unique constrained minimum.  They differ only in the path taken.
 Since runs stop at a budget rather than at convergence, that path difference
 is what you actually get: measured at 512^2 x 10 slices x 100 frames,
   mode 2: 50 iterations / 1145 s -> chi^2 2.4141e7, mean relative RMSE 63.0%
   mode 1: 170 iterations / 3400 s -> chi^2 2.5127e7, mean relative RMSE 80.4%
 and mode 2 is better at every wavelength.  Mode 2 is the production setting;
 mode 1 is retained as a known-good reference for validation.

 NOTE ON phi'(0):  any phi with phi'(0) = 0 -- o = v^2, or any pure power
 o = v^p with p > 1 -- makes v = 0 a stationary point for ANY gradient.  Zeroed
 pixels can then never grow, warm starts from a sparse estimate stall outright,
 and the optimiser can halt at points that are NOT KKT points of the
 constrained problem.  The legacy o = v^2 scheme (previously positivity = 3)
 had exactly this defect and has been removed.  Do not reintroduce a phi whose
 derivative vanishes at the origin.

 Estimate FITS files are written in OBJECT space, so a run started under one
 setting can be warm-started under another.
"""
positivity = a.positivity

if positivity not in (1, 2):
    raise ValueError(f"positivity must be 1 (bounds-only) or 2 (v^2+2cv); "
                     f"got {positivity}.  The legacy v^2 scheme has been "
                     f"removed -- see the note on phi'(0) above.")

_PHI_C = 0.0 if positivity == 1 else read_noise * np.sqrt(gain)

if positivity == 2 and not _PHI_C > 0.0:
    raise ValueError(f"positivity=2 requires c = read_noise*sqrt(gain) > 0, "
                     f"got c={_PHI_C}.  c = 0 reduces phi to the removed v^2 "
                     f"scheme, whose derivative vanishes at the origin.")

def phi(v):
    """object from optimisation variable."""
    if positivity == 1:
        return v
    return v * v + 2.0 * _PHI_C * v

def dphi(v):
    """d(object)/d(variable).  Bounded away from zero on v >= 0 in both modes:
    1 for positivity=1, 2c for positivity=2."""
    if positivity == 1:
        return np.ones_like(v)
    return 2.0 * (v + _PHI_C)

def phi_inv(o):
    """optimisation variable from object (inverse of phi on o >= 0).

    Written as o / (sqrt(o + c^2) + c) rather than sqrt(o + c^2) - c.  The two
    are algebraically identical, but the latter cancels catastrophically when
    o << c^2 -- i.e. exactly in the read-noise-limited regime the companions
    live in (o ~ 1e-8 e- loses ~8 digits at c = 2)."""
    o = np.maximum(o, 0.0)
    if positivity == 1:
        return o
    return o / (np.sqrt(o + _PHI_C ** 2) + _PHI_C)

# --- Error-metric whitening (Change 2) ----------------------------------------
"""
 chi^2 = sum (data - model)^2 / sigma^2, with the same sigma law as
 pd_mrl2.residual():   sigma^2 = sclip(base)/gain + read_noise^2
 (sclip = smooth rectifier, a C-infinity stand-in for clip(x, 0)).
   whiten = 0 : previous unweighted metric, reproduced EXACTLY.
   whiten = 1 : base = DATA. Poisson variance estimated by the data frame;
                weights are FIXED during the optimization.
   whiten = 2 : base = MODEL. Switch to this AFTER the model is a good
                representation of the data (e.g. on a restart = 1 run).
                The gradient then carries the extra d(sigma^2)/d(model)
                term; verify with nderiv = 1 if in doubt.
"""
whiten = a.whiten

# --- MRL error metric (Change 4) ----------------------------------------------
"""
 error_metric = 'ls'  : whitened least-squares (current behaviour, unchanged).
 error_metric = 'mrl' : Jefferies & Hart MRL metric -- minimises the ACF of
                        the whitened residuals. The sigma law follows `whiten`
                        (whiten=1 -> data-based sigma, fixed weights;
                         whiten=2 -> model-based sigma).
                        Can be used standalone (start from the flat initial
                        guess) or as a second stage after LS convergence
                        (restart = 1 to warm-start from the LS solution).
"""
# Spatial lag radius and MC calibration draws:
error_metric   = a.error   # 'ls' | 'mrl'
mrl_R          = 25      # spatial lag radius -- ideally set to ~HWHM of PSF in pixels
mrl_tmax       = 10      # max temporal lag: 0 = 2D spatial only; >0 adds temporal-only lags
mrl_psf_radius = 50      # analysis mask dilation radius in pixels (0 = full frame).
                         # Set to ~radius of PSF footprint so only signal-containing
                         # pixels enter the ACF sums. Support is dilated by this radius.

# Random seed parameter (set to None for non-reproducible results)
random_seed = 41  # Change this for different random realizations
if random_seed is not None:
    np.random.seed(random_seed)
    print(f"Random seed set to: {random_seed} for reproducible results")
else:
    print("No random seed set - results will be non-reproducible")

# Function to load estimate from FITS file for restart
def load_estimate_for_restart(filename):
    """
    Load estimate from FITS file and convert back to internal format
    for use in restart optimization.
    """
    print(f"Loading estimate from {filename} for restart...")

    # Load FITS file: shape is (nslice, dim, dim)
    loaded_data = fits.getdata(filename)
    print(f"Loaded FITS shape: {loaded_data.shape}")

    # Convert back to internal format: (dim, dim, nslice)
    # Step 1: Un-flip the y-axis (flip axis 1 back)
    unflipped = np.flip(loaded_data, axis=1)

    # Step 2: Swap axes back to (dim, dim, nslice)
    internal_format = unflipped.swapaxes(1,2).swapaxes(0, 2)

    print(f"Converted to internal format: {internal_format.shape}")

    return internal_format

# Read object estimate if restarting
if restart == 1:
    #xx = fits.getdata('/Users/stuartjefferies/Desktop/Dispersion_results/Zenith_0_39/estimate_grey.fits')
    #_estimate_path = '/Users/jackr/OneDrive/Documents/Desktop/Hedin/20260930_012915/estimate.fits'
    if a.estimate_path == 'default':
        raise ValueError("No estimate path given.")
    
    _estimate_path = a.estimate_path
    _estimate_dir  = os.path.dirname(_estimate_path)  # directory containing estimate + truth
    xx = load_estimate_for_restart(_estimate_path)
else:
    _estimate_dir = _run_dir   # no restart: source dir is the current run dir


# Compute simulation parameters
"""
 Fried parameter at the shortest wavelength, along the LINE OF SIGHT.
   lambda scaling : r0 ~ lambda^(6/5)
   zenith scaling : r0 ~ (cos z)^(3/5)  -- the turbulence integral along a
                    slant path is sec(z) times the vertical one, and
                    r0 ~ (integral Cn^2 dh)^(-3/5).
 WITHOUT the zenith term an off-zenith run gets the benefit of atmospheric
 dispersion (wavelength-dependent image displacement) with none of its cost
 (degraded seeing), which would make any off-zenith study optimistic.
 No-op at zen = 0.
 """
r0 = r0_500 * ((lamb1/500.0)**(6./5.)) * (np.cos(zenith)**0.6)
print(f'r0 at {lamb1:.0f} nm, zen={zen} deg: {r0:.2f} cm '
      f'(zenith factor (cos z)^0.6 = {np.cos(zenith)**0.6:.4f}, '
      f'D/r0 = {D/r0:.1f})')
delta_slice = int(incr)
nimages = int(nim)
npts = int(D1/2)
dim = int(2**int(np.log10(float(npts))/np.log10(2.0)+1))

print('Number of pixels along one side of phase screen =', dim)
Dscreen = dim*D1/float(npts)
print('Physical size of phase screen=', Dscreen/100.0, 'meters')

nn = dim // 2
n1 = dim // 4
n2 = 3 * dim // 4

# Wavelength slice parameters
rad_max = int(float(dim)/4.0)
rad_min = int(rad_max*(lamb1/lamb2))+1
nslice = int((rad_max - rad_min)/delta_slice)+1
rad_min = rad_max - nslice*delta_slice+delta_slice

print('rad_max', rad_max)
print('rad_min', rad_min)
print('nslice =', nslice)

if nimages < nslice:
    print('number of images needs to be at least the number of wavelength slices')

@numba.jit(nopython=True)
def refract(lam, P, T, fwv):
    """Optimized refraction calculation with Numba JIT"""
    nl15760 = (64.328+(29498.1/(146-(1/lam)**2))+(255.4/(41-(1/lam)**2)))
    nlTP = nl15760 * (P*(1+(1.049-0.0157*T)*1e-6*P)/(720.883*(1+0.003661*T)))
    nlTP = nlTP-fwv*(0.06224-(0.000680/lam**2))/(1+0.003661*T)
    return nlTP

@numba.jit(nopython=True)
def generate_phase_variance(dim, Dscreen, r0):
    """JIT-compiled phase variance calculation"""
    variance = np.zeros((dim, dim))
    i0 = dim // 2
    j0 = dim // 2

    for i in range(dim):
        for j in range(dim):
            if i == i0 and j == j0:
                variance[i, j] = 0.0
            else:
                x = i - i0
                y = j - j0
                r = np.sqrt(x*x + y*y)
                variance[i, j] = 0.023 * (2.0*Dscreen/r0)**(5./3.) * r**(-11./3.)
    return variance

@numba.jit(nopython=True)
def compute_psf_core(pupil_real, pupil_imag, dim):
    """JIT-compiled PSF computation"""
    # This would need to be implemented without FFT since numba doesn't support it
    # For now, keep this in regular Python but pre-allocate arrays
    pass

def load_multiwavelength_object(filename):
    """
    Load pre-generated multi-wavelength object from FITS file.
    
    The FITS file from multiwavelength_object_generator.py has shape:
    (nslice, dim, dim) = (wavelength, y, x)
    
    This function converts it to the internal format:
    (dim, dim, nslice) = (y, x, wavelength)
    
    Parameters:
    -----------
    filename : str
        Path to the multi_wavelength_object.fits file
    
    Returns:
    --------
    object_array : ndarray
        Object array in internal format (dim, dim, nslice)
    """
    print(f"Loading multi-wavelength object from {filename}...")
    
    # Load FITS file: shape is (nslice, dim, dim)
    loaded_data = fits.getdata(filename)
    print(f"Loaded FITS shape: {loaded_data.shape}")
    
    # Convert to internal format: (dim, dim, nslice)
    # Step 1: Un-flip the y-axis (flip axis 1 back)
    unflipped = np.flip(loaded_data, axis=1)
    
    # Step 2: Swap axes back to (dim, dim, nslice)
    object_array = unflipped.swapaxes(1, 2).swapaxes(0, 2)
    
    print(f"Converted to internal format: {object_array.shape}")
    print(f"Object value range: [{object_array.min():.2e}, {object_array.max():.2e}]")
    
    return object_array

# Initialize arrays
lamb = np.zeros(nslice)
rad = np.zeros(nslice)
mask_arr = np.zeros((dim, dim, nslice))
amp = np.zeros((dim, dim, nslice))
psf = np.zeros((dim, dim, nslice))
FTpsf = np.zeros((dim, dim, nslice, nimages), dtype=complex)
object = np.zeros((dim, dim, nslice))
data = np.zeros((dim, dim, nimages))
FTobject = np.zeros((dim, dim, nslice), dtype=complex)
psf_broad = np.zeros((dim, dim, nimages))

# Pre-compute wavelength-dependent parameters
for k in range(nslice):
    rad[k] = rad_min + k*delta_slice
    lamb[k] = lamb1*(float(dim)/4.0)/rad[k]
    mask_arr[:,:,k] = aotools.functions.pupil.circle(rad[k], dim)
    amp[:,:,k] = mask_arr[:,:,k]/np.sqrt(np.sum(mask_arr[:,:,k]*mask_arr[:,:,k]))
    print('rad =', rad[k], ' lambi = ', lamb[k])

# Parameter for multi-wavelength object file
# multiwavelength_object_filename = "multi_wavelength_object.fits"

# Load pre-generated multi-wavelength object from FITS file
start_time = time.time()
object = load_multiwavelength_object(multiwavelength_object_filename)

# Verify dimensions match
if object.shape != (dim, dim, nslice):
    raise ValueError(f"Object dimensions {object.shape} don't match expected ({dim}, {dim}, {nslice})")

print(f"Object loading time: {time.time() - start_time:.2f}s")

# --- Radiometric calibration of the object cube (Change 1) -------------------
# Scale the object so that ONE frame detects `flux_total` e- at `target_mag`,
# split across slices by the detected-count weights from radiometry.calibrate().
# The object's own disk-integrated spectrum enters as the reflectance term, so
# the calibration preserves the object's color (modulated by atm x optics x QE,
# as detected counts must be). Spatial structure within each slice is untouched.
if add_noise == 1:
    import radiometry

    slice_flux = np.array([object[:, :, k].sum() for k in range(nslice)])
    if np.any(slice_flux <= 0):
        raise ValueError("radiometric calibration requires every object slice to have positive flux")
    # TODO: slice_flux currently sums ALL pixels including the delta function,
    # so the magnitude calibration is slightly biased by the delta function flux.
    # Fix: use only support-mask pixels for slice_flux when support_mask_file is set.
    # Impact is small when CSO_scale << 1 but grows as CSO_scale -> 1.

    order = np.argsort(lamb)                     # radiometry expects an ascending grid
    lam_sorted = lamb[order]
    refl_sorted = slice_flux[order] / radiometry.blackbody_flambda(lam_sorted, float(T1))
    # airmass MUST track `zen`: radiometry.calibrate defaults to airmass=1.0,
    # so leaving it unset silently models a zenith observation no matter what
    # `zen` is set to.  sec(z) is accurate to <0.3% out to z = 60 deg; beyond
    # that a Kasten-Young parameterisation would be needed.  No-op at zen = 0.
    _airmass = 1.0 / np.cos(zenith)
    if zen > 60.0:
        print(f"  WARNING: zen={zen} deg -- sec(z)={_airmass:.3f} understates the "
              f"true airmass; consider a Kasten-Young parameterisation.")
    weights_sorted, flux_total, rad_info = radiometry.calibrate(
        target_mag, mag_band, mag_system, lam_nm=lam_sorted,
        T_sun=float(T1),                 # MUST match the B_lambda(T1) division above
        reflectance=refl_sorted, D_tel_m=D / 100.0, airmass=_airmass,
        obstruction_frac=obstruction_frac, t_exp_s=t_exp_s, verbose=True)
    weights = np.empty(nslice)
    weights[order] = weights_sorted              # map back to the (descending) lamb order

    for k in range(nslice):
        object[:, :, k] *= flux_total * weights[k] / slice_flux[k]
    print(f"Object calibrated: total detected flux per frame = {flux_total:.3e} e-")

# Compute FFT of object
for k in range(nslice):
    FTobject[:,:,k] = np.fft.fft2(object[:,:,k])

if display == 1:
    for kk in range(nslice):
        plt.imshow(object[:,:,kk])
        plt.title('Truth object @' + str(int(lamb[kk])) + ' nm')
        _save_plot('truth_object')

# Refraction reference
lam = 0.5  # reference wavelength in microns
n500 = refract(lam, P, T, fwv) * 1e-6 - 1

# Main simulation loop - generate atmospheric turbulence and PSFs
i0 = dim // 2
j0 = dim // 2

# Pre-allocate and pre-compute everything possible
A = np.zeros((dim, dim), dtype=complex)
C = np.zeros((dim, dim), dtype=complex)
pupil = np.zeros((dim, dim), dtype=complex)
phase = np.zeros((dim, dim))
# (the `delta` scratch array is gone: dispersion now uses disp_kernel below)

# Pre-compute phase variance map (biggest speedup!)
phase_variance = generate_phase_variance(dim, Dscreen, r0)

# Pre-compute refraction values for all wavelengths
refraction_values = np.zeros(nslice)
for k in range(nslice):
    lam = lamb[k] * 1e-3
    nj = refract(lam, P, T, fwv) * 1e-6 - 1
    DeltaR = 206265.0 * (nj - n500) * ma.tan(zenith)
    refraction_values[k] = DeltaR / pixscale
    print('lambda = ',lam,' dispersion = ',DeltaR, ' [arcseec]')

# --- Dispersion sanity check (Fix 4) -----------------------------------------
# The Fourier shift below is periodic: an over-large displacement WRAPS to the
# opposite edge instead of clipping, which is silent and would corrupt the PSF.
# Fail loudly rather than produce a plausible-looking wrapped result.
if disperse == 1:
    _disp_max  = float(np.abs(refraction_values).max())
    _disp_span = float(refraction_values.max() - refraction_values.min())
    print(f'Dispersion: max |shift| = {_disp_max:.2f} px, '
          f'span across band = {_disp_span:.2f} px  (array is {dim} px)')
    if _disp_max > dim / 4.0:
        raise ValueError(
            f"atmospheric dispersion shift {_disp_max:.1f} px exceeds dim/4 = "
            f"{dim/4:.0f} px at zen={zen} deg; the Fourier shift would wrap. "
            f"Reduce `zen`, or enlarge `dim`.")

# --- Pre-computed dispersion shift kernels (Fix 3) ---------------------------
# The displacement is a general real number, so applying it by planting a delta
# at int(i0 - dr) quantised it to whole pixels -- up to ~0.5 px of error, which
# at zen = 15 deg is ~17% of the spacing between adjacent slices.  The relative
# slice offsets ARE the wavelength diversity, so they must be sub-pixel exact.
# The shift theorem is used instead, matching the convention already used for
# the companion PSF templates in the spectral-extraction section.
#
# Kernel = FT of a delta at (i0 - dr, i0), i.e. exactly what the old code
# convolved with, but evaluated in closed form and off the integer grid.  It is
# frame-independent, so hoisting it out of the frame loop also removes
# nslice*nimages redundant forward FFTs.
if disperse == 1:
    _fy_disp = np.fft.fftfreq(dim)[:, None]
    _fx_disp = np.fft.fftfreq(dim)[None, :]
    disp_kernel = np.empty((nslice, dim, dim), dtype=complex)
    for k in range(nslice):
        disp_kernel[k] = np.exp(-1j * 2.0 * np.pi * (
            (i0 - refraction_values[k]) * _fy_disp + i0 * _fx_disp))
    del _fy_disp, _fx_disp

# Pre-compute scaling factors
scaling_factors = lamb1 / lamb  # s2 values

# --- Diffraction-limited PSF peaks for Strehl computation -------------------
# The DL PSF is the PSF with zero phase: FT of the unaberrated pupil amp[:,:,k].
# amp[:,:,k] is already normalised to unit energy (L322), so the DL PSF peak
# is directly comparable to the aberrated PSF peak (same normalisation).
# Strehl = aberrated_peak / DL_peak, computed per frame and accumulated.
dl_psf_peak = np.zeros(nslice)
for k in range(nslice):
    _dl_ft  = np.fft.ifft2(amp[:, :, k]) * float(dim)
    _dl_psf = _dl_ft.real**2 + _dl_ft.imag**2
    dl_psf_peak[k] = _dl_psf.max()
    print(f"  DL PSF peak @ {lamb[k]:.1f} nm (slice {k}): {dl_psf_peak[k]:.4e}")

# Strehl running accumulators: sum and sum-of-squares per slice over all frames
strehl_sum  = np.zeros(nslice)
strehl_sum2 = np.zeros(nslice)

print("Starting main simulation loop...")
start_time = time.time()

for l in range(nimages):
    # Generate phase screen - use pre-computed variance
    A.real = np.random.normal(0, 1, (dim, dim)) * np.sqrt(phase_variance/2)
    A.imag = np.random.normal(0, 1, (dim, dim)) * np.sqrt(phase_variance/2)
    A[i0, j0] = 0  # Zero spatial frequency

    # Fourier transform
    C = np.fft.fft2(np.fft.fftshift(A))

    Dscreenr0 = (np.var(C.real)/1.03)**0.6
    Dr0 = Dscreenr0/2.0
    print('Frame #', l, ': D/r_0 = ', round(Dr0, 2), '(at shortest wavelength, Nyquist sampled)')

    # Process each wavelength slice
    for k in range(nslice):
        s2 = scaling_factors[k]  # Use pre-computed value
        temp = rescale(mask_arr[:,:,nslice-1]*C.real, scale=s2, order=3)
        s0 = int(s2*dim+0.5)
        nn1 = int((dim-s0)/2+0.5)
        nn2 = int(nn1+s0+0.5)

        phase.fill(0)
        if nn2 <= dim and nn1 >= 0:  # Bounds check
            phase[nn1:nn2, nn1:nn2] = temp[:,:] * scaling_factors[k]

        # Build complex pupil function (vectorized)
        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)
        pupil.real = amp[:,:,k] * cos_phase
        pupil.imag = amp[:,:,k] * sin_phase

        # Compute PSF
        FT = np.fft.ifft2(pupil) * float(dim)
        psftmp = FT.real**2 + FT.imag**2

        if disperse == 1:
            # Sub-pixel atmospheric dispersion via the shift theorem.
            # Equivalent to the previous delta-convolution but not quantised to
            # the integer grid; kernel precomputed above.
            psf[:,:,k] = np.fft.fftshift(np.real(np.fft.ifft2(
                np.fft.fft2(psftmp) * disp_kernel[k])))
        else:
            psf[:,:,k] = psftmp

        # Accumulate Strehl: peak of aberrated PSF / peak of DL PSF
        _s = psf[:, :, k].max() / dl_psf_peak[k]
        strehl_sum[k]  += _s
        strehl_sum2[k] += _s * _s

        psf_broad[:,:,l] += psf[:,:,k]
        FTpsf[:,:,k,l] = np.fft.fft2(psf[:,:,k])

        # Compute data (vectorized)
        data[:,:,l] += np.real(np.fft.ifft2(FTobject[:,:,k] * FTpsf[:,:,k,l]))

    if display == 1:
        plt.imshow(data[:,:,l])
        plt.title("data frame #" + str(l))
        _save_plot('data_frame')
        plt.imshow(np.fft.fftshift(psf_broad[:,:,l]))
        plt.title('Broadband PSF')
        _save_plot('broadband_psf')

print(f"Main simulation loop time: {time.time() - start_time:.2f}s")

# --- Strehl statistics across frames -----------------------------------------
strehl_mean = strehl_sum  / nimages
strehl_std  = np.sqrt(np.maximum(strehl_sum2 / nimages - strehl_mean**2, 0.0))
print("\nStrehl ratio statistics per wavelength slice:")
print(f"  {'Slice':>5}  {'lambda(nm)':>10}  {'Mean Strehl':>12}  {'Std Strehl':>11}")
for k in range(nslice):
    print(f"  {k:5d}  {lamb[k]:10.1f}  {strehl_mean[k]:12.4f}  {strehl_std[k]:11.4f}")

# --- Add photon (Poisson) + read (Gaussian) noise (Change 1) -----------------
if add_noise == 1:
    data_noisefree = data.copy()
    noise_rng = np.random.default_rng(random_seed)   # independent, reproducible stream
    data = (noise_rng.poisson(np.clip(data, 0.0, None)).astype(float)
            + noise_rng.normal(0.0, read_noise, data.shape))
    mean_frame_counts = data_noisefree.sum() / nimages
    print(f"Noise added: mean counts/frame = {mean_frame_counts:.3e} e- "
          f"(calibrated target {flux_total:.3e} e-), read noise = {read_noise:.1f} e-")

# --- Support mask projection setup -------------------------------------------
# NOTE: moved ABOVE the initialisation so the uniform starting object can be
# spread over the free pixels only.  Previously the mask was applied after the
# uniform object was built, discarding the out-of-support flux and leaving the
# initial model too faint by a further factor n_support/(dim*dim).
SUPPORT_MASK = None   # None = no constraint
if support_mask_file != '':
    _smraw = fits.getdata(support_mask_file).astype(float)
    if _smraw.ndim != 2 or _smraw.shape != (dim, dim):
        raise ValueError(f"support_mask_file must be a 2D ({dim}x{dim}) FITS image, "
                         f"got shape {_smraw.shape}")
    # fit[i,j,k] is saved to FITS row dim-1-i (due to flip in save transform).
    # To constrain FITS rows R, we need internal rows dim-1-R to be free.
    # Y-flipping the mask achieves this mapping.
    SUPPORT_MASK = (np.flip(_smraw, axis=0) > 0.5).astype(float)   # (dim, dim)
    n_support = int(SUPPORT_MASK.sum())
    print(f"Support mask loaded: {n_support} / {dim*dim} pixels active "
          f"({100.*n_support/(dim*dim):.2f}%)")

# --- Initialise optimisation variables ---------------------------------------
"""
 The optimiser works on `vars`; the object is o = phi(vars) (see `positivity`).

 Cold start: uniform object over the free pixels, scaled so the initial object
 flux equals the mean detected flux per frame.  Each psf[:,:,k] is normalised
 to unit sum (amp is L2-normalised above, so by Parseval
 sum(|ifft2(pupil)*dim|^2) = sum(amp^2) = 1), hence

     sum_pixels data[:,:,l]  =  sum_k sum_pixels object[:,:,k]

 i.e. the per-frame detected flux IS the total object flux.  Therefore

     o_init = (sum(data)/nimages) / n_free      and      vars = phi_inv(o_init)

 The previous version set vars = sum(data)/(dim*dim*nslice), i.e. it assigned
 the flux target to VARS rather than to the OBJECT, and summed data over all
 frames instead of averaging.  For dim=128, nslice=10, nimages=100 and
 ~1e8 e-/frame that made the initial object ~6e6 times too bright.
"""

_n_free      = int(SUPPORT_MASK.sum()) * nslice if SUPPORT_MASK is not None \
               else dim * dim * nslice
_flux_target = max(float(np.sum(data)) / nimages, 0.0)   # e- per frame
_o_init      = _flux_target / _n_free                    # e- per free pixel
_v_init      = float(phi_inv(np.asarray(_o_init)))

vars = np.full(dim * dim * nslice, _v_init)
if SUPPORT_MASK is not None:
    vars = (vars.reshape(nslice, dim, dim) * SUPPORT_MASK[None, :, :]).flatten()

print(f"Cold start: positivity={positivity} (phi offset c={_PHI_C:.4g})")
print(f"            target object flux = {_flux_target:.4e} e- per frame "
      f"over {_n_free} free pixels")
print(f"            object = {_o_init:.4e} e-/pixel/slice, vars = {_v_init:.4e}")
print(f"            initial object flux = {float(phi(vars).sum()):.4e} e-  "
      f"(target {_flux_target:.4e})")

# Known-answer check on the flux bookkeeping above: FTpsf[0,0,k,l] is the DC
# term of psf[:,:,k], i.e. sum(psf).  If the PSFs are not unit-sum the flux
# identity is wrong and every downstream flux number is mis-scaled.
_psf_dc = np.abs(FTpsf[0, 0, :, :].real)
if not np.allclose(_psf_dc, 1.0, rtol=1e-6, atol=1e-9):
    print(f"  WARNING: PSF sums are not unity "
          f"[{_psf_dc.min():.6f}, {_psf_dc.max():.6f}]; "
          f"cold-start flux scaling assumes unit-sum PSFs")

if restart == 1:
    # Warm-start vars from the estimate loaded at the top of the script.
    # xx is (dim, dim, nslice); vars is flattened as (nslice, dim, dim).
    # Estimate FITS files are in OBJECT space, so they are interchangeable
    # between positivity settings.
    vars = phi_inv(xx).transpose(2, 0, 1).ravel().copy()
    print(f"Restart: vars warm-started from estimate FITS "
          f"(object flux {float(phi(vars).sum()):.4e} e-)")

if SUPPORT_MASK is not None:
    # Enforce the support on whichever initialisation was used.
    vars = (vars.reshape(nslice, dim, dim) *
            SUPPORT_MASK[None, :, :]).flatten()
    _obj0 = phi(vars.reshape(nslice, dim, dim))
    _flux_in  = float(_obj0[:, SUPPORT_MASK > 0.5].sum())
    _flux_out = float(_obj0[:, SUPPORT_MASK <= 0.5].sum())
    print(f"  Support constraint applied to initial vars")
    print(f"  Flux inside support : {_flux_in:.4e}")
    print(f"  Flux outside support: {_flux_out:.4e}  (should be 0.0)")

resid = np.zeros((dim, dim, nimages))
amodel = np.zeros((dim, dim, nimages))
grad = np.zeros(dim * dim * nslice)

# --- Whitening machinery (Change 2) --------------------------------------------
def _sclip(x, eps=1.0):
    """Smooth rectifier from pd_mrl2: C-infinity, differs from clip(x, 0) by
    <= eps/2 (half a count). Keeps the model-based sigma differentiable where
    the model crosses zero (a hard clip kinks the gradient and kills the
    L-BFGS-B line search -- pd_mrl2's measured failure mode)."""
    return 0.5 * (x + np.sqrt(x * x + eps * eps))

def _dsclip(x, eps=1.0):
    """d/dx of _sclip."""
    return 0.5 * (1.0 + x / np.sqrt(x * x + eps * eps))

# grad_seed holds dchi2/dmodel per pixel per frame; func() fills it, dfunc()
# correlates it with the PSFs. For the unweighted metric grad_seed = -2*resid,
# which reproduces the previous gradient exactly.
grad_seed = np.zeros((dim, dim, nimages))

if whiten != 0:
    if add_noise == 0:
        print("WARNING: whiten != 0 with add_noise = 0 -- the sigma law assumes "
              "Poisson + read noise; weighting noise-free data is not meaningful.")
    if whiten == 1:
        # Data-based Poisson variance estimate: FIXED during the optimization.
        inv_var_data = 1.0 / (_sclip(data) / gain + read_noise ** 2)
        print(f"Whitened metric (whiten=1): sigma^2 = sclip(DATA)/gain + read^2 "
              f"(fixed weights); 1/sigma^2 in "
              f"[{inv_var_data.min():.3e}, {inv_var_data.max():.3e}]")
    elif whiten == 2:
        print("Whitened metric (whiten=2): IRLS -- sigma^2 = sclip(MODEL)/gain + read^2 "
              "(fixed per outer iteration; no d(sigma^2)/d(model) term in gradient)")
        # Shared inverse-variance array updated by the IRLS loop before each
        # minimize call. Initialised here from the starting model.
        inv_var_model = np.ones((dim, dim, nimages))  # will be set in IRLS loop
    else:
        raise ValueError(f"whiten must be 0, 1, or 2 (got {whiten})")

# --- MRL setup (Change 4) -----------------------------------------------------
if error_metric == 'mrl':
    if whiten == 0:
        print("WARNING: error_metric='mrl' with whiten=0 -- MRL residuals will "
              "be unweighted (sigma=1 everywhere). whiten=1 or 2 is recommended.")

    # Build the lag menu:
    #   tau=0 : half-plane spatial lags within circular radius mrl_R
    #           (Δy>0, or Δy=0 and Δx>0) -- avoids double-counting.
    #   tau=1..mrl_tmax : ZERO-SPATIAL-LAG temporal lags only (dy=0, dx=0).
    #           Static model error appears in all frames; photon/read noise
    #           is independent between frames so temporal lags are noise-bias-free.
    # Each entry is (tau, dy, dx).
    _menu = []
    for dy in range(-mrl_R, mrl_R + 1):
        for dx in range(-mrl_R, mrl_R + 1):
            if (dy > 0 or (dy == 0 and dx > 0)) and (dy**2 + dx**2) <= mrl_R**2:
                _menu.append((0, dy, dx))
    for tau in range(1, mrl_tmax + 1):
        _menu.append((tau, 0, 0))   # zero-spatial-lag temporal lags only
    MRL_TAU = np.array([m[0] for m in _menu], dtype=int)
    MRL_DY  = np.array([m[1] for m in _menu], dtype=int)
    MRL_DX  = np.array([m[2] for m in _menu], dtype=int)
    n_lags    = len(_menu)
    n_spatial = sum(1 for m in _menu if m[0] == 0)
    n_temporal = n_lags - n_spatial
    print(f"MRL: {n_lags} lags total -- {n_spatial} spatial (tau=0, circular R={mrl_R})"
          + (f", {n_temporal} temporal (zero-spatial-lag, tau=1..{mrl_tmax})" if mrl_tmax > 0 else ""))

    # --- Analysis mask: support dilated by mrl_psf_radius ---
    # Only pixels inside this mask contribute to the ACF sums.
    # This restricts analysis to the signal-containing region, avoiding
    # dilution by noise-only pixels far from the satellite.
    if mrl_psf_radius > 0 and SUPPORT_MASK is not None:
        from scipy.ndimage import binary_dilation
        _struct = np.ones((2*mrl_psf_radius+1, 2*mrl_psf_radius+1))
        MRL_MASK = binary_dilation(SUPPORT_MASK > 0.5, structure=_struct).astype(float)
        print(f"MRL: analysis mask = support dilated by {mrl_psf_radius} px, "
              f"{int(MRL_MASK.sum())} / {dim*dim} pixels active")
    else:
        MRL_MASK = np.ones((dim, dim))   # full frame
        print(f"MRL: analysis mask = full frame ({dim}x{dim})")

    def _mrl_residual(vars_3d):
        """Whitened residual (dim, dim, nimages) from the current object vars."""
        mod = np.zeros((dim, dim, nimages))
        for _k in range(nslice):
            obj_k = phi(vars_3d[_k])
            for _l in range(nimages):
                mod[:, :, _l] += np.real(np.fft.ifft2(
                    np.fft.fft2(obj_k) * FTpsf[:, :, _k, _l]))
        if whiten == 2:
            # IRLS: use the fixed inv_var_model from the outer loop
            _sig = np.sqrt(1.0 / (inv_var_model + 1e-30))
        elif whiten == 1:
            _sig = np.sqrt(inv_var_data ** -1)
        else:
            _sig = np.ones_like(mod)
        return (data - mod) / _sig, mod, _sig

    # --- Analytic null -------------------------------------------------------
    # The null is the distribution of C(tau,dy,dx) for WHITE residuals.  The lag
    # set excludes (0,0,0), so for independent zero-mean residuals
    #
    #     E[C] = 0                             exactly, at every lag,
    #     Var[C] = s^4 * N_mask / (nimages - tau),
    #
    # because C sums (nimages - tau) * N_mask product terms, each of variance
    # E[r0^2] E[r1^2] = s^4, and the terms are pairwise uncorrelated (the value
    # r(x0) appears in two terms of a tau=0 sum, but their covariance is
    # E[r(x0)^2] E[r(x0+d)] E[r(x0-d)] = 0).  Poisson's non-Gaussian higher
    # moments do not enter: Var[r0*r1] = E[r0^2] E[r1^2] for independent factors
    # whatever their skewness.
    #
    # This replaces a Monte-Carlo calibration that resampled about `data`.  Where
    # data < 0 -- about half of a read-noise-dominated background -- clipping the
    # Poisson mean at zero gave those pixels a deterministic positive residual
    # mean, and the resulting null was offset by ~1.5e4 per lag.  The error was
    # invisible while the analysis mask covered only bright pixels and dominant
    # once it covered the field.  A closed-form null cannot acquire that class of
    # bug, is exact where 32 MC draws gave sigma_0 to only ~13%, and makes
    # mu_0 = 0 the target by definition, so any departure from whiteness -- rather
    # than being absorbed into the null -- registers as objective.
    #
    # s^2 is measured rather than assumed to be 1: _sclip adds 0.5 in the faint
    # limit, so the whitened residual has variance ~0.89 where the model is near
    # zero, and sigma_0 scales as s^2.
    # sigma_0 depends on the whitening through s^2, so the null must be
    # (re)calibrated whenever inv_var_model changes -- i.e. after it is first
    # initialised and at every IRLS outer step, exactly like the weights
    # themselves.  It must stay FIXED across an inner minimize, or the objective
    # would move under L-BFGS-B.  Held in a dict to avoid global/nonlocal scoping.
    _MRL_NULL = {"mean": np.zeros(n_lags),
                 "std":  np.ones(n_lags),
                 "ready": False}

    def _mrl_calibrate_null(vars_flat, quiet=False):
        _Nmask = float(MRL_MASK.sum())
        _r, _, _ = _mrl_residual(vars_flat.reshape(nslice, dim, dim))
        _s2 = float((_r ** 2 * MRL_MASK[:, :, None]).sum() / (_Nmask * nimages))
        _MRL_NULL["mean"] = np.zeros(n_lags)
        _MRL_NULL["std"]  = _s2 * np.array([np.sqrt(_Nmask / (nimages - tau))
                                            for (tau, dy, dx) in _menu])
        _MRL_NULL["ready"] = True
        if not quiet:
            print(f"MRL: analytic null, mu_0 = 0, "
                  f"sigma_0 = s^2*sqrt(N_mask/(L-tau)); N_mask = {int(_Nmask)}, "
                  f"s^2 = {_s2:.4f}, sigma_0 in "
                  f"[{_MRL_NULL['std'].min():.2f}, {_MRL_NULL['std'].max():.2f}]")
        if not 0.5 < _s2 < 2.0:
            print(f"  WARNING: whitened-residual variance s^2 = {_s2:.4f} is far "
                  f"from unity; the noise model and the MRL null are inconsistent.")
        return _s2

    def func_mrl(vars):
        """MRL objective: sum over (tau, dy, dx) of standardised ACF^2."""
        global function_eval_count
        function_eval_count += 1
        if not _MRL_NULL["ready"]:
            raise RuntimeError("MRL objective evaluated before the null was "
                               "calibrated; call _mrl_calibrate_null() after "
                               "inv_var_model is set.")
        vars_3d = vars.reshape(nslice, dim, dim)
        r, _, _ = _mrl_residual(vars_3d)
        val = 0.0
        for _j, (tau, dy, dx) in enumerate(_menu):
            r0 = r[:, :, :nimages - tau] if tau > 0 else r
            r1 = np.roll(np.roll(
                     r[:, :, tau:] if tau > 0 else r,
                     -dy, axis=0), -dx, axis=1)
            acf_lag = float((r0 * r1 * MRL_MASK[:,:,None]).sum(axis=(0,1)).mean())
            val += ((acf_lag - _MRL_NULL['mean'][_j]) / _MRL_NULL['std'][_j]) ** 2
        # Per-lag normalisation: sigma_0 scales as sqrt(N_mask), so without this
        # the objective is not comparable between runs with different analysis
        # masks or lag sets.  Normalised, it reads as chi^2 per lag and should
        # sit near 1 for white residuals.
        val /= n_lags
        if function_eval_count == 1 and val < 0.5:
            print(f"  WARNING: initial MRL objective {val:.4f} per lag is well "
                  f"below 1; residuals cannot be whiter than white, so the null "
                  f"calibration or the noise model is inconsistent.")
        chi_square_history.append(val)
        function_eval_history.append(function_eval_count)
        return val

    def dfunc_mrl(vars):
        """Gradient of MRL objective via chain rule through the 3D ACF."""
        vars_3d = vars.reshape(nslice, dim, dim)
        r, mod, sig = _mrl_residual(vars_3d)
        dL_dr = np.zeros_like(r)
        for _j, (tau, dy, dx) in enumerate(_menu):
            r0 = r[:, :, :nimages - tau] if tau > 0 else r
            r1 = np.roll(np.roll(
                     r[:, :, tau:] if tau > 0 else r,
                     -dy, axis=0), -dx, axis=1)
            acf_lag = float((r0 * r1 * MRL_MASK[:,:,None]).sum(axis=(0,1)).mean())
            # /n_lags matches the per-lag normalisation applied in func_mrl
            z = ((acf_lag - _MRL_NULL['mean'][_j])
                 / _MRL_NULL['std'][_j] ** 2 / n_lags)
            n_pairs = nimages - tau   # frame pairs at this tau
            # gradient w.r.t. r0 frames: d(ACF)/d(r0) = r1 / n_pairs
            # gradient w.r.t. r1 frames: d(ACF)/d(r1) = r0_unshifted / n_pairs
            r0_unshift = np.roll(np.roll(r0, dy, axis=0), dx, axis=1)
            # Correct mask gradient: two terms have different mask support.
            # Term 1 (from r0): MRL_MASK * r1
            # Term 2 (from r1): roll(MRL_MASK, dy, dx) * r0_unshift
            _M_shifted = np.roll(np.roll(MRL_MASK, dy, axis=0), dx, axis=1)
            if tau == 0:
                # r0 == r1 == r: both contributions go to the same frames
                dL_dr += 2.0 * z * (MRL_MASK[:,:,None]*r1 + _M_shifted[:,:,None]*r0_unshift) / n_pairs
            else:
                # r0 = r[:,:,0:nimages-tau], r1 = shifted r[:,:,tau:nimages]
                dL_dr[:, :, :nimages - tau] += 2.0 * z * MRL_MASK[:,:,None]   * r1         / n_pairs
                dL_dr[:, :, tau:]           += 2.0 * z * _M_shifted[:,:,None] * r0_unshift  / n_pairs
        # chain rule: dL/dmodel
        # whiten==2 IRLS: sigma fixed, no d(sigma^2)/d(model) term
        if whiten == 2:
            dL_dmod = -dL_dr / sig
        elif whiten == 1:
            dL_dmod = -dL_dr / sig
        else:
            dL_dmod = -dL_dr
        # chain rule through convolution and obj = phi(vars)
        grad_3d = np.zeros((nslice, dim, dim))
        for _k in range(nslice):
            cross = np.zeros((dim, dim), dtype=complex)
            for _l in range(nimages):
                cross += np.fft.ifft2(
                    np.fft.fft2(dL_dmod[:, :, _l]) * np.conj(FTpsf[:, :, _k, _l]))
            grad_3d[_k] = cross.real * dphi(vars_3d[_k])
        return grad_3d.flatten()

# Track optimization progress
chi_square_history = []
iteration_history = []
function_eval_history = []
current_iteration = 0
function_eval_count = 0

def func(vars):
    """Optimized objective function with progress tracking"""
    global resid, amodel, function_eval_count

    function_eval_count += 1

    # Reshape vars to 3D for faster access
    obj = phi(vars.reshape(nslice, dim, dim))   # see `positivity` switch
    amodel.fill(0)  # Clear previous model
    chi2 = 0.0

    for l in range(nimages):
        amodel_frame = np.zeros((dim, dim))

        # Vectorized convolution for all wavelengths
        for k in range(nslice):
            conv = np.fft.ifft2(np.fft.fft2(obj[k]) * FTpsf[:,:,k,l])
            amodel_frame += np.real(conv)

        amodel[:,:,l] = amodel_frame

        # Compute residuals and chi-square
        resid[:,:,l] = data[:,:,l] - amodel[:,:,l]
        if whiten == 2:
            # IRLS: inv_var_model is fixed for this minimize call,
            # updated by the outer IRLS loop. No d(sigma^2)/d(model) term.
            chi2 += np.sum(resid[:,:,l] ** 2 * inv_var_model[:,:,l])
            grad_seed[:,:,l] = -2.0 * resid[:,:,l] * inv_var_model[:,:,l]
        elif whiten == 1:
            # sigma^2 from the data: fixed weights, gradient seed is linear
            chi2 += np.sum(resid[:,:,l] ** 2 * inv_var_data[:,:,l])
            grad_seed[:,:,l] = -2.0 * resid[:,:,l] * inv_var_data[:,:,l]
        else:
            chi2 += np.sum(resid[:,:,l]**2)
            grad_seed[:,:,l] = -2.0 * resid[:,:,l]

    # Track progress (store every function evaluation for detailed plotting)
    chi_square_history.append(chi2)
    function_eval_history.append(function_eval_count)

    return chi2

def callback_func(xk):
    """Callback function to track iterations"""
    global current_iteration
    current_iteration += 1
    iteration_history.append(current_iteration)

    # Print progress every few iterations
    if current_iteration % 5 == 0 or current_iteration <= 5:
        current_chi2 = chi_square_history[-1] if chi_square_history else 0
        _mlabel = "MRL" if error_metric == 'mrl' else "chi2"
        print(f"   Iteration {current_iteration}: {_mlabel} = {current_chi2:.6e}")

def dfunc(vars):
    """Optimized gradient function"""
    grad_3d = np.zeros((nslice, dim, dim))  # Work in 3D space
    vars_3d = vars.reshape(nslice, dim, dim)

    # Vectorized cross-correlation
    for k in range(nslice):
        cross = np.zeros((dim, dim), dtype=complex)
        for l in range(nimages):
            FTres = np.fft.fft2(grad_seed[:,:,l])
            cross += np.fft.ifft2(FTres * np.conj(FTpsf[:,:,k,l]))

        # grad_seed = dchi2/dmodel from func(); chain rule through the
        # convolution and obj = phi(vars).  dphi is bounded away from zero
        # (1 for positivity=1, 2c for positivity=2), so no pixel can have an
        # identically vanishing gradient purely because its object value is 0.
        grad_3d[k] = cross.real * dphi(vars_3d[k])

    return grad_3d.flatten()  # Return flattened for optimizer

# Run optimization with L-BFGS-B
print("\n" + "="*60)
print("OPTIMIZATION SETTINGS")
print("="*60)
print(f"  Restart         : {restart}  "
      f"{'(warm-start from estimate FITS)' if restart == 1 else '(cold start, flat initial guess)'}")
print(f"  Error metric    : {error_metric.upper()}"
      + (f"  (MRL circular R={mrl_R}, psf_radius={mrl_psf_radius}, tmax={mrl_tmax}, {n_lags} lags, analytic null)"
         if error_metric == 'mrl' else "  (whitened least-squares)"))
_wlabel = {0: "NONE (unweighted, sigma=1)", 1: "DATA-based sigma (fixed weights)",
           2: "MODEL-based sigma (updates each iteration)"}
print(f"  Whitening       : whiten={whiten}  {_wlabel[whiten]}")
print(f"  Support mask    : "
      + (f"{support_mask_file}" if support_mask_file else "none"))
print(f"  add_noise       : {add_noise}"
      + (f"  (mag={target_mag} {mag_system}-{mag_band}, "
         f"read_noise={read_noise} e-, gain={gain})" if add_noise == 1 else ""))
print(f"  niter / nslice  : {niter} / {nslice}")
print("="*60 + "\n")
print("Starting optimization...")
start_time = time.time()

if niter == 0:
    print("  niter=0: skipping optimization, using current vars as solution.")
    res1 = vars
else:
    # For whiten==2: initialise inv_var_model from current vars before any
    # evaluation so that chi-square is computed with the correct weights.
    if whiten == 2:
        _iv_obj = phi(vars.reshape(nslice, dim, dim))
        _iv_mod = np.zeros((dim, dim, nimages))
        for _k in range(nslice):
            for _l in range(nimages):
                _iv_mod[:,:,_l] += np.real(np.fft.ifft2(
                    np.fft.fft2(_iv_obj[_k]) * FTpsf[:,:,_k,_l]))
        inv_var_model[:] = 1.0 / (_sclip(_iv_mod) / gain + read_noise**2)
        print(f"  inv_var_model initialised from current vars: "
              f"[{inv_var_model.min():.3e}, {inv_var_model.max():.3e}]")
    if error_metric != 'ls':
        # Null depends on the whitening; calibrate now that inv_var_model is set.
        _mrl_calibrate_null(vars)
    y = func(vars) if error_metric == 'ls' else func_mrl(vars)
    print(f'Initial {"chi-square" if error_metric == "ls" else "MRL objective"} value =', y)
    if error_metric == 'mrl':
        _chi2_initial = func(vars)
        print(f'Initial chi-square (data fit) = {_chi2_initial:.6e}  '
              f'(weighted by model-based sigma from loaded estimate)')
    yy = dfunc(vars) if error_metric == 'ls' else dfunc_mrl(vars)

    # Enhanced derivative validation if requested
    if nderiv == 1:
        print("\n" + "="*60)
        print("DERIVATIVE VALIDATION")
        print("="*60)

        def finite_difference_test(vars, func, dfunc, n_tests=20, rel_tol=1e-6):
            """
            Comprehensive finite difference gradient validation
            """
            grad_analytic = dfunc(vars)
            n_vars = len(vars)

            # Test multiple step sizes to find optimal
            step_sizes = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
            best_step = None
            best_error = float('inf')

            print(f"Testing {n_tests} random variables with multiple step sizes...")

            # Select indices where vars is nonzero (inside support, object has flux).
            np.random.seed(123)  # For reproducible derivative tests
            # Near-zero pixels carry real gradient in both modes (dphi(0) is 1
            # or 2c, never 0) and are exactly the regime the companions live
            # in, so they are worth including rather than screening out.
            _nonzero_idx = np.where(vars > 1e-10)[0]
            if len(_nonzero_idx) < n_tests:
                print(f'WARNING: only {len(_nonzero_idx)} nonzero vars found; '
                      f'testing all of them')
                test_indices = _nonzero_idx
            else:
                test_indices = np.random.choice(_nonzero_idx, size=n_tests, replace=False)
            print(f'  Testing {len(test_indices)} nonzero vars '
                  f'(from {len(_nonzero_idx)} nonzero out of {n_vars} total)')

            results = []

            for step in step_sizes:
                errors = []
                max_rel_error = 0

                for i, idx in enumerate(test_indices):
                    orig_val = vars[idx]

                    # Forward difference: f(x + h)
                    vars[idx] = orig_val + step
                    f_plus = func(vars)

                    # Backward difference: f(x - h)
                    vars[idx] = orig_val - step
                    f_minus = func(vars)

                    # Restore and recompute grad_seed at original point
                    vars[idx] = orig_val
                    func(vars)                          # refresh grad_seed at x
                    analytic_grad_i = dfunc(vars)[idx]  # gradient at x

                    # Central difference approximation
                    numeric_grad = (f_plus - f_minus) / (2 * step)

                    # Calculate relative error
                    if abs(analytic_grad_i) > 1e-10:
                        rel_error = abs(numeric_grad - analytic_grad_i) / abs(analytic_grad_i)
                    else:
                        rel_error = abs(numeric_grad - analytic_grad_i)

                    errors.append(rel_error)
                    max_rel_error = max(max_rel_error, rel_error)

                    if len(results) < n_tests and step == step_sizes[0]:
                        results.append({
                            'idx': idx,
                            'numeric': numeric_grad,
                            'analytic': analytic_grad_i,
                            'abs_error': abs(numeric_grad - analytic_grad_i),
                            'rel_error': rel_error,
                            'step': step
                        })

                mean_error = np.mean(errors)
                print(f"Step size {step:.1e}: Mean rel error = {mean_error:.2e}, Max rel error = {max_rel_error:.2e}")

                if mean_error < best_error:
                    best_error = mean_error
                    best_step = step

            print(f"\nBest step size: {best_step:.1e} (mean rel error: {best_error:.2e})")

            # Detailed results with best step size
            print(f"\nRe-testing with optimal step size {best_step:.1e}:")
            print("-" * 80)
            print(f"{'Index':<8} {'Numeric':<12} {'Analytic':<12} {'Abs Error':<12} {'Rel Error':<12} {'Status':<10}")
            print("-" * 80)

            # Re-run with best step size for detailed output
            passed = 0
            failed = 0

            for i, idx in enumerate(test_indices[:15]):  # Show first 15 for readability
                orig_val = vars[idx]

                vars[idx] = orig_val + best_step
                f_plus = func(vars)
                vars[idx] = orig_val - best_step
                f_minus = func(vars)
                vars[idx] = orig_val
                func(vars)                          # refresh grad_seed at x
                analytic_grad = dfunc(vars)[idx]    # gradient at x

                numeric_grad = (f_plus - f_minus) / (2 * best_step)

                abs_error = abs(numeric_grad - analytic_grad)
                if abs(analytic_grad) > 1e-10:
                    rel_error = abs_error / abs(analytic_grad)
                else:
                    rel_error = abs_error

                status = "PASS" if rel_error < rel_tol else "FAIL"
                if status == "PASS":
                    passed += 1
                else:
                    failed += 1

                print(f"{idx:<8} {numeric_grad:<12.5e} {analytic_grad:<12.5e} "
                      f"{abs_error:<12.5e} {rel_error:<12.5e} {status:<10}")

            print("-" * 80)
            print(f"Summary: {passed} PASSED, {failed} FAILED out of {min(15, len(test_indices))} tests")

            # Overall assessment
            if best_error < 1e-10:
                print("PERFECT: Gradients match to machine precision (analytical implementation is exact!)")
            elif best_error < 1e-6:
                print("EXCELLENT: Gradients match to machine precision")
            elif best_error < 1e-4:
                print("GOOD: Gradients are accurate")
            elif best_error < 1e-2:
                print("WARNING: Gradients may have some errors")
            else:
                print("POOR: Significant gradient errors detected")

            # Additional robustness tests for perfect gradients
            if best_error < 1e-10:
                print("\n🔬 Running additional robustness tests...")

                # Test gradient at boundary cases
                boundary_tests = []

                # Test near zero values
                vars_test = vars.copy()
                vars_test[test_indices[:5]] = 1e-12
                grad_near_zero = dfunc(vars_test)
                boundary_tests.append(("Near-zero values", np.max(np.abs(grad_near_zero[test_indices[:5]]))))

                # Test with larger values
                vars_test = vars.copy()
                vars_test[test_indices[:5]] *= 1000
                grad_large = dfunc(vars_test)
                boundary_tests.append(("Large values", np.max(np.abs(grad_large[test_indices[:5]]))))

                print("Boundary condition tests:")
                for test_name, max_grad in boundary_tests:
                    if max_grad < 1e20:  # Reasonable gradient magnitude
                        print(f"  {test_name}: Stable (max gradient: {max_grad:.2e})")
                    else:
                        print(f"  {test_name}: Large gradients ({max_grad:.2e}) - check for numerical issues")

                print("Your gradient implementation is exceptionally robust!")

            # Specific locations test (original behavior)
            print(f"\nTesting specific locations (backward compatibility):")
            test_locations = [200, 201, 202, 1000, 5000]
            for test_idx in test_locations:
                if test_idx < n_vars:
                    orig_val = vars[test_idx]
                    vars[test_idx] = orig_val + best_step
                    y2 = func(vars)
                    vars[test_idx] = orig_val - best_step
                    y3 = func(vars)
                    vars[test_idx] = orig_val

                    numeric = (y2 - y3) / (2 * best_step)
                    analytic = grad_analytic[test_idx]
                    rel_err = abs(numeric - analytic) / max(abs(analytic), 1e-10)

                    print(f'Index {test_idx}: numeric={numeric:.6e}, analytic={analytic:.6e}, rel_error={rel_err:.2e}')

            return best_error < rel_tol

        # Run comprehensive derivative test
        gradient_ok = finite_difference_test(vars,
                          func_mrl if error_metric == 'mrl' else func,
                          dfunc_mrl if error_metric == 'mrl' else dfunc,
                          n_tests=25, rel_tol=1e-6)

        if gradient_ok:
            print("Gradient validation PASSED - proceeding with optimization")
        else:
            print("Gradient validation FAILED - check derivative implementation")
            response = input("Continue anyway? (y/n): ").lower()
            if response != 'y':
                print("Stopping optimization due to gradient errors")
                exit(1)

        print("="*60 + "\n")

    # Use L-BFGS-B optimizer with improved settings
    from scipy.optimize import minimize

    print("Tracking optimization progress...")
    _obj_fun = func_mrl if error_metric == 'mrl' else func
    _obj_jac = dfunc_mrl if error_metric == 'mrl' else dfunc

    # Build L-BFGS-B bounds from the support mask.
    # Lower bound 0 on vars everywhere.  In both modes this IS the positivity
    # constraint and is load-bearing: phi is strictly increasing with
    # phi(0) = 0, so o >= 0 if and only if v >= 0.
    # Pixels outside the support get upper bound = 0, forcing them to stay
    # at zero without disrupting the quasi-Newton history for active pixels.
    if SUPPORT_MASK is not None:
        # support repeated for all nslice wavelength planes
        _sm_flat = np.tile(SUPPORT_MASK.flatten(), nslice)
        _bounds = [(0.0, None) if _sm_flat[i] > 0.5 else (0.0, 0.0)
                   for i in range(len(vars))]
        print(f"  Support bounds: {int((_sm_flat > 0.5).sum())} free, "
              f"{int((_sm_flat <= 0.5).sum())} fixed at zero")
    else:
        _bounds = [(0.0, None)] * len(vars)

    if whiten == 2:
        # IRLS outer loop: update inv_var from current model, then minimize
        # with fixed weights. Repeat until iteration budget is exhausted.
        # Each inner call gets niter_inner iterations; the outer loop runs
        # until the total iteration count reaches niter.
        niter_inner  = max(10, niter // 5)   # inner iterations per IRLS step
        niter_remain = niter
        result       = None
        irls_step    = 0
        irls_total_nit  = 0   # total L-BFGS-B iterations across all steps
        irls_total_nfev = 0   # total function evaluations across all steps
        irls_initial_chi2 = None
        while niter_remain > 0:
            irls_step    += 1
            n_this        = min(niter_inner, niter_remain)
            niter_remain -= n_this
            # ---- update inv_var from current model ----
            _cur_obj = phi(vars.reshape(nslice, dim, dim))
            _cur_mod = np.zeros((dim, dim, nimages))
            for _k in range(nslice):
                for _l in range(nimages):
                    _cur_mod[:,:,_l] += np.real(np.fft.ifft2(
                        np.fft.fft2(_cur_obj[_k]) * FTpsf[:,:,_k,_l]))
            inv_var_model[:] = 1.0 / (_sclip(_cur_mod) / gain + read_noise**2)
            if error_metric != 'ls':
                _mrl_calibrate_null(vars, quiet=True)
            print(f"  IRLS step {irls_step}: inv_var updated "
                  f"[{inv_var_model.min():.3e}, {inv_var_model.max():.3e}], "
                  f"{n_this} inner iterations")
            result = minimize(_obj_fun, vars, method='L-BFGS-B', jac=_obj_jac,
                             bounds=_bounds, callback=callback_func,
                             options={
                                 'maxiter': n_this,
                                 'ftol': 1e-15,
                                 'gtol': 1e-12,
                                 'maxfun': n_this * 5,
                                 'maxcor': 15
                             })
            vars = result.x
            irls_total_nit  += result.nit
            irls_total_nfev += result.nfev
            if irls_initial_chi2 is None:
                irls_initial_chi2 = chi_square_history[0] if chi_square_history else result.fun
            if result.success and niter_remain > 0:
                print(f"  IRLS step {irls_step}: converged early, "
                      f"continuing with updated weights")
    else:
        result = minimize(_obj_fun, vars, method='L-BFGS-B', jac=_obj_jac,
                         bounds=_bounds, callback=callback_func,
                         options={
                             'maxiter': niter,
                             'ftol': 1e-15,
                             'gtol': 1e-12,
                             'maxfun': niter * 5,
                             'maxcor': 15
                         })

    optimization_time = time.time() - start_time

    res1 = result.x
    _metric_label = 'chi-square' if error_metric == 'ls' else 'MRL objective'
    if whiten == 2:
        # IRLS: report totals across all steps
        _final_chi2 = chi_square_history[-1] if chi_square_history else result.fun
        _init_chi2  = irls_initial_chi2 or _final_chi2
        print(f"{'Check' if result.success else 'Warning '} IRLS complete ({irls_step} outer steps)")
        print(f" Time: {optimization_time:.2f}s")
        print(f" Final {_metric_label}: {_final_chi2:.6e}")
        print(f" Total iterations: {irls_total_nit}  "
              f"(across {irls_step} IRLS steps of {niter_inner} each)")
        print(f" Total function evaluations: {irls_total_nfev}")
        print(f" Chi-square reduction: {_init_chi2:.6e} -> {_final_chi2:.6e} "
              f"(factor {_init_chi2/_final_chi2:.2f})")
    elif result.success:
        print(f"L-BFGS-B optimization successful!")
        print(f"Time: {optimization_time:.2f}s")
        print(f"Final {_metric_label}: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
    else:
        print(f"L-BFGS-B stopped: {result.message}")
        print(f"Time: {optimization_time:.2f}s")
        print(f"Final {_metric_label}: {result.fun:.6e}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")

    # Create convergence plots
    if len(chi_square_history) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Convert to numpy arrays for easier manipulation
        chi_values = np.array(chi_square_history)
        func_evals = np.array(function_eval_history)

        # Plot 1: Log-Linear (linear x-axis, log y-axis)
        ax1.semilogy(func_evals, chi_values, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
        ax1.set_xlabel('Function Evaluation Number', fontsize=12)
        ax1.set_ylabel('Chi-Square (log scale)', fontsize=12)
        ax1.set_title('Convergence: Log-Linear Scale', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # Add some statistics to the plot
        initial_chi2 = chi_values[0]
        final_chi2 = chi_values[-1]
        reduction_factor = initial_chi2 / final_chi2

        ax1.axhline(y=final_chi2, color='r', linestyle='--', alpha=0.7,
                   label=f'Final chi2 = {final_chi2:.2e}')
        ax1.axhline(y=initial_chi2, color='g', linestyle='--', alpha=0.7,
                   label=f'Initial chi2 = {initial_chi2:.2e}')
        ax1.legend()

        # Plot 2: Log-Log (log x-axis, log y-axis)
        ax2.loglog(func_evals, chi_values, 'r-', linewidth=2, marker='s', markersize=4, alpha=0.7)
        ax2.set_xlabel('Function Evaluation Number (log scale)', fontsize=12)
        ax2.set_ylabel('Chi-Square (log scale)', fontsize=12)
        ax2.set_title('Convergence: Log-Log Scale', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # Add convergence rate information
        if len(chi_values) > 10:
            # Fit a line to the last part of the log-log plot to estimate convergence rate
            log_evals = np.log10(func_evals[-20:])  # Last 20 points
            log_chi = np.log10(chi_values[-20:])

            if len(log_evals) > 5:  # Need enough points for a fit
                slope, intercept = np.polyfit(log_evals, log_chi, 1)
                ax2.text(0.05, 0.95, f'Convergence rate ≈ {slope:.2f}',
                        transform=ax2.transAxes, fontsize=10,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add reduction factor to both plots
        fig.suptitle(f'Optimization Convergence (chi2 reduced by factor of {reduction_factor:.1e})',
                    fontsize=16, y=1.02)

        plt.tight_layout()
        _save_plot('convergence')

        # Print convergence statistics
        _mlabel = 'chi-square' if error_metric == 'ls' else 'MRL objective'
        print(f"\n Convergence Statistics:")
        print(f"   Initial {_mlabel}: {initial_chi2:.6e}")
        print(f"   Final {_mlabel}:   {final_chi2:.6e}")
        print(f"   Reduction factor:   {reduction_factor:.2e}")
        if whiten == 2:
            print(f"   IRLS outer steps:   {irls_step}")
            print(f"   Total L-BFGS-B iterations: {irls_total_nit}")
            print(f"   Total function evaluations: {irls_total_nfev}")
        else:
            print(f"   Total function evaluations: {len(chi_values)}")
        print(f"   Average {_mlabel} per iteration: {np.mean(chi_values):.6e}")

        # Save convergence data if requested
        if fitsout == 1:
            convergence_data = np.column_stack((func_evals, chi_values))
            np.savetxt(os.path.join(_run_dir, 'convergence_history.txt'), convergence_data,
                       header='Function_Evaluation  Chi_Square',
                       fmt='%8d %16.6e')
            print("   Convergence data saved to 'convergence_history.txt'")
        if error_metric == 'mrl':
            _chi2_final = func(res1)
            print(f"   Final chi-square (data fit)   = {_chi2_final:.6e}")
            if '_chi2_initial' in dir():
                print(f"   Chi-square change             = {_chi2_final - _chi2_initial:.6e}"
                      f"  ({100*(_chi2_final-_chi2_initial)/_chi2_initial:+.2f}%)")

    else:
        print(" No convergence data to plot (optimization stopped too early)")

# Enforce support constraint on res1 explicitly after optimizer returns
# (L-BFGS-B may not zero bounded variables to exact machine precision)
if SUPPORT_MASK is not None:
    _sm_flat = np.tile(SUPPORT_MASK.flatten(), nslice)
    res1 = res1.copy()
    res1[_sm_flat <= 0.5] = 0.0
    print(f"Support projection applied to res1: "
          f"flux outside = {float(phi(res1[_sm_flat<=0.5]).sum()):.4e}")

# Reconstruct final object
# res1 is flattened as (nslice, dim, dim); fit is (dim, dim, nslice).
fit = phi(res1.reshape(nslice, dim, dim)).transpose(1, 2, 0).copy()

if SUPPORT_MASK is not None:
    _fit_in  = float(fit[SUPPORT_MASK[None,:,:].repeat(nslice,axis=0).transpose(1,2,0) > 0.5].sum())
    _fit_out = float(fit[SUPPORT_MASK[None,:,:].repeat(nslice,axis=0).transpose(1,2,0) <= 0.5].sum())
    print(f"Post-optimization flux inside support : {_fit_in:.4e}")
    print(f"Post-optimization flux outside support: {_fit_out:.4e}  (should be 0.0)")

for k in range(nslice):
    if display == 1:
        plt.imshow(object[:,:,k])
        plt.title('Truth object @' + str(int(lamb[k])) + ' nm')
        _save_plot('truth_object_k')
        plt.imshow(fit[:,:,k])
        plt.title('Reconstructed object @' + str(int(lamb[k])) + ' nm')
        _save_plot('estimate_k')
        plt.imshow(resid[:,:,k])
        plt.title('Residuals @' + str(int(lamb[k])) + ' nm')
        _save_plot('residuals_k')

    print('volume of truth object @' + str(lamb[k]) + ' nm = ', np.sum(object[:,:,k]))
    print('volume of estimated object @' + str(lamb[k]) + ' nm = ', np.sum(fit[:,:,k]))

# Wavelength-integrated (total) volumes
_vol_truth_total = np.sum(object)
_vol_est_total   = np.sum(fit)
print(f'\nWavelength-integrated total volume:')
print(f'  Truth     : {_vol_truth_total:.6e}')
print(f'  Estimated : {_vol_est_total:.6e}')
print(f'  Ratio     : {_vol_est_total / _vol_truth_total:.4f}  '
      f"({'over' if _vol_est_total > _vol_truth_total else 'under'}-estimated by "
      f'{abs(_vol_est_total/_vol_truth_total - 1)*100:.2f}%)')

# Create results figure for satellite object
fig = plt.figure(figsize=(10, 7))

# Satellite results - plot all non-zero pixels
rows = 8
columns = 6
mn = int(1)

# Find all pixels with non-zero object values
for i in range(dim):
    for j in range(dim):
        # Check if this pixel has any non-zero values across wavelengths
        if np.any(object[i, j, :] > 0):
            if mn > rows * columns:
                break
            
            fig.add_subplot(rows, columns, mn)
            plt.axis('off')
            plt.tick_params(axis='both', left=False, top=False, right=False,
                           bottom=False, labelleft=False, labeltop=False,
                           labelright=False, labelbottom=False)
            plt.plot(np.flip(object[i, j, :]), 'b')
            plt.plot(np.flip(fit[i, j, :]), 'k', linestyle="dashed")
            mn += 1
    if mn > rows * columns:
        break

_save_plot('spectra')

# Calculate RMSE between truth and recovered object at each wavelength
# Support mask for RMSE: use SUPPORT_MASK if available, else truth object footprint
_rmse_mask = (SUPPORT_MASK > 0.5) if (SUPPORT_MASK is not None) else (object.sum(axis=2) > 0)
_nmask = _rmse_mask.sum()
print(f'\nRMSE computed over {_nmask} support-mask pixels (of {dim*dim} total)')

rmse_values      = np.zeros(nslice)   # full-frame (as before)
rmse_mask_values = np.zeros(nslice)   # support-mask only
for k in range(nslice):
    diff = object[:,:,k] - fit[:,:,k]
    rmse_values[k]      = np.sqrt(np.mean(diff**2))
    rmse_mask_values[k] = np.sqrt(np.mean(diff[_rmse_mask]**2))
    truth_vol_k = object[:,:,k][_rmse_mask].sum()
    pct = rmse_mask_values[k] / (truth_vol_k / _nmask) * 100 if truth_vol_k > 0 else 0
    print(f'RMSE @ {lamb[k]:.1f} nm = {rmse_values[k]:.6e} (full-frame)  '
          f'{rmse_mask_values[k]:.6e} (support-masked, {pct:.1f}% of mean truth pixel)')

# Create RMSE vs wavelength plot
plt.figure(figsize=(10, 6))
plt.plot(lamb, rmse_values, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Root Mean Square Error vs Wavelength (Full Frame)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale often better for RMSE values
plt.tight_layout()

# Add some statistics to the plot
mean_rmse = np.mean(rmse_values)
min_rmse = np.min(rmse_values)
max_rmse = np.max(rmse_values)

plt.axhline(y=mean_rmse, color='r', linestyle='--', alpha=0.7, label=f'Mean RMSE = {mean_rmse:.2e}')
plt.legend()

# Print summary statistics
print(f"\nRMSE Statistics:")
print(f"  Mean RMSE: {mean_rmse:.6e}")
print(f"  Min RMSE:  {min_rmse:.6e} @ {lamb[np.argmin(rmse_values)]:.1f} nm")
print(f"  Max RMSE:  {max_rmse:.6e} @ {lamb[np.argmax(rmse_values)]:.1f} nm")
print(f"  Std RMSE:  {np.std(rmse_values):.6e}")

_save_plot('rmse_vs_wavelength')

# Optional: Calculate relative RMSE (normalized by truth object magnitude)
rel_rmse_values = np.zeros(nslice)
for k in range(nslice):
    truth_magnitude = np.sqrt(np.mean(object[:,:,k]**2))
    if truth_magnitude > 0:
        rel_rmse_values[k] = rmse_values[k] / truth_magnitude
    else:
        rel_rmse_values[k] = 0


# Plot relative RMSE as well
plt.figure(figsize=(10, 6))
plt.plot(lamb, rel_rmse_values * 100, 'go-', linewidth=2, markersize=8)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Relative RMSE (%)', fontsize=12)
plt.title('Relative RMSE vs Wavelength (Normalized by Truth Object Magnitude)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

mean_rel_rmse = np.mean(rel_rmse_values) * 100
plt.axhline(y=mean_rel_rmse, color='r', linestyle='--', alpha=0.7,
           label=f'Mean Relative RMSE = {mean_rel_rmse:.2f}%')
plt.legend()

print(f"\nRelative RMSE Statistics:")
print(f"  Mean Relative RMSE: {np.mean(rel_rmse_values)*100:.2f}%")
print(f"  Min Relative RMSE:  {np.min(rel_rmse_values)*100:.2f}% @ {lamb[np.argmin(rel_rmse_values)]:.1f} nm")
print(f"  Max Relative RMSE:  {np.max(rel_rmse_values)*100:.2f}% @ {lamb[np.argmax(rel_rmse_values)]:.1f} nm")

_save_plot('relative_rmse')

# --- Strehl vs wavelength + contrast curve -----------------------------------
# Contrast C(k) = delta-function flux / satellite flux at wavelength k,
# computed from the truth object using the support mask to separate the two.
# The support mask covers the satellite body; pixels outside it are the delta
# function (unresolved companion).
# Only meaningful when do_matched_filter=1 (support mask and delta function
# both present). Falls back gracefully if no support mask is available.
if SUPPORT_MASK is not None:
    _sat_mask  = SUPPORT_MASK > 0.5          # (dim, dim) True = satellite body
    _delta_mask = ~_sat_mask                 # True = delta function / companion

    contrast_curve  = np.zeros(nslice)
    delta_flux_k    = np.zeros(nslice)
    sat_flux_k      = np.zeros(nslice)
    for k in range(nslice):
        delta_flux_k[k] = object[:, :, k][_delta_mask].sum()
        sat_flux_k[k]   = object[:, :, k][_sat_mask].sum()
        if sat_flux_k[k] > 0:
            contrast_curve[k] = delta_flux_k[k] / sat_flux_k[k]

    # Combined detection metric: Strehl x Contrast
    detection_metric = strehl_mean * contrast_curve

    print("\nPer-slice contrast (delta flux / satellite flux) and Strehl:")
    print(f"  {'Slice':>5}  {'lambda(nm)':>10}  {'Delta flux':>12}  "
          f"{'Sat flux':>12}  {'Contrast':>10}  {'Strehl':>8}  {'S x C':>10}")
    for k in range(nslice):
        print(f"  {k:5d}  {lamb[k]:10.1f}  {delta_flux_k[k]:12.4e}  "
              f"{sat_flux_k[k]:12.4e}  {contrast_curve[k]:10.4e}  "
              f"{strehl_mean[k]:8.4f}  {detection_metric[k]:10.4e}")

    # Plot: dual y-axis, Strehl on left, contrast on right, S x C as third curve
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Strehl (left axis)
    color_strehl = 'steelblue'
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Mean Strehl ratio', fontsize=12, color=color_strehl)
    l1 = ax1.plot(lamb, strehl_mean, 'o-', color=color_strehl,
                  linewidth=2, markersize=8, label='Mean Strehl')
    ax1.fill_between(lamb,
                     strehl_mean - strehl_std,
                     strehl_mean + strehl_std,
                     alpha=0.2, color=color_strehl, label='±1σ Strehl')
    ax1.tick_params(axis='y', labelcolor=color_strehl)
    ax1.set_ylim(bottom=0)

    # Contrast (right axis)
    ax2 = ax1.twinx()
    color_contrast = 'darkorange'
    ax2.set_ylabel('Contrast  (δ flux / satellite flux)', fontsize=12,
                   color=color_contrast)
    l2 = ax2.plot(lamb, contrast_curve, 's--', color=color_contrast,
                  linewidth=2, markersize=8, label='Contrast C(λ)')
    ax2.tick_params(axis='y', labelcolor=color_contrast)
    ax2.set_ylim(bottom=0)

    # Strehl x Contrast (right axis, same scale normalised to max=1)
    _sc_norm = detection_metric / detection_metric.max() * contrast_curve.max()
    l3 = ax2.plot(lamb, _sc_norm, '^:', color='green',
                  linewidth=2, markersize=8,
                  label='Strehl X Contrast (normalised)')

    # Combined legend
    lines  = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left')

    plt.title('Mean Strehl ratio and per-slice contrast vs wavelength\n'
              '(Strehl X Contrast = effective per-slice detection metric)',
              fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_plot('strehl_vs_wavelength')
    print("Strehl + contrast plot saved.")

else:
    # No support mask: plot Strehl only
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Mean Strehl ratio', fontsize=12)
    ax1.plot(lamb, strehl_mean, 'o-', color='steelblue',
             linewidth=2, markersize=8, label='Mean Strehl')
    ax1.fill_between(lamb,
                     strehl_mean - strehl_std,
                     strehl_mean + strehl_std,
                     alpha=0.2, color='steelblue', label='±1σ Strehl')
    ax1.set_ylim(bottom=0)
    ax1.legend(fontsize=10)
    plt.title('Mean Strehl ratio vs wavelength', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _save_plot('strehl_vs_wavelength')
    print("Strehl plot saved (no support mask: contrast curve not available).")


# =============================================================================
# COMPANION SPECTRAL EXTRACTION
# Joint linear regression in the whitened residual space.
# Requires: restart=1, niter=0, support_mask_file != '', companion_spectrum=1
# =============================================================================
if companion_spectrum == 1:
    print("\n" + "="*60)
    print("COMPANION SPECTRAL EXTRACTION")
    print("="*60)

    # ---- Prerequisites check ------------------------------------------------
    _cs_ok = True
    if restart != 1:
        print("  ERROR: companion_spectrum requires restart=1 (converged estimate needed)")
        _cs_ok = False
    if niter != 0:
        print("  ERROR: companion_spectrum requires niter=0 (no further optimization)")
        _cs_ok = False
    if SUPPORT_MASK is None:
        print("  ERROR: companion_spectrum requires support_mask_file to be set")
        _cs_ok = False

    if _cs_ok:
        # ---- Convert DS9 coordinates to internal (row, col) -----------------
        # DS9: 1-based, y-axis flipped relative to internal numpy convention
        # internal row = (dim-1) - (ds9_y - 1)
        # internal col = ds9_x - 1
        _cs_row = (dim - 1) - (companion_ds9_y - 1)
        _cs_col = companion_ds9_x - 1
        print(f"  Companion position: DS9 (x,y)=({companion_ds9_x},{companion_ds9_y}) "
              f"-> internal (row,col)=({_cs_row},{_cs_col})")
        print(f"  Aperture radius: {companion_ap_rad} pixels")

        # ---- Build aperture mask around companion position ------------------
        # Exclude the satellite to avoid contamination from its model residuals.
        # Two choices of "the satellite", see companion_sat_excl_frac:
        #   frac = 0 : the object support itself (where the flux is)
        #   frac > 0 : the satellite MODEL footprint (where its residuals are),
        #              i.e. the support convolved with the frame-averaged PSF,
        #              thresholded at that fraction of its peak.
        # The same mask is applied to both b and A so the system stays
        # self-consistent.
        _yy, _xx = np.indices((dim, dim))
        _in_rad  = ((_yy - _cs_row)**2 + (_xx - _cs_col)**2) <= companion_ap_rad**2
        if companion_sat_excl_frac > 0.0:
            _hbar = np.zeros((dim, dim), dtype=complex)
            for _l in range(nimages):
                _hbar += FTpsf[:, :, :, _l].mean(axis=2)
            _hbar /= nimages
            _foot = np.real(np.fft.ifft2(np.fft.fft2(SUPPORT_MASK) * _hbar))
            _foot = np.maximum(_foot, 0.0)
            _sat_support = _foot > companion_sat_excl_frac * _foot.max()
            print(f"  Satellite exclusion: model footprint > "
                  f"{companion_sat_excl_frac:.3g} x peak "
                  f"({int(_sat_support.sum())} px; object support is "
                  f"{int((SUPPORT_MASK > 0.5).sum())} px)")
        else:
            _sat_support = (SUPPORT_MASK > 0.5)
            print(f"  Satellite exclusion: object support only "
                  f"({int(_sat_support.sum())} px) -- note that satellite model "
                  f"residuals extend over the PSF footprint, which is larger; "
                  f"see companion_sat_excl_frac")
        _ap_mask = _in_rad & ~_sat_support
        _n_ap    = int(_ap_mask.sum())
        _n_excl  = int((_sat_support & _in_rad).sum())
        print(f"  Aperture contains {_n_ap} pixels "
              f"({_n_excl} satellite pixels excluded of "
              f"{int(_in_rad.sum())} within the radius)")
        if _n_ap < 200:
            print(f"  WARNING: only {_n_ap} aperture pixels remain; the "
                  f"exclusion may be removing the companion templates. Check "
                  f"the per-slice SNRs and the sum(T^2) diagnostic below.")

        # ---- Build masked satellite model (Option B: mask after sim loop) ---
        # Zero vars outside support, then recompute model from masked estimate.
        # If cs_use_truth=1, use the truth object instead of the reconstruction
        # to test the regression code in isolation from reconstruction errors.
        if cs_use_truth == 1:
            print("  Using TRUTH object as satellite model (cs_use_truth=1)")
            print("  Using in-memory object array (exact object used to generate data).")
            print("  Support mask zeros delta function, leaving only satellite in model.")
            # Use the in-memory `object` (dim,dim,nslice) that generated the data
            # in THIS run -- guarantees model matches data exactly (same calibration).
            _fit_masked = np.zeros((nslice, dim, dim))
            for _k in range(nslice):
                _fit_masked[_k] = object[:, :, _k].copy()
                _fit_masked[_k][SUPPORT_MASK <= 0.5] = 0.0
            print(f"  In-memory object used (no disk load).")
        else:
            # Use reconstructed estimate with support mask applied
            _vars_masked = vars.copy()
            _sm_flat_cs  = np.tile(SUPPORT_MASK.flatten(), nslice)
            _vars_masked[_sm_flat_cs <= 0.5] = 0.0
            _fit_masked  = phi(_vars_masked.reshape(nslice, dim, dim))  # (nslice,dim,dim)

        # Recompute satellite model for all frames using masked estimate
        print("  Recomputing satellite model from support-masked estimate...")
        _cs_model = np.zeros((dim, dim, nimages))
        for _l in range(nimages):
            for _k in range(nslice):
                _cs_model[:, :, _l] += np.real(np.fft.ifft2(
                    np.fft.fft2(_fit_masked[_k]) * FTpsf[:, :, _k, _l]))

        # ---- Compute residuals ----------------------------------------------
        # r_l(x) = data_l(x) - M_hat_l(x)
        # Contains companion signal from all slices + noise
        _cs_resid = data - _cs_model    # (dim, dim, nimages)

        # ---- Noise variance -------------------------------------------------
        # CRITICAL: use MODEL-based variance (whiten=2 style), NOT data-based.
        # Data-based weighting (sigma^2 = data/gain + read^2) correlates the
        # inverse-variance weights with the residual noise: when data fluctuates
        # low, r_l is negative AND inv_s2 is high, so negative fluctuations are
        # up-weighted -- producing a systematic negative bias in b. Using the
        # noise-free satellite model for the variance breaks this correlation.
        _cs_sigma2 = np.maximum(_cs_model, 0.0) / gain + read_noise**2  # (dim,dim,nimages)

        # ---- Build PSF templates at companion position ----------------------
        # T_kl(x) = h_kl(x - x0): PSF at slice k frame l centred at companion.
        # Built via the FFT shift theorem. The data was generated as
        # ifft2(FTobject * FTpsf), and a delta at (r0,c0) contributes
        # FTobject = exp(-i2pi(r0*fy + c0*fx)), so the template must carry the
        # same sign convention. This places the PSF AT the companion; no
        # mirroring is involved (the "flip" of a convolution never enters --
        # the matched filter is sum_x d(x) h(x - x0), evaluated at one x0).
        _fy = np.fft.fftfreq(dim)[:, None]
        _fx = np.fft.fftfreq(dim)[None, :]
        _shift_kernel = np.exp(-1j * 2 * np.pi * (_cs_row * _fy + _cs_col * _fx))

        # ---- Nuisance columns ------------------------------------------------
        # The satellite subtraction is imperfect, so the residual carries a
        # local pedestal (and possibly a gradient) under the aperture. PSFs
        # have positive sum, so a pedestal p couples into b_k with weight
        # p * sum(T_k / sigma^2) -- a weight that GROWS with aperture area.
        # Left unmodelled it biases every alpha_k in the same direction and
        # is invisible in the output.
        #
        # Fix: append frame-independent nuisance templates to the design
        # matrix and solve for them jointly. Marginalisation is then automatic
        # -- the top-left nslice block of inv(A) is the covariance with the
        # nuisance directions projected out, so alpha_sigma correctly inflates
        # to reflect what the pedestal costs. Columns are scaled to unit
        # weighted RMS purely for conditioning; that is a reparametrisation
        # and does not change alpha.
        #
        #   cs_nuisance = 'none'   : no nuisance terms (previous behaviour)
        #                 'const'  : local pedestal
        #                 'linear' : pedestal + x and y gradients
        _nuis_names = []
        if cs_nuisance == 'const':
            _nuis_names = ['pedestal']
        elif cs_nuisance == 'linear':
            _nuis_names = ['pedestal', 'grad_y', 'grad_x']
        elif cs_nuisance != 'none':
            print(f"  WARNING: unknown cs_nuisance='{cs_nuisance}', using 'none'")
        _n_nuis = len(_nuis_names)
        _ntot   = nslice + _n_nuis
        print(f"  Nuisance model: {cs_nuisance} "
              f"({_n_nuis} extra column(s): {', '.join(_nuis_names) or 'none'})")

        _ap_rows, _ap_cols = np.where(_ap_mask)
        _N_ap = np.zeros((_n_nuis, _n_ap))
        if _n_nuis >= 1:
            _N_ap[0] = 1.0
        if _n_nuis == 3:
            _norm = max(float(companion_ap_rad), 1.0)
            _N_ap[1] = (_ap_rows - _cs_row) / _norm
            _N_ap[2] = (_ap_cols - _cs_col) / _norm

        # ---- Accumulate A and b, streaming over frames ----------------------
        # A_ab = sum_{l,x in ap} T_al(x) T_bl(x) / sigma_l^2(x)
        # b_a  = sum_{l,x in ap} r_l(x)  T_al(x) / sigma_l^2(x)
        # Templates are built and discarded frame by frame: storing the full
        # (nslice, nimages, n_ap) cube would be ~2.6 GB at nimages=400 and
        # companion_ap_rad=160.
        print(f"  Accumulating {_ntot}x{_ntot} normal equations over "
              f"{nimages} frame(s) x {nslice} slice(s)...")
        _A = np.zeros((_ntot, _ntot))
        _b = np.zeros(_ntot)
        _t2_prof = np.zeros(int(companion_ap_rad) + 2)   # sum T^2 vs radius
        _ap_rad_pix = np.hypot(_ap_rows - _cs_row, _ap_cols - _cs_col)
        _ap_rbin    = np.minimum(_ap_rad_pix.astype(int), len(_t2_prof) - 1)

        for _l in range(nimages):
            _T = np.zeros((_ntot, _n_ap))
            for _k in range(nslice):
                _psf_shifted = np.real(np.fft.ifft2(
                    FTpsf[:, :, _k, _l] * _shift_kernel))
                _T[_k] = _psf_shifted[_ap_mask]
            if _n_nuis:
                _T[nslice:] = _N_ap

            _w  = 1.0 / np.maximum(_cs_sigma2[:, :, _l][_ap_mask], 1e-30)
            _Tw = _T * _w                       # (ntot, n_ap)
            _A += _Tw @ _T.T
            _b += _Tw @ _cs_resid[:, :, _l][_ap_mask]

            np.add.at(_t2_prof, _ap_rbin, (_T[:nslice] ** 2).sum(axis=0))

        # ---- Aperture size diagnostic ---------------------------------------
        # How much of the signal actually lives inside the aperture? Pixels
        # beyond the radius enclosing ~99% of sum T^2 add essentially no
        # information but do add exposure to subtraction systematics.
        _cum = np.cumsum(_t2_prof)
        if _cum[-1] > 0:
            _cum /= _cum[-1]
            _r90  = int(np.searchsorted(_cum, 0.90))
            _r99  = int(np.searchsorted(_cum, 0.99))
            _r999 = int(np.searchsorted(_cum, 0.999))
            print(f"  Aperture diagnostic: 90/99/99.9% of sum(T^2) lies within "
                  f"{_r90}/{_r99}/{_r999} px of the companion "
                  f"(aperture radius is {companion_ap_rad} px)")
            if companion_ap_rad > 3 * max(_r99, 1):
                print(f"  NOTE: aperture is much larger than the signal "
                      f"footprint. With the nuisance model this is safe but "
                      f"wasteful; consider companion_ap_rad ~ {3*max(_r99,1)}.")

        # ---- Rescale nuisance columns for conditioning ----------------------
        # Pure reparametrisation: alpha for the physical slices is unchanged.
        _scale = np.sqrt(np.maximum(np.diag(_A), 1e-300))
        _Anorm = _A / np.outer(_scale, _scale)

        # ---- Condition number check -----------------------------------------
        # Reported on the scale-normalised (correlation-like) matrix, which is
        # the meaningful quantity once columns with very different units are
        # mixed together.
        _cond = np.linalg.cond(_Anorm)
        print(f"  Condition number of A (scale-normalised): {_cond:.3e}")
        if _cond > 1e6:
            print(f"  WARNING: A is ill-conditioned (cond={_cond:.2e}). "
                  f"Spectral cross-talk is severe. Consider Tikhonov "
                  f"regularisation, or fewer/coarser wavelength slices.")
        elif _cond > 1e3:
            print(f"  CAUTION: A is moderately ill-conditioned "
                  f"(cond={_cond:.2e}). Spectral estimates may have "
                  f"significant cross-talk.")

        # ---- Solve A theta = b ----------------------------------------------
        try:
            _Cnorm  = np.linalg.inv(_Anorm)
            _Cfull  = _Cnorm / np.outer(_scale, _scale)    # full covariance
            _theta  = _Cfull @ _b
            _solve_ok = True
        except np.linalg.LinAlgError as _e:
            print(f"  WARNING: exact solve failed ({_e}); falling back to pinv")
            try:
                _Cfull = np.linalg.pinv(_A)
                _theta = _Cfull @ _b
                _solve_ok = True
            except Exception as _e2:
                print(f"  ERROR: linear solve failed ({_e2}). A may be singular.")
                _solve_ok = False

        if _solve_ok:
            # Physical amplitudes and their MARGINALISED covariance: the
            # top-left nslice block of the full inverse already has the
            # nuisance directions projected out.
            _alpha       = _theta[:nslice]
            _A_inv       = _Cfull[:nslice, :nslice]
            _alpha_sigma = np.sqrt(np.maximum(np.diag(_A_inv), 0.0))
            _snr_k       = np.where(_alpha_sigma > 0,
                                    _alpha / _alpha_sigma, 0.0)
            _nuis_vals   = _theta[nslice:]
            _nuis_sigma  = np.sqrt(np.maximum(
                np.diag(_Cfull)[nslice:], 0.0)) if _n_nuis else np.array([])

            if _n_nuis:
                print(f"\n  Fitted nuisance terms (marginalised out of alpha):")
                for _n, _nm in enumerate(_nuis_names):
                    _nsnr = (_nuis_vals[_n] / _nuis_sigma[_n]
                             if _nuis_sigma[_n] > 0 else 0.0)
                    print(f"    {_nm:10s} = {_nuis_vals[_n]:12.4e} "
                          f"+/- {_nuis_sigma[_n]:11.4e}  ({_nsnr:6.2f} sigma)")
                if abs(_nuis_vals[0]) > 2 * _nuis_sigma[0]:
                    print(f"    -> pedestal is significant; without this term "
                          f"it would have leaked into every alpha_k.")
            # ---- Report results ---------------------------------------------
            print(f"\n  {'Slice':>5}  {'lambda(nm)':>10}  {'alpha_k':>12}  "
                  f"{'sigma_k':>12}  {'SNR_k':>8}  {'Status':>12}")
            print(f"  {'-'*65}")
            for _k in range(nslice):
                _status = ('DETECTED' if abs(_snr_k[_k]) >= 3
                           else 'upper limit' if _snr_k[_k] >= 0
                           else 'negative')
                print(f"  {_k:5d}  {lamb[_k]:10.1f}  {_alpha[_k]:12.4e}  "
                      f"{_alpha_sigma[_k]:12.4e}  {_snr_k[_k]:8.2f}  {_status:>12}")

            # Truth comparison if object is available
            if np.any(object > 0):
                print(f"\n  Truth companion flux per slice (from object outside support):")
                _outside = SUPPORT_MASK <= 0.5
                for _k in range(nslice):
                    _truth_k = float(object[:, :, _k][_outside].sum())
                    _ratio   = _alpha[_k] / _truth_k if _truth_k > 0 else float('nan')
                    print(f"    slice {_k:2d}  lambda={lamb[_k]:7.1f}nm  "
                          f"truth={_truth_k:.4e}  recovered={_alpha[_k]:.4e}  "
                          f"ratio={_ratio:.3f}")

            # ---- Save spectrum to text file ----------------------------------
            _cs_outpath = os.path.join(_run_dir, 'companion_spectrum.txt')
            _cs_lines   = [
                f"# Companion spectral extraction — joint linear regression",
                f"# DS9 position: (x,y) = ({companion_ds9_x},{companion_ds9_y})",
                f"# Internal position: (row,col) = ({_cs_row},{_cs_col})",
                f"# Aperture radius: {companion_ap_rad} px  ({_n_ap} pixels)",
                f"# Condition number of A: {_cond:.3e}",
                f"# Slice  lambda(nm)      alpha_k       sigma_k      SNR_k",
            ]
            for _k in range(nslice):
                _cs_lines.append(
                    f"  {_k:5d}  {lamb[_k]:10.3f}  "
                    f"{_alpha[_k]:12.4e}  {_alpha_sigma[_k]:12.4e}  "
                    f"{_snr_k[_k]:10.4f}")
            with open(_cs_outpath, 'w') as _f:
                _f.write('\n'.join(_cs_lines) + '\n')
            print(f"\n  Spectrum saved to {_cs_outpath}")

            # Save covariance matrix A_inv
            _cs_covpath = os.path.join(_run_dir, 'companion_spectrum_covariance.txt')
            np.savetxt(_cs_covpath, _A_inv,
                       header=f'A_inverse covariance matrix ({nslice}x{nslice}), '
                              f'rows/cols ordered by slice index 0..{nslice-1}',
                       fmt='%14.6e')
            print(f"  Covariance matrix saved to {_cs_covpath}")

            # ---- Plot: flux spectrum + SNR -----------------------------------
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            # Flux spectrum
            ax1.errorbar(lamb, _alpha, yerr=_alpha_sigma,
                         fmt='o-', color='steelblue', linewidth=2,
                         markersize=8, capsize=4, label='Recovered flux')
            if np.any(object > 0):
                _truth_spec = np.array([
                    float(object[:, :, _k][_outside].sum()) for _k in range(nslice)])
                ax1.plot(lamb, _truth_spec, 's--', color='darkorange',
                         linewidth=2, markersize=8, label='Truth flux')
            ax1.axhline(0, color='grey', linewidth=0.8, linestyle=':')
            ax1.set_ylabel('Companion flux (e⁻)', fontsize=12)
            ax1.set_title('Companion spectrum: joint regression in whitened residual space',
                          fontsize=12)
            ax1.legend(fontsize=10)
            ax1.grid(alpha=0.3)

            # SNR per slice
            _colours = ['green' if abs(_s) >= 3 else 'grey' for _s in _snr_k]
            ax2.bar(lamb, _snr_k, width=(lamb[1]-lamb[0])*0.6 if nslice > 1 else 10,
                    color=_colours, alpha=0.7, label='SNR per slice')
            ax2.axhline(3,  color='orange', linewidth=1.5, linestyle='--',
                        label='3σ threshold')
            ax2.axhline(-3, color='orange', linewidth=1.5, linestyle='--')
            ax2.axhline(0,  color='grey',   linewidth=0.8, linestyle=':')
            ax2.set_xlabel('Wavelength (nm)', fontsize=12)
            ax2.set_ylabel('Detection SNR', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)

            plt.tight_layout()
            _save_plot('companion_spectrum')
            print(f"  Spectrum plot saved.")
            print("="*60)

# Output FITS files. Need to flip the y-axis because MATLAB has (0,0) in
# top left of image, whereas FITS has (0,0) in bottom left.
if fitsout == 1:
    print("Saving FITS files with correct format and wavelength ordering...")

    # CRITICAL: Only apply fftshift to spatial dimensions, not wavelength
    psf0 = np.fft.fftshift(psf, axes=(0, 1))  # Only shift x,y axes

    # Now handle the axis swapping and flipping correctly:
    # Original shape: (dim, dim, nslice) = (y, x, wavelength)
    # Target FITS shape: (nslice, dim, dim) = (wavelength, x, y)

    # Step 1: Swap axes to get (nslice, dim, dim) = (wavelength, y, x)
    object_swapped = object.swapaxes(0, 2).swapaxes(1,2)      # (nslice, dim, dim)
    fit_swapped = fit.swapaxes(0, 2).swapaxes(1,2)            # (nslice, dim, dim)
    psf0_swapped = psf0.swapaxes(0, 2).swapaxes(1,2)          # (nslice, dim, dim)
    data_swapped = data.swapaxes(0, 2).swapaxes(1,2)          # (nimages, dim, dim)

    # Step 2: Now flip the y-axis (which is now axis 1 after swapping)
    object_final = np.flip(object_swapped, axis=1)  # Flip y-axis
    fit_final = np.flip(fit_swapped, axis=1)        # Flip y-axis
    psf0_final = np.flip(psf0_swapped, axis=1)      # Flip y-axis
    data_final = np.flip(data_swapped, axis=1)      # Flip y-axis

    # Save with correct FITS format
    fits.writeto(os.path.join(_run_dir,'object.fits'), object_final, overwrite=True)
    fits.writeto(os.path.join(_run_dir,'estimate.fits'), fit_final, overwrite=True)
    fits.writeto(os.path.join(_run_dir,'data.fits'), data_final, overwrite=True)
    fits.writeto(os.path.join(_run_dir,'psf.fits'), psf0_final, overwrite=True)

    # wavelength-averaged unit-volume PSF per frame for the matched filter
    # shape: (nimages, dim, dim), corner-centred (FFT convention)
    _psf_mean = np.zeros((nimages, dim, dim), dtype=np.float32)
    for _l in range(nimages):
        _ps = np.zeros((dim, dim))
        for _k in range(nslice):
            _ps += np.real(np.fft.ifft2(FTpsf[:, :, _k, _l]))
        _ps /= nslice
        _vol = _ps.sum()
        _psf_mean[_l] = (_ps / _vol if _vol > 0 else _ps).astype(np.float32)
    fits.writeto(os.path.join(_run_dir,'psf_mean.fits'),
                 np.flip(np.fft.fftshift(_psf_mean, axes=(1,2)), axis=1),
                 overwrite=True)

    print("FITS files saved with format:")
    print(f"  object.fits: {object_final.shape} = (wavelength, x, y)")
    print(f"  estimate.fits: {fit_final.shape} = (wavelength, x, y)")
    print(f"  psf.fits: {psf0_final.shape} = (wavelength, x, y)")
    print(f"  data.fits: {data_final.shape} = (image, x, y)")
    print(f"  psf_mean.fits: {_psf_mean.shape} = (frame, x, y) centred, unit volume")

    # Also save RMSE data
    rmse_data = np.column_stack((lamb, rmse_values, rmse_mask_values, rel_rmse_values))
    np.savetxt(os.path.join(_run_dir, 'rmse_vs_wavelength.txt'), rmse_data,
               header='Wavelength(nm)  RMSE_fullframe  RMSE_support_masked  Relative_RMSE',
               fmt='%10.3f %12.6e %12.6e %12.6f')
    print("RMSE data saved to 'rmse_vs_wavelength.txt'")

# --- Post-convergence residuals -------------------------------------------
# Reconstruct the model from the converged object estimate
fit_model = np.zeros((dim, dim, nimages))
fit_obj = fit   # (dim, dim, nslice), positivity already enforced in fit[][]
for l in range(nimages):
    for k in range(nslice):
        fit_model[:, :, l] += np.real(np.fft.ifft2(
            np.fft.fft2(fit_obj[:, :, k]) * FTpsf[:, :, k, l]))

# Whitened residual r = (data - model) / sigma
# Use the same sigma law as the metric that was run.
if whiten == 2:
    _base = fit_model
elif whiten == 1:
    _base = data
else:
    _base = np.ones_like(data)   # sigma=1 for unweighted case

_sigma = np.sqrt(_sclip(_base) / gain + read_noise ** 2)
r_white = (data - fit_model) / _sigma          # (dim, dim, nimages)

# Save per-frame residuals to FITS: (nimages, dim, dim), one slice per frame.
# Unwhitened (data - model) and whitened r are both saved so the viewer can
# choose. Header records the whiten setting and noise parameters.
if fitsout == 1:
    from astropy.io import fits as _fits
    hdr = _fits.Header()
    hdr['WHITEN']  = (whiten,     'whitening mode: 0=none 1=data 2=model')
    hdr['RDNOISE'] = (read_noise, 'read noise [e-]')
    hdr['GAIN']    = (gain,       'gain [e-/count]')
    hdr['COMMENT'] = 'Axis order: (frame, y, x)'

    raw_resid = np.flip(np.moveaxis(data - fit_model, 2, 0), axis=1)  # (nimages, dim, dim)
    wht_resid = np.flip(np.moveaxis(r_white,          2, 0), axis=1)  # (nimages, dim, dim)

    _fits.writeto(os.path.join(_run_dir,'residuals_raw.fits'), raw_resid.astype(np.float32), hdr,
                  overwrite=True)
    _fits.writeto(os.path.join(_run_dir,'residuals_whitened.fits'), wht_resid.astype(np.float32), hdr,
                  overwrite=True)
    print("  Residual FITS saved: residuals_raw.fits, residuals_whitened.fits"
          f"  shape {raw_resid.shape}")


print("Simulation complete!")

# --- Matched-filter point-source detection -----------------------------------
if do_matched_filter == 1:
    print("\n" + "="*60)
    print("MATCHED-FILTER POINT-SOURCE DETECTION")
    print("="*60)

    # ---- load estimate ----
    if mf_use_truth == 1:
        _est_path = mf_estimate_file if mf_estimate_file else os.path.join(_run_dir, 'object.fits')
        print(f"  Using TRUTH object from {_est_path} "
              f"(delta function zeroed by support mask)")
    else:
        _est_path = mf_estimate_file if mf_estimate_file else os.path.join(_run_dir, 'estimate.fits')
        print(f"  Loading estimate from {_est_path}")
    _est_raw      = fits.getdata(_est_path).astype(float)   # (nslice, dim, dim)
    _est_internal = np.flip(_est_raw, axis=1).swapaxes(1, 2).swapaxes(0, 2)

    # ---- Step 1: mask from support_mask_file ----
    # Use the same support mask as the optimization (display space, same
    # convention as the estimate). No threshold prompt needed.
    if support_mask_file != '':
        _smraw = fits.getdata(support_mask_file).astype(float)
        _mask_display = (np.flip(_smraw, axis=0) > 0.5).astype(float)
        print(f"  Using support mask from {support_mask_file}: "
              f"{int(_mask_display.sum())} / {dim*dim} pixels active")
    else:
        # No support mask: use all nonzero pixels of the estimate
        _est_sum_display = _est_internal.sum(axis=2)
        _mask_display = (_est_sum_display > 0).astype(float)
        print(f"  No support mask: using all nonzero estimate pixels "
              f"({int(_mask_display.sum())} / {dim*dim})")

    # ---- Step 2: apply mask to estimate ----
    masked_est = _est_internal * _mask_display[:, :, None]   # (dim, dim, nslice)
    _flux_frac = masked_est.sum() / (_est_internal.sum() + 1e-30)
    print(f"  Flux captured by mask: {100*_flux_frac:.1f}% of estimate total")

    # ---- V-band contrast: delta function vs satellite ----
    # Always load object.fits for the contrast calculation so we get the
    # true delta function flux regardless of which estimate is used for
    # the model subtraction.
    # - Satellite flux: object * mask, summed over pixels and wavelength,
    #   weighted by the Johnson-V filter.
    # - Delta function flux: object * complement of mask, summed over pixels
    #   and wavelength, weighted by the Johnson-V filter.
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from radiometry import filter_response
        _vband_weights = filter_response('V', lamb, system='johnson')
        print(f"  V-band filter: Johnson-V Gaussian (centre 540 nm, FWHM 88 nm)")
    except ImportError:
        _vband_weights = np.exp(-0.5 * ((lamb - 540.0) / (88.0 / 2.3548)) ** 2)
        print(f"  V-band filter: Gaussian approximation (centre 540 nm, FWHM 88 nm)")

    _obj_raw  = fits.getdata(os.path.join(_run_dir, 'object.fits')).astype(float)   # (nslice, dim, dim)
    _obj_int  = np.flip(_obj_raw, axis=1).swapaxes(1,2).swapaxes(0,2)  # (dim,dim,nslice)
    _outside_mask = (1.0 - _mask_display)   # 1 outside support, 0 inside

    _sat_flux_v   = sum(_vband_weights[k] * (_obj_int[:,:,k] * _mask_display).sum()
                        for k in range(nslice))
    _delta_flux_v = sum(_vband_weights[k] * (_obj_int[:,:,k] * _outside_mask).sum()
                        for k in range(nslice))

    if _sat_flux_v > 0 and _delta_flux_v > 0:
        _contrast_v   = _delta_flux_v / _sat_flux_v
        _total_flux_v = _sat_flux_v + _delta_flux_v
        _dmag_v       = -2.5 * np.log10(_contrast_v)
        _mag_sat_v    = target_mag + 2.5 * np.log10(_total_flux_v / _sat_flux_v)
        _mag_delta_v  = _mag_sat_v + _dmag_v
        print(f"  Delta function flux (V-band) : {_delta_flux_v:.4e}  e-")
        print(f"  Satellite flux (V-band)      : {_sat_flux_v:.4e}  e-")
        print(f"  Contrast (V-band)            : {_contrast_v:.4e}  ({_contrast_v*100:.4f}%)")
        print(f"  Delta magnitude diff (V)     : {_dmag_v:.3f}  mag")
        print(f"  Satellite magnitude (V)      : {_mag_sat_v:.3f}  mag")
        print(f"  Delta function magnitude (V) : {_mag_delta_v:.3f}  mag")
    elif _sat_flux_v > 0:
        print(f"  Delta function flux (V-band) : 0.0  (no flux outside support)")
        print(f"  Satellite flux (V-band)      : {_sat_flux_v:.4e}  e-")
        print(f"  Contrast (V-band)            : 0.0")
    if fitsout == 1:
        # Flip back to FITS display convention so satellite_mask.fits
        # aligns with object.fits and support.fits in ds9
        fits.writeto(os.path.join(_run_dir,'satellite_mask.fits'),
                     np.flip(_mask_display, axis=0).astype(np.float32),
                     overwrite=True)
        _me_save = np.flip(masked_est.swapaxes(0, 2).swapaxes(1, 2), axis=1)
        fits.writeto(os.path.join(_run_dir,'masked_estimate.fits'), _me_save.astype(np.float32),
                     overwrite=True)
        print(f"  Saved satellite_mask.fits  (display space, centred)")
        print(f"  Saved masked_estimate.fits  shape {_me_save.shape}")

    # ---- Steps 3-4: per-wavelength model subtraction and mean OTF ----
    # Model = masked estimate convolved with per-wavelength per-frame PSF,
    # summed over wavelength. Subtracted from data to leave noise + any
    # faint point source signal.
    print("  Computing per-frame residuals (data - masked satellite model)...")
    mf_FTpsf_mean = np.zeros((dim, dim, nimages), dtype=complex)
    mf_residual   = np.zeros((dim, dim, nimages))
    mf_var        = np.zeros((dim, dim, nimages))
    for _l in range(nimages):
        _model_l   = np.zeros((dim, dim))
        _FTpsf_sum = np.zeros((dim, dim), dtype=complex)
        for _k in range(nslice):
            _model_l   += np.real(np.fft.ifft2(
                np.fft.fft2(masked_est[:, :, _k]) * FTpsf[:, :, _k, _l]))
            _FTpsf_sum += FTpsf[:, :, _k, _l]
        mf_residual[:, :, _l] = data[:, :, _l] - _model_l
        # per-pixel noise variance from the model (Poisson + read noise)
        mf_var[:, :, _l] = _sclip(_model_l) / gain + read_noise**2
        _FTmean = _FTpsf_sum / nslice
        _vol    = float(np.real(_FTmean[0, 0]))
        mf_FTpsf_mean[:, :, _l] = _FTmean / _vol if _vol > 0 else _FTmean

    if fitsout == 1:
        _res_save = np.flip(np.moveaxis(mf_residual, 2, 0), axis=1).astype(np.float32)
        fits.writeto(os.path.join(_run_dir,'mf_residual.fits'), _res_save, overwrite=True)
        print(f"  Saved mf_residual.fits  shape {_res_save.shape}")

    # ---- Step 5: plain matched filter and co-add ----
    # Correlate raw residual (data - model, in electrons) with the
    # wavelength-averaged PSF and co-add over frames. The delta function
    # signal builds coherently; noise averages down as 1/sqrt(N).
    print("  Applying matched filter and co-adding frames...")
    mf_coadd_cc = np.zeros((dim, dim))
    for _l in range(nimages):
        mf_coadd_cc += np.real(np.fft.ifft2(
            np.fft.fft2(np.fft.fftshift(mf_residual[:, :, _l])) *
            np.conj(mf_FTpsf_mean[:, :, _l])))
    mf_coadd_cc /= nimages
    mf_coadd = np.fft.fftshift(mf_coadd_cc)

    if fitsout == 1:
        # y-flip (axis=0) so all mf_* files share the object.fits frame convention
        fits.writeto(os.path.join(_run_dir,'mf_coadd.fits'),
                     np.flip(mf_coadd, axis=0).astype(np.float32), overwrite=True)
        print(f"  Saved mf_coadd.fits     shape {mf_coadd.shape}  (centred, electrons)")
        if mf_smooth_sigma > 0:
            from scipy.ndimage import gaussian_filter
            mf_coadd_smooth = gaussian_filter(mf_coadd, sigma=mf_smooth_sigma)
            mf_coadd_hpf    = mf_coadd - mf_coadd_smooth
            fits.writeto(os.path.join(_run_dir,'mf_coadd_smooth.fits'),
                         np.flip(mf_coadd_smooth, axis=0).astype(np.float32), overwrite=True)
            fits.writeto(os.path.join(_run_dir,'mf_coadd_hpf.fits'),
                         np.flip(mf_coadd_hpf, axis=0).astype(np.float32), overwrite=True)
            print(f"  Saved mf_coadd_smooth.fits  (Gaussian sigma={mf_smooth_sigma} px)")
            print(f"  Saved mf_coadd_hpf.fits     (mf_coadd - mf_coadd_smooth)")

            # ---- Step 6b: empirical noise from Gaussian fit to the HPF histogram ----
            # The pixel histogram of mf_coadd_hpf is close to Gaussian (noise core)
            # plus a small non-Gaussian tail from the source and residual systematics.
            # Fit a Gaussian to the histogram core to get a robust noise sigma,
            # then rescale mf_coadd_hpf into SNR units. Peak SNR in this map is a
            # direct statistical-significance statistic (e.g. >5 = detection).
            from scipy.optimize import curve_fit

            _hpf_vals = mf_coadd_hpf.ravel()
            # Initial robust estimate: sigma from MAD (immune to the source tail)
            _med  = np.median(_hpf_vals)
            _mad  = np.median(np.abs(_hpf_vals - _med))
            _sig0 = 1.4826 * _mad if _mad > 0 else _hpf_vals.std()

            # Histogram the central +/-5 sigma region only (excludes source tail)
            _lo, _hi = _med - 5*_sig0, _med + 5*_sig0
            _counts, _edges = np.histogram(_hpf_vals, bins=201, range=(_lo, _hi))
            _centers = 0.5*(_edges[:-1] + _edges[1:])

            def _gauss(x, A, mu, sig):
                return A * np.exp(-0.5*((x - mu)/sig)**2)

            try:
                _popt, _ = curve_fit(_gauss, _centers, _counts,
                                     p0=[_counts.max(), _med, _sig0],
                                     maxfev=10000)
                _hpf_mu, _hpf_sigma = float(_popt[1]), abs(float(_popt[2]))
                _fit_ok = True
            except Exception as _e:
                print(f" WARNING: Gaussian fit failed ({_e}); falling back to MAD sigma")
                _hpf_mu, _hpf_sigma = float(_med), float(_sig0)
                _fit_ok = False

            mf_coadd_hpf_snr = (mf_coadd_hpf - _hpf_mu) / _hpf_sigma
            fits.writeto(os.path.join(_run_dir,'mf_coadd_hpf_snr.fits'),
                         np.flip(mf_coadd_hpf_snr, axis=0).astype(np.float32), overwrite=True)

            _hpf_peak_snr = float(mf_coadd_hpf_snr.max())
            _hpf_peak_ij  = np.unravel_index(np.argmax(mf_coadd_hpf_snr),
                                             mf_coadd_hpf_snr.shape)
            # Convert internal (row, col) to FITS display coordinates:
            # FITS y is flipped relative to internal: fits_y = (dim-1) - internal_y
            _hpf_peak_fits_y = (dim - 1) - _hpf_peak_ij[0]
            _hpf_peak_fits_x = _hpf_peak_ij[1]
            print(f"  Saved mf_coadd_hpf_snr.fits (empirical SNR units)")
            print(f"  HPF histogram Gaussian fit : mu={_hpf_mu:.4e}  sigma={_hpf_sigma:.4e}"
                  f"  ({'curve_fit' if _fit_ok else 'MAD fallback'})")
            print(f"  MAD-based sigma (check)    : {_sig0:.4e}")
            print(f"  Peak SNR (HPF, empirical)  : {_hpf_peak_snr:.2f}"
                  f"  at FITS display (y,x)=({_hpf_peak_fits_y},{_hpf_peak_fits_x})"
                  f"  [internal (y,x)=({_hpf_peak_ij[0]},{_hpf_peak_ij[1]})]")
            _sig_level = ('significant (>5 sigma)' if _hpf_peak_snr > 5
                          else 'marginal (3-5 sigma)' if _hpf_peak_snr > 3
                          else 'not significant (<3 sigma)')
            print(f"  Detection assessment       : {_sig_level}")

    _peak_val  = float(mf_coadd.max())
    print(f"  Matched-filter peak value   : {_peak_val:.4e}  e-")
    print("="*60)


# --- Generate JSON summary and HTML report -----------------------------------
import json, base64, io

# Collect all metrics into a dict
_run_data = {
    "timestamp": _log_timestamp,
    "parameters": {
        "error_metric": error_metric,
        "whiten": whiten,
        "niter": niter,
        "restart": restart,
        "nslice": nslice,
        "nimages": nimages,
        "dim": dim,
        "target_mag": target_mag,
        "mag_band": mag_band,
        "mag_system": mag_system,
        "add_noise": add_noise,
        "read_noise": read_noise,
        "gain": gain,
        "random_seed": random_seed,
        "D_cm": D,
        "t_exp_s": t_exp_s,
        "support_mask_file": support_mask_file,
        "mrl_R": mrl_R,
        "mrl_tmax": mrl_tmax,
        "do_matched_filter": do_matched_filter,
        "mf_use_truth": mf_use_truth,
        "output_files": {
            "object":             'object.fits',
            "estimate":           'estimate.fits',
            "residuals_raw":      'residuals_raw.fits',
            "residuals_whitened": 'residuals_whitened.fits',
        },
        "mf_smooth_sigma": mf_smooth_sigma,
    },
    "reconstruction": {
        "mean_relative_rmse_pct": float(np.mean(rel_rmse_values) * 100),
        "min_relative_rmse_pct":  float(np.min(rel_rmse_values) * 100),
        "max_relative_rmse_pct":  float(np.max(rel_rmse_values) * 100),
        "rmse_per_wavelength": {f"{lamb[k]:.1f}nm": float(rmse_values[k])
                                for k in range(nslice)},
        "volume_truth":    {f"{lamb[k]:.1f}nm": float(object[:,:,k].sum())
                            for k in range(nslice)},
        "volume_estimate": {f"{lamb[k]:.1f}nm": float(fit[:,:,k].sum())
                            for k in range(nslice)},
    },
}

# Add matched filter results if available
if do_matched_filter == 1:
    _run_data["matched_filter"] = {
        "mf_use_truth": mf_use_truth,
        "support_mask_file": support_mask_file,
        "files": {
            "mf_coadd_hpf": 'mf_coadd_hpf.fits',
            "mf_coadd":     'mf_coadd.fits',
        },
        "contrast_vband": float(_contrast_v) if '_contrast_v' in dir() else None,
        "delta_flux_vband": float(_delta_flux_v) if '_delta_flux_v' in dir() else None,
        "satellite_flux_vband": float(_sat_flux_v) if '_sat_flux_v' in dir() else None,
        "delta_mag_diff": float(_dmag_v) if '_dmag_v' in dir() else None,
        "satellite_mag_v": float(_mag_sat_v) if '_mag_sat_v' in dir() else None,
        "delta_mag_v": float(_mag_delta_v) if '_mag_delta_v' in dir() else None,
        "peak_mf_value": float(_peak_val) if '_peak_val' in dir() else None,
    }

# Save JSON
_json_path = os.path.join(_run_dir, 'run.json')
with open(_json_path, 'w') as _jf:
    json.dump(_run_data, _jf, indent=2)
print(f"JSON summary saved: {_json_path}")

# --- HTML report -------------------------------------------------------------
def _fits_to_png_b64(fitsfile, slice_idx=0, cmap='hot', pct_lo=1, pct_hi=99,
                     zoom=1, log_scale=False):
    """Render a FITS file slice as a base64 PNG string.
    zoom=2: crop to bounding box of nonzero pixels + 20px margin.
    log_scale: apply log10(max(d, max*1e-4)) before display.
    """
    try:
        d = fits.getdata(fitsfile).astype(float)
        if d.ndim == 3:
            d = d[slice_idx]
        if zoom > 1:
            # find bounding box of significant pixels
            _thresh = np.percentile(np.abs(d), 90)
            _nz = np.argwhere(np.abs(d) > _thresh)
            if len(_nz) > 0:
                _r0 = max(0, _nz[:,0].min() - 20)
                _r1 = min(d.shape[0], _nz[:,0].max() + 20)
                _c0 = max(0, _nz[:,1].min() - 20)
                _c1 = min(d.shape[1], _nz[:,1].max() + 20)
                d = d[_r0:_r1, _c0:_c1]
        if log_scale:
            _dmax = np.abs(d).max()
            if _dmax > 0:
                d = np.log10(np.maximum(d, _dmax * 1e-4))
        vlo = np.percentile(d, pct_lo)
        vhi = np.percentile(d, pct_hi)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(d, origin='lower', cmap=cmap, vmin=vlo, vmax=vhi)
        _title = os.path.basename(fitsfile)
        ax.set_title(_title, fontsize=7)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        return None

_html_path = os.path.join(_run_dir, 'run.html')
# (filename, slice_idx, cmap, pct_lo, pct_hi, zoom, log_scale)
_imgs = [
    (os.path.join(_run_dir,'object.fits'),            0,    'gray',   1, 99, 2, True),
    (os.path.join(_run_dir,'estimate.fits'),          0,    'gray',   1, 99, 2, True),
    (os.path.join(_run_dir,'residuals_whitened.fits'),0,    'RdBu_r', 1, 99, 1, False),
    (os.path.join(_run_dir,'mf_coadd_hpf.fits'), None, 'RdBu_r', 1, 99, 2, False),
]

_img_html = ''
for _fname, _sl, _cm, _plo, _phi, _zoom, _log in _imgs:
    if os.path.exists(_fname):
        _b64 = _fits_to_png_b64(_fname, _sl if _sl is not None else 0, _cm, _plo, _phi, _zoom, _log)
        if _b64:
            _caption = _fname
            if _zoom > 1: _caption += f' | zoom {_zoom}x'
            if _log:      _caption += ' | log scale'
            _img_html += (f'<div style="display:inline-block;margin:8px;text-align:center">'
                         f'<img src="data:image/png;base64,{_b64}" style="max-width:280px"/>'
                         f'<br/><small>{_caption}</small></div>')

_log_text = open(_log_path).read()

_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Broadband_AI Run {_log_timestamp}</title>
<style>
  body {{font-family:monospace; background:#1a1a2e; color:#e0e0e0; margin:20px}}
  h1 {{color:#00d4ff}} h2 {{color:#7ec8e3; margin-top:30px}}
  table {{border-collapse:collapse; margin:10px 0}}
  td,th {{border:1px solid #444; padding:4px 10px; font-size:12px}}
  th {{background:#2a2a4e; color:#7ec8e3}}
  pre {{background:#0d0d1a; padding:15px; overflow-x:auto; font-size:11px;
        border-left:3px solid #00d4ff; white-space:pre-wrap}}
  .metric {{color:#ffd700; font-weight:bold}}
</style></head><body>
<h1>Broadband_AI Run Report</h1>
<p>Timestamp: <span class="metric">{_log_timestamp}</span> &nbsp;|&nbsp;
   Metric: <span class="metric">{error_metric.upper()}</span> &nbsp;|&nbsp;
   Whiten: <span class="metric">{whiten}</span> &nbsp;|&nbsp;
   Niter: <span class="metric">{niter}</span> &nbsp;|&nbsp;
   Mag: <span class="metric">+{target_mag}</span> {mag_band}</p>

<h2>FITS Images</h2>
<div>{_img_html}</div>

<h2>Run Plots</h2>
<div>{chr(10).join(
    f'<div style="display:inline-block;margin:8px;text-align:center"><img src="data:image/png;base64,'
    + __import__("base64").b64encode(open(_p,"rb").read()).decode()
    + f'" style="max-width:400px"/><br/><small>{_l}</small></div>'
    for _p, _l in _plot_files if __import__("os").path.exists(_p)
)}</div>

<h2>Reconstruction Quality</h2>
<table>
<tr><th>Wavelength (nm)</th><th>RMSE</th><th>Rel RMSE (%)</th>
    <th>Volume Truth</th><th>Volume Estimate</th></tr>
{''.join(f"<tr><td>{lamb[k]:.1f}</td><td>{rmse_values[k]:.3e}</td>"
          f"<td>{rel_rmse_values[k]*100:.2f}</td>"
          f"<td>{object[:,:,k].sum():.3e}</td>"
          f"<td>{fit[:,:,k].sum():.3e}</td></tr>"
          for k in range(nslice))}
<tr><th>Mean</th><th>{np.mean(rmse_values):.3e}</th>
    <th class="metric">{np.mean(rel_rmse_values)*100:.2f}</th>
    <td>—</td><td>—</td></tr>
</table>

{'<h2>Matched Filter</h2><table>' +
 ''.join(f"<tr><th>{k}</th><td>{v}</td></tr>"
          for k,v in _run_data.get('matched_filter',{}).items()) +
 '</table>' if do_matched_filter else ''}

<h2>Full Run Log</h2>
<pre>{_log_text}</pre>
</body></html>"""

with open(_html_path, 'w') as _hf:
    _hf.write(_html)
print(f"HTML report saved: {_html_path}")

sys.stdout = _orig_stdout   # restore stdout
_log_fh.close()
print(f"Run log saved: {_log_path}")
