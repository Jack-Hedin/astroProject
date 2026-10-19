import numpy as np
import math
from astropy.io import fits
import time
import matplotlib.pyplot as plt

# Constants (only used for binary star mode)
LIGHT_SPEED = 2.997e8  # m/s
PLANCK_CONSTANT = 6.6260693e-34  # J*s
BOLTZMANN_CONSTANT = 1.380658e-23  # J/K

# Binary star temperatures (only used if obj_type = 0)
T1 = 6000  # Temperature of star #1 in Kelvin
T2 = 4000  # Temperature of star #2 in Kelvin

# Telescope parameters
D = 360.0  # Diameter of telescope aperture [cm]
print('Diameter of telescope =', D / 100.0, 'meters')
D1 = 2.0 * D  # Phase screen diameter

# Wavelength parameters
lamb1 = 400.0  # min wavelength [nm]
lamb2 = 1000.0  # max wavelength [nm]

# Simulation parameters
incr = 8
obj_type = 1  # 0 = binary star, else satellite object
CSO_scale = 0.05
display = 1  # Set to 1 to plot generated objects
template_filename = "template.fits"  # Path and filename for satellite template FITS file

# Absorption band parameters for solar panel with absorption lines (material 7)
# Each tuple: (center_nm, fwhm_nm, min_reflectivity_fraction)
# Band 1: 500 nm center, 10 nm FWHM, 15% minimum reflectivity
# Band 2: 900 nm center, 10 nm FWHM, 15% minimum reflectivity
absorption_bands = [
    (500, 10, 0.15),   # (center_nm, fwhm_nm, min_reflectivity)
    (900, 10, 0.15)    # (center_nm, fwhm_nm, min_reflectivity)
]

# Compute simulation parameters
delta_slice = int(incr)

dim = int(2 ** int(np.log10(float(D1 / 2)) / np.log10(2.0) + 1))

print('Number of pixels along one side of phase screen =', dim)

rad_max = int(float(dim) / 4.0)
rad_min = int(rad_max * (lamb1 / lamb2)) + 1
nslice = int((rad_max - rad_min) / delta_slice) + 1
rad_min = rad_max - nslice * delta_slice + delta_slice

print('rad_max', rad_max)
print('rad_min', rad_min)
print('nslice =', nslice)


def gaussian_absorption(wavelength, center, fwhm, min_transmission):
    """
    Create a Gaussian absorption feature.
    
    Parameters:
    -----------
    wavelength : array-like
        Wavelength values [nm]
    center : float
        Center wavelength of absorption [nm]
    fwhm : float
        Full Width at Half Maximum [nm]
    min_transmission : float
        Minimum transmission at center (e.g., 0.05 for 5% transmission)
    
    Returns:
    --------
    transmission : array
        Transmission factor (1.0 = no absorption, 0.0 = complete absorption)
    """
    # Convert FWHM to standard deviation
    # FWHM = 2 * sqrt(2 * ln(2)) * sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Create Gaussian profile centered at 'center'
    # At center: absorption is maximum (transmission is minimum)
    # Far from center: absorption is zero (transmission is 1.0)
    gaussian = np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)
    
    # Scale so that:
    # - At center (gaussian=1): transmission = min_transmission
    # - Far away (gaussian=0): transmission = 1.0
    transmission = 1.0 - (1.0 - min_transmission) * gaussian
    
    return transmission


def compute_satellite_object_ultra_fast(template, lamb, dim, nslice, CSO_scale, absorption_bands):
    """
    Ultra-fast vectorized material property calculation for satellite object.
    Now includes support for material 7: solar panel with absorption lines.
    
    Parameters:
    -----------
    template : array
        Material template defining which material at each pixel
    lamb : array
        Wavelength array [nm]
    dim : int
        Dimension of the array
    nslice : int
        Number of wavelength slices
    CSO_scale : float
        Scaling factor for CSO solar panel
    absorption_bands : list of tuples
        Each tuple: (center_nm, fwhm_nm, min_reflectivity_fraction)
        Applied to material 7 only
    
    Returns:
    --------
    object_array : array
        Object array with material properties at each wavelength
    """
    coefficients = {
        1: np.array([9.667504e2, -4.583580e0, 8.008073e-3, -6.110959e-6, 1.73911e-9]),
        2: np.array([-1.196472E5, 1.460390E3, -7.648047E0, 2.246897E-2, -4.056309E-5,
                      4.615807E-8, -3.238676E-11, 1.283035E-14, -2.200156E-18]),
        3: np.array([-7.964498E3, 7.512566E1, -2.883463E-1, 5.812354E-4,
                     -6.488131E-7, 3.801200E-10, -9.129042E-14]),
        4: np.array([-7.668973E4, 9.501756E2, -5.055507E0, 1.509819E-2, -2.771163E-5,
                     3.204811E-8, -2.283223E-11, 9.171536E-15, -1.591887E-18]),
        5: np.array([-2.223456E4, 2.305905E2, -1.007697E0, 2.404597E-3, -3.379637E-6,
                     2.797231E-9, -1.262900E-12, 2.401227E-16])
    }

    object_array = np.zeros((dim, dim, nslice))

    for k in range(nslice):
        lam = lamb[k]
        values = {}
        for material_id, coeffs in coefficients.items():
            powers = lam ** np.arange(len(coeffs))
            values[material_id] = np.dot(coeffs, powers)
        
        # Create value for material 7 (solar panel with absorption lines)
        # Start with base solar panel spectrum (material 1)
        values[7] = values[1]
        
        # Apply absorption bands to material 7
        for center, fwhm, min_refl in absorption_bands:
            transmission = gaussian_absorption(np.array([lam]), center, fwhm, min_refl)[0]
            values[7] *= transmission

        # Vectorized conditional assignment based on template values
        conditions = [template == i for i in range(1, 8)]
        choices = [
            values[1],  # material 1: solar panel
            values[2],  # material 2: kapton
            values[3],  # material 3: aluminized mylar
            values[4],  # material 4: kapton aged
            values[5],  # material 5: aluminized mylar aged
            CSO_scale * values[1],  # material 6: CSO solar panel
            values[7]   # material 7: solar panel with absorption lines
        ]

        object_array[:, :, k] = np.select(conditions, choices, default=0.0)

    object_array *= 0.01
    return object_array


def generate_binary_star_object(dim, nslice, lamb):
    """Generate binary star object with blackbody behavior."""
    object_array = np.zeros((dim, dim, nslice))
    x1 = dim // 2
    y1 = dim // 2 + 20
    y0 = dim // 2 - 20

    for k in range(nslice):
        bb_factor = 8 * np.pi * PLANCK_CONSTANT * LIGHT_SPEED / (lamb[k] * 1e-9) ** 5
        object_array[y1, x1, k] = math.floor(
            bb_factor / (np.exp(PLANCK_CONSTANT * LIGHT_SPEED / (lamb[k] * 1e-9 * BOLTZMANN_CONSTANT * T1)) - 1))
        object_array[y0, x1, k] = math.floor(
            bb_factor / (np.exp(PLANCK_CONSTANT * LIGHT_SPEED / (lamb[k] * 1e-9 * BOLTZMANN_CONSTANT * T2)) - 1))
    return object_array


# Pre-allocate wavelength array
lamb = np.zeros(nslice)

# Load or create a template for satellite object materials (required if obj_type != 0)
if obj_type != 0:
    template = fits.getdata(template_filename)
else:
    template = None

# Calculate wavelength for each slice
for k in range(nslice):
    rad = rad_min + k * delta_slice
    lamb[k] = lamb1 * (float(dim) / 4.0) / rad
    print('rad =', rad, 'lambda =', lamb[k])

start_time = time.time()

if obj_type == 0:
    # Generate binary star object
    object_array = generate_binary_star_object(dim, nslice, lamb)
else:
    if template is None:
        raise RuntimeError("Template is required for satellite object generation.")
    object_array = compute_satellite_object_ultra_fast(template, lamb, dim, nslice, CSO_scale, absorption_bands)

print(f"Object generation time: {time.time() - start_time:.2f}s")

# Display spectral curves for different materials
if display == 1:
    material_ids = [1, 2, 3, 4, 5, 7]  # Include material 7 in the display
    
    # Find a representative pixel (y,x) for each material type in template
    pixel_coords = []
    for mat in material_ids:
        pos = np.argwhere(template == mat)
        if len(pos) > 0:
            pixel_coords.append(tuple(pos[0]))  # Take first occurrence
        else:
            pixel_coords.append(None)  # Material not found, handle gracefully
    
    # Prepare data for plotting
    for coords in pixel_coords:
        if coords is None:
            continue
        y, x = coords
        mat_id = template[y, x]
        values = object_array[y, x, :]  # Values over all wavelength slices
        plt.plot(lamb, values, label=f'Material {mat_id} at ({x},{y})')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Object Value')
    plt.title('Values vs Wavelength for Material Pixels')
    plt.legend()
    plt.grid(True)
    plt.show()


# After object_array generation, prepare for FITS saving
# Swap axes from (x, y, wavelength) to (wavelength, y, x)
object_fits = np.swapaxes(object_array, 0, 2)
object_fits = np.swapaxes(object_fits, 1, 2)

# Flip y-axis since FITS origin is bottom-left, numpy is top-left
object_fits = np.flip(object_fits, axis=1)

# Save to FITS file
fits_filename = 'multi_wavelength_object.fits'
fits.writeto(fits_filename, object_fits, overwrite=True)
print(f"Multi-wavelength object saved as FITS file: {fits_filename}")
