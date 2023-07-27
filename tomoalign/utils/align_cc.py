# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from skimage.registration import phase_cross_correlation as pcc
from scipy import ndimage


def cor_flipping(proj_0, proj_180):
    """
    Correct center-of-rotation by cross-correlating projections that are 180 degrees apart.
    :param proj_0: Projection at 0 degrees
    :param proj_180: Projection at 180 degrees
    :return: horizontal shift in pixels
    """
    
    # reflect the image at 180 degrees
    proj_180 = np.fliplr(proj_180)
    
    out = pcc(proj_0, proj_180, upsample_factor=16)
    
    return out[0][1]


def cross_correlation_skimage(projections, sinogram_order='True'):
    
    n_proj, nx, nz = projections.shape
    offsets = np.zeros((n_proj, 2))
    aligned_proj = projections.copy()
    
    for i in range(1, n_proj):
        shifts, error, phase_diff = pcc(aligned_proj[i-1], aligned_proj[i], upsample_factor=100)
        offsets[i, :] = shifts
        aligned_proj[i] = ndimage.shift(aligned_proj[i], shifts)
    
    return offsets, aligned_proj


def cross_correlation_numpy(projections):
    
    n_proj, nx, nz = projections.shape
    offsets = np.zeros((n_proj, 2))
    aligned_proj = projections.copy()
    
    # frequency space
    kx = np.fft.fftfreq(nx)
    kz = np.fft.fftfreq(nz)
    [kx, kz] = np.meshgrid(kx, kz)
    abs_k = np.sqrt(kx**2 + kz**2)
    cutoff = 4
    filter_k = (abs_k <= (0.5/cutoff)) * np.sin(2*np.pi*cutoff*abs_k)**2
    
    # real space
    x = np.linspace(1, nx, nx)
    z = np.linspace(1, nz, nz)
    [x, z] = np.meshgrid(x, z)
    filter_r = (np.sin(np.pi * x/nx) * np.sin(np.pi * z/nz))**2
    
    for i in range(1, n_proj):
        
        offsets[i, :], aligned_proj[i] = crossCorrelationAlign(aligned_proj[i], aligned_proj[i-1],
                                                               filter_r, filter_k)

    ind_z = np.where(offsets[:, 0] > nz / 2)
    offsets[ind_z, 0] -= nz
    ind_x = np.where(offsets[:, 1] > nx / 2)
    offsets[ind_x, 1] -= nx
    
    return offsets, aligned_proj
    

def crossCorrelationAlign(image, reference, rFilter, kFilter):
    
    """Align image to reference by cross-correlation"""
    image_f = np.fft.fft2((image - np.mean(image)) * rFilter)
    reference_f = np.fft.fft2((reference - np.mean(reference)) * rFilter)
    xcor = abs(np.fft.ifft2(np.conj(image_f) * reference_f * kFilter))
    shifts = np.unravel_index(xcor.argmax(), xcor.shape)
    
    # shift image
    output_image = np.roll(image, shifts[0], axis=0)
    output_image = np.roll(output_image, shifts[1], axis=1)
    
    return shifts, output_image
