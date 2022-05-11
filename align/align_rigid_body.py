# ------------------------------------------------
# Copyright 2021 Kanupriya Pande
# Contact kpande@lbl.gov
# ------------------------------------------------

import numpy as np
from projectors import projection_operators


class RigidAlignment(object):
    
    def __init__(self, geometry, align_what, comm=None):
        """
        :param geometry: geometry object
        :param align_what: 6-tuple of 0s and 1s giving which parameters to align
        :param comm: MPI communicator
        """
        
        self.geometry = geometry
        self.align_what = align_what
        self.comm = comm
        self.proj_grad_obj = projection_operators.ProjectionGradient(self.geometry, comm=self.comm)
        
        #if self.comm is None:
        #    self.my_proj_index = np.arange(self.nproj)
        #    self.my_nproj = self.nproj
        #else:
        #    self.my_proj_index = np.array_split(np.arange(self.nproj), self.comm.Get_size())[self.comm.Get_rank()]
        #    self.my_nproj = np.size(self.my_proj_index)
    
    def cost(self, recon, projection, angles, xyz, cor_shift):
        
        alpha, beta, phi = angles
        this_proj, _ = self.proj_grad_obj.proj_gradient(recon, alpha, beta, phi, xyz, cor_shift)
        residual = projection.ravel() - this_proj
        
        return residual
    
    def gradient(self, recon, projection, angles, xyz, cor_shift):
        
        alpha, beta, phi = angles
        this_proj, this_proj_grad = self.proj_grad_obj.proj_gradient(recon, alpha, beta, phi, xyz, cor_shift)
        residual = projection.ravel() - this_proj
        this_proj_grad *= -1
        
        return residual, this_proj_grad
    

def cost(parameters, align_obj, recon, projection, angles_start, xyz_start, cor_shift,
         return_vector=False):
    """
    :param parameters: array of parameters being optimized in this order (x, y, z, alpha, beta, phi)
    :param align_obj: alignment object containing geometry
    :param recon: reconstruction
    :param angles_start: starting angles
    :param xyz_start: starting translations
    :param projection: projection to be aligned
    :param return_vector: whether to return a float vector or float number
    :return: cost function as a vector or number
    """
    
    xyz = xyz_start + parameters[:3] * align_obj.align_what[:3]
    angles = angles_start + parameters[3:] * align_obj.align_what[3:]
    
    cost = align_obj.cost(recon, projection, angles, xyz, cor_shift)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost)**2


def gradient(parameters, align_obj, recon, projection, angles_start, xyz_start, cor_shift,
             return_vector=False):
    """
    :param parameters: array of parameters being optimized in this order (x, y, z, alpha, beta, phi)
    :param align_obj: alignment object containing geometry
    :param recon: reconstruction
    :param angles_start: starting angles
    :param xyz_start: starting translations
    :param projection: projection to be aligned
    :param return_vector: whether to return a float vector or float number
    :param cor_shift: list
    :return:
    """
    
    xyz = xyz_start + parameters[:3] * align_obj.align_what[:3]
    angles = angles_start + parameters[3:] * align_obj.align_what[3:]
    
    cost, grad = align_obj.gradient(recon, projection, angles, xyz, cor_shift)
    grad = np.array(align_obj.align_what)[:, np.newaxis] * grad
    
    if return_vector:
        return grad.T
    
    grad = np.dot(grad, cost)
    
    return grad

def cost_xz(parameters, align_obj, rec, projection, angles_in, xyz_in, cor_shift, return_vector=False):

    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])

    cost = align_obj.cost(rec, projection, angles_in, translations, cor_shift)

    if return_vector:
        return cost

    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_xz(parameters, align_obj, rec, projection, angles_in, xyz_in, cor_shift, return_vector=False):

    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])

    residual, s = align_obj.gradient(rec, projection, angles_in, translations, cor_shift)

    vary_parameter = np.array([True, False, True, False, False, False])
    s = s[vary_parameter]

    #if scale_factor is None:
    #    scale_factor = np.array([1.0, 1.0])

    #s = s * scale_factor[:, np.newaxis]

    if return_vector:
        return s.T

    grad = np.dot(s, residual)
    print(translations[0], translations[2], 0.5*np.linalg.norm(residual)**2, grad)
    
    return grad
