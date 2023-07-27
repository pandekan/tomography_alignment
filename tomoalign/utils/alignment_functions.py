import numpy as np
from scipy import sparse, optimize
from scipy.optimize import line_search
from scipy.optimize.linesearch import line_search_armijo, line_search_wolfe1


class AlignmentUtilities(object):
    
    def __init__(self, proj, proj_obj, geometry):
        
        self.proj = proj
        self.proj_obj = proj_obj
        self.proj_mask = proj > 0
        self.geometry = geometry
        
    def cost(self, rec, angles, translations):
        
        phi, alpha, beta = angles
        xyz_shift = translations
        this_proj, _ = self.proj_obj.projection_gradient(rec=rec, alpha=alpha, beta=beta, phi=phi, xyz_shift=xyz_shift, 
                                                         cor_shift=self.geometry.cor_shift)
        
        residual = self.proj.ravel() - this_proj

        return residual
    
    def gradient(self, rec, angles, translations):
    
        phi, alpha, beta = angles
        xyz_shift = translations
        this_proj, this_grad = self.proj_obj.projection_gradient(rec=rec, alpha=alpha, beta=beta, phi=phi, 
                                                                 xyz_shift=xyz_shift, cor_shift=self.geometry.cor_shift)
    
        residual = self.proj.ravel() - this_proj
        this_grad *= -1
        
        return residual, this_grad
    
    
def gradient_descent(x, cost_function, gradient_function, args=(), options={}):
    
    n_itmax = options['maxiter'] if 'maxiter' in options else 100
    step_search = options['step_search'] if 'step_search' in options else 'armijo'
    eps = options['eps'] if 'eps' in options else 1.e-6
    verbose = options['verbose'] if 'verbose' in options else False
    
    align_obj, rec, angles_in, xyz_in, scale_factor = args
    
    cost = np.zeros(n_itmax+1)
    stop = 0
    it = 0
    
    f = cost_function(x, align_obj, rec, angles_in, xyz_in, scale_factor=scale_factor, return_vector=False)
    fp = gradient_function(x, align_obj, rec, angles_in, xyz_in, scale_factor=scale_factor, return_vector=False)

    cost[it] = f
    ls_counter = 0
    alpha = 0.0
    while not stop and it < n_itmax:
        if verbose:
            print(it, f, alpha, fp, x)
        
        # linesearch with armijo
        search_dir = -fp#/np.linalg.norm(fp)
        if step_search == 'armijo':
            alpha, _, f_new = line_search_armijo(cost_function, x, search_dir, fp, cost[it], alpha0=1.0,
                                                 args=(align_obj, rec, angles_in, xyz_in, None))
        elif step_search == 'wolfe':
            if it > 1:
                old_old_val = cost[it-1]
            else:
                old_old_val = None
            #alpha,_,_, f_new, f_old, fp_new = line_search(cost_function, gradient_function, x, -fp, gfk=fp, old_fval=cost[it], 
            #                                              old_old_fval=old_old_val, 
            #                                              args=(align_obj, rec, angles_in, xyz_in, None)) 
            alpha, _, _, f_new, f_old, fp_new = line_search_wolfe1(cost_function, gradient_function, x, search_dir,
                                                                   gfk=fp, amax=1.e-3, amin=1.e-12,
                                                                   args=(align_obj, rec, angles_in, xyz_in, None, None))
        if alpha is None:
            print('%s line search failed' %(step_search))
            ls_success = False
            ls_counter += 1
            alpha = 1.0
            # starting at alpha=1.e-6, try successively dividing alpha by 10 to find step size
            # and terminate GD here
            while not ls_success and alpha > 1.e-15:
                alpha = alpha/10
                x_new = x - alpha * fp
                f_new = cost_function(x_new, align_obj, rec, angles_in, xyz_in, scale_factor, False)
                if f_new < cost[it]:
                    ls_success = True
            
            if not ls_success or ls_counter >= 2:
                # either linesearch or more than 2 successive brute linesearch
                # return previous values
                stop = 2
                it += 1
                if verbose:
                    print('either linesearch failed or two successive brute linesearch iterations')
        else:
            x = x - alpha * fp
            it += 1
            f = cost_function(x, align_obj, rec, angles_in, xyz_in, scale_factor, False)
            fp = gradient_function(x, align_obj, rec, angles_in, xyz_in, scale_factor, False)

            cost[it] = f
            if np.abs(cost[it] - cost[it-1])/max(cost[it], cost[it-1], 1.0) <= eps:
                stop = 1
    
    return x, f, stop


def cost_xzpab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([parameters[0]+xyz_in[0], xyz_in[1], xyz_in[2]+parameters[1]])
    phi, alpha, beta = angles_in + parameters[2:]
    angles = np.array([phi, alpha, beta])
    
    cost = align_obj.cost(rec, angles, translations)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_xzpab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([parameters[0]+xyz_in[0], xyz_in[1], xyz_in[2]+parameters[1]])
    phi, alpha, beta = angles_in + parameters[2:]
    angles = np.array([phi, alpha, beta])
    
    residual, s = align_obj.gradient(rec, angles, translations)
    
    vary_parameter = np.array([True, False, True, True, True, True])
    s = s[vary_parameter, :]
    
    if scale_factor is None:
        scale_factor = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    s = s * scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T
    
    grad = np.dot(s, residual)
    
    return grad


def cost_xzab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    phi = angles_in[0]
    alpha, beta = angles_in[1:] + parameters[2:]
    angles = np.array([phi, alpha, beta])
    
    cost = align_obj.cost(rec, angles, translations)
    
    if return_vector:
        return cost
    
    return 0.5*np.linalg.norm(cost)**2
    

def gradient_xzab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    phi = angles_in[0]
    alpha, beta = angles_in[1:] + parameters[2:]
    angles = np.array([phi, alpha, beta])
    
    residual, s = align_obj.gradient(rec, angles, translations)
    
    vary_parameter = np.array([True, False, True, False, True, True])
    s = s[vary_parameter]
    
    if scale_factor is None:
        scale_factor = np.array([1.0, 1.0, 1.0, 1.0])
    
    s = s*scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T

    grad = np.dot(s, residual)
    
    return grad


def cost_xz(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    
    cost = align_obj.cost(rec, angles_in, translations)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_xz(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    
    residual, s = align_obj.gradient(rec, angles_in, translations)
    
    vary_parameter = np.array([True, False, True, False, False, False])
    s = s[vary_parameter]
    
    if scale_factor is None:
        scale_factor = np.array([1.0, 1.0])
    
    s = s * scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T
    
    grad = np.dot(s, residual)
    
    return grad


def gradient_xz_fd(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    eps = np.array([1.e-4, 1.e-3, 1.e-4])
    grad = np.zeros(3,)
    for i in range(3):
        tt = np.zeros(3,)
        tt[i] = 1.0
        trans_plus = translations + tt*eps
        trans_minus = translations - tt*eps
        cost_plus = align_obj.cost(rec, angles_in, trans_plus)
        cost_minus = align_obj.cost(rec, angles_in, trans_minus)
        cost_plus = 0.5*np.linalg.norm(cost_plus)**2
        cost_minus = 0.5*np.linalg.norm(cost_minus)**2
        grad[i] = (cost_plus - cost_minus)/(2*eps[i])
        
    return np.array([grad[0], grad[2]])


def cost_x(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]])
    
    cost = align_obj.cost(rec, angles_in, translations)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_x(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]])
    
    residual, s = align_obj.gradient(rec, angles_in, translations)
    
    vary_parameter = np.array([True, False, False, False, False, False])
    s = s[vary_parameter]
    
    if scale_factor is None:
        scale_factor = np.array([1.0])
    
    s = s * scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T
    
    grad = np.dot(s, residual)
    
    return grad


def cost_z(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0], xyz_in[1], xyz_in[2]+parameters[0]])
    
    cost = align_obj.cost(rec, angles_in, translations)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_z(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    translations = np.array([xyz_in[0], xyz_in[1], xyz_in[2]+parameters[0]])
    
    residual, s = align_obj.gradient(rec, angles_in, translations)
    
    vary_parameter = np.array([False, False, True, False, False, False])
    s = s[vary_parameter]
    
    if scale_factor is None:
        scale_factor = np.array([1.0])
    
    s = s * scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T
    
    grad = np.dot(s, residual)
    
    return grad


def cost_ab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    phi = angles_in[0]
    alpha, beta = angles_in[1:]+parameters
    angles = np.array([phi, alpha, beta])
    cost = align_obj.cost(rec, angles, xyz_in)
    
    if return_vector:
        return cost
    
    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_ab(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    phi = angles_in[0]
    alpha, beta = angles_in[1:] + parameters
    angles = np.array([phi, alpha, beta])
    residual, s = align_obj.gradient(rec, angles, xyz_in)
    
    vary_parameter = np.array([False, False, False, False, True, True])
    s = s[vary_parameter]
    
    if scale_factor is None:
        scale_factor = np.array([1.0, 1.0])
    
    s = s * scale_factor[:, np.newaxis]
    
    if return_vector:
        return s.T
    
    grad = np.dot(s, residual)
    
    return grad


def cost_a(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    phi = angles_in[0]
    alpha = angles_in[1] + parameters[0] 
    beta = angles_in[2]
    angles = np.array([phi, alpha, beta])
    cost = align_obj.cost(rec, angles, xyz_in)

    if return_vector:
        return cost

    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_a(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    phi = angles_in[0]
    alpha = angles_in[1] + parameters[0] 
    beta = angles_in[2]
    angles = np.array([phi, alpha, beta])
    residual, s = align_obj.gradient(rec, angles, xyz_in)

    vary_parameter = np.array([False, False, False, False, True, False])
    s = s[vary_parameter]

    if scale_factor is None:
        scale_factor = np.array([1.0])

    s = s * scale_factor[:, np.newaxis]

    if return_vector:
        return s.T

    grad = np.dot(s, residual)#[vary_parameter]

    return grad


def cost_b(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    phi = angles_in[0]
    alpha = angles_in[1] 
    beta = angles_in[2] + parameters[0]
    angles = np.array([phi, alpha, beta])
    cost = align_obj.cost(rec, angles, xyz_in)

    if return_vector:
        return cost

    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_b(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    phi = angles_in[0]
    alpha = angles_in[1]
    beta = angles_in[2] + parameters[0]
    angles = np.array([phi, alpha, beta])
    residual, s = align_obj.gradient(rec, angles, xyz_in)

    vary_parameter = np.array([False, False, False, False, False, True])
    s = s[vary_parameter]

    if scale_factor is None:
        scale_factor = np.array([1.0])

    s = s * scale_factor[:, np.newaxis]

    if return_vector:
        return s.T

    grad = np.dot(s, residual)#[vary_parameter]

    return grad


def gradient_ab_fd(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):
    
    phi = angles_in[0]
    alpha, beta = angles_in[1:] + parameters
    angles = np.array([phi, alpha, beta])
    residual, s = align_obj.gradient(rec, angles, xyz_in)
    
    eps = np.array([1.e-3, 1.e-4, 1.e-4])
    grad = np.zeros(3,)
    for i in range(3):
        tt = np.zeros(3,)
        tt[i] = 1.0
        angles_plus = angles_in + tt*eps
        angles_minus = angles_in - tt*eps

        cost_plus = align_obj.cost(rec, angles_plus, xyz_in)
        cost_minus = align_obj.cost(rec, angles_minus, xyz_in)
        cost_plus = 0.5*np.linalg.norm(cost_plus)**2
        cost_minus = 0.5*np.linalg.norm(cost_minus)**2
        grad[i] = (cost_plus - cost_minus)/(2*eps[i])
    
    return grad[1:]


def cost_xzb(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    translations = np.array([xyz_in[0]+parameters[0], xyz_in[1], xyz_in[2]+parameters[1]])
    phi = angles_in[0]
    alpha = angles_in[1]
    beta = angles_in[2] + parameters[2]
    angles = np.array([phi, alpha, beta])
    cost = align_obj.cost(rec, angles, translations)

    if return_vector:
        return cost

    return 0.5 * np.linalg.norm(cost) ** 2


def gradient_xzb(parameters, align_obj, rec, angles_in, xyz_in, scale_factor=None, return_vector=False):

    translations = np.array([xyz_in[0] + parameters[0], xyz_in[1], xyz_in[2] + parameters[1]])
    phi = angles_in[0]
    alpha = angles_in[1]
    beta = angles_in[2] + parameters[2]
    angles = np.array([phi, alpha, beta])
    residual, s = align_obj.gradient(rec, angles, translations)

    vary_parameter = np.array([True, False, True, False, False, True])
    s = s[vary_parameter]

    if scale_factor is None:
        scale_factor = np.array([1.0, 1.0, 1.0])

    s = s * scale_factor[:, np.newaxis]

    if return_vector:
        return s.T

    grad = np.dot(s, residual)

    return grad
