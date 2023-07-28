import numpy as np
from tomoalign.utils import projection_operators
from tomoalign.utils import linear_operators as tomo_linop
from scipy import sparse


class CGLS(object):
    def __init__(self, geometry, projections, angles, xyz_shift, options={}):
        self.geometry = geometry
        self.projections = projections
        self.angles = angles
        self.xyz_shift = xyz_shift
        self.n_proj = angles.shape[0]
        self.ground_truth = (
            options["ground_truth"] if "ground_truth" in options else None
        )
        self.rec = options["rec"] if "rec" in options else None
        if self.rec is None:
            self.rec = np.zeros((self.geometry.n_vox,), dtype=self.projections.dtype)
        self.precision = object["precision"] if "precision" in options else np.float32
        self.rms_error = None
        self.f_proj_obj = None
        self.proj_mat = None
        self._initialize()

    def _initialize(self):
        if self.f_proj_obj is None:
            # create an instance of the projection operator
            self.f_proj_obj = projection_operators.ProjectionMatrix(
                self.geometry, precision=self.precision
            )
            self.proj_mat = self.f_proj_obj.projection_matrix(
                phi=self.angles[:, 0],
                alpha=self.angles[:, 1],
                beta=self.angles[:, 2],
                xyz_shift=self.xyz_shift,
            )
        self._r = self.projections - sparse.csr_matrix.dot(
            self.proj_mat, self.rec
        ).reshape(self.n_proj, -1)
        self._p = sparse.csc_matrix.dot(
            sparse.csr_matrix.transpose(self.proj_mat), self._r.ravel()
        )

        self._gamma = np.linalg.norm(self._p) ** 2

    def run_main_iteration(self, make_plot=False, niter=100, debug=False):
        if self.ground_truth is None:
            norm_factor = np.linalg.norm(self.projections)
        else:
            norm_factor = np.linalg.norm(self.ground_truth)

        stop = 0
        k = 0
        conv = np.zeros((niter,))
        self.rms_error = np.zeros((niter,))
        reinit_iter = 0
        while not stop and k < niter:
            if self.method == "linop":
                r = self.f_proj_obj.project(
                    self._p, self.n_proj, self.angles, self.xyz_shift
                )
            else:
                r = sparse.csr_matrix.dot(self.proj_mat, self._p).reshape(
                    self.n_proj, -1
                )

            alpha = self._gamma / np.linalg.norm(r) ** 2
            self.rec += alpha * self._p
            conv[k] = np.linalg.norm(
                self.projections
                - sparse.csr_matrix.dot(self.proj_mat, self.rec).reshape(
                    self.n_proj, -1
                )
            )
            if k > 0 and conv[k] > conv[k - 1]:
                # re-initialize only if we did not re-initialize in the previous iteration
                print("reinitializing at iteration %d" % k)
                if reinit_iter + 1 == k:
                    print(
                        "need to re-initialize at two consecutive iterations: quitting"
                    )
                    return self.rec, self.rms_error[:k]
                self.rec -= alpha * self._p
                self._initialize()
                reinit_iter = k

            self._r -= alpha * r
            # now back-project
            p = sparse.csc_matrix.dot(
                sparse.csr_matrix.transpose(self.proj_mat), self._r.ravel()
            )
            gamma = np.linalg.norm(p) ** 2
            beta = gamma / self._gamma

            # update gamma and p
            self._gamma = gamma
            self._p = p + beta * self._p
            if self.ground_truth is None:
                self.rms_error[k] = np.linalg.norm(self._r) / norm_factor
            else:
                self.rms_error[k] = (
                    np.linalg.norm(self.rec - self.ground_truth.ravel()) / norm_factor
                )

            if make_plot:
                if k == 0:
                    import matplotlib.pyplot as plt

                    plt.ion()
                    fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

                elif k % 20 == 0:
                    ax0.imshow(
                        self.rec.reshape(self.geometry.vox_shape)[
                            :, :, self.geometry.vox_shape[2] // 2
                        ]
                    )
                    ax0.set_title("SIRT iteration %3d" % (k))

                    ax1.cla()
                    ax1.set_title("Root Mean-Squared Error")
                    ax1.semilogy(self.rms_error[1:k])

                    ax2.cla()
                    ax2.set_title("Convergence")
                    ax2.semilogy(conv[1:k])
                    plt.pause(0.1)
            k += 1

        return self.rec, self.rms_error[:k]
