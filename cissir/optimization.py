from abc import ABC, abstractmethod
import time
import numpy as np
from numpy import linalg as la
from numpy.fft import rfft, irfft
from numpy.polynomial import Polynomial
from numba import jit
import cvxpy as cp


def herm(matrix):
    return matrix.swapaxes(-2, -1).conj()


def rank1mat2vec(mat: np.ndarray) -> np.ndarray:
    """
    Extract vector :math:`\mathbf{w}` from matrix :math:`\mathbf{M} = \mathbf{w} \mathbf{w}^H`.
    Rank of :math:`\mathbf{M}` is not checked
    :param mat: Matrix :math:`\mathbf{M}` to decompose
    :return: Extracted vector :math:`\mathbf{w}`
    """

    eigvals, eigvecs = np.linalg.eigh(mat)
    eigval1 = eigvals[..., -1:]
    eigvec1 = eigvecs[..., -1]

    return np.sqrt(eigval1) * eigvec1


def si_split(mat):

    u, s, vh = la.svd(mat, full_matrices=False)    # SVD decomposition to split channel between tx and rx
    s = s[..., None] * np.eye(s.shape[-1])[None,]
    mat_left = np.sum(u @ s @ herm(u), axis=0)
    mat_right = np.sum(herm(vh) @ s @ vh, axis=0)

    return mat_left, mat_right


def ordered_eigvals(mat, hermitian=False):
    if hermitian:
        evals = la.eigvalsh(mat)
    else:
        evals = la.eigvals(mat)

    return np.sort(np.abs(evals.flatten()))[::-1]


def max_eigval(mat, hermitian=False):
    eigvals = ordered_eigvals(mat, hermitian)
    return eigvals[0]


def min_eigval(mat, hermitian=False):
    eigvals = ordered_eigvals(mat, hermitian)
    return eigvals[-1]


def spec_nuc_ratio(mat, axes=(-1, -2)):
    """
    Compute :math:`\|\mathbf{M}\|_{2}/\|\mathbf{M}\|_{*}` ,
    the ratio between the spectral and nuclear norm of :math:`\mathbf{M}` ,
    as a metric of "one-rankness"
    :param mat: Matrix :math:`\mathbf{M}`
    :param axes: Matrix axes on which to compute the ratio
    :return: Spectral-to-nuclear norm ratio
    """

    return la.norm(mat, ord=2, axis=axes)/la.norm(mat, ord="nuc", axis=axes)


def split_si_threshold(si_threshold_power, beta):
    return si_threshold_power ** beta, si_threshold_power ** (1 - beta)


def codebook_si(tx_codebook, rx_codebook, si_channel):

    mat_prod = herm(rx_codebook[None,]) @ si_channel @ tx_codebook[None,]
    return np.abs(mat_prod).sum(axis=0)


def sdr_si(tx_codebook_sdr, rx_codebook_sdr, si_channel):
    hsi_bcast = si_channel[:, None, None, ]
    tx_bcast = tx_codebook_sdr[None, None, :,]
    rx_bcast = rx_codebook_sdr[None, :, None,]
    si_sdr = np.sum(np.sqrt(np.maximum(np.trace(tx_bcast @ herm(hsi_bcast) @
                                                rx_bcast @ hsi_bcast,
                                                axis1=-2, axis2=-1).real, 0)), axis=0)
    return si_sdr


def codebook_deviation_power(opt_codebook, ref_codebook, axis=None):
    # Frobenius for matrix, 2-norm for vectors
    ref_norm2 = la.norm(ref_codebook, ord=None, axis=axis) ** 2
    # ref_norm2 = ref_codebook.shape[-1]   # Squared norm equals number of codebook words
    return la.norm(opt_codebook - ref_codebook, ord=None, axis=axis) ** 2 / ref_norm2


def sdr_codebook_deviation(opt_codebook_sdr, ref_codebook_gram):
    ref_norm2 = ref_codebook_gram.shape[0]  # Squared norm equals number of codebook words

    norm2 = 2 * (ref_norm2 - np.sum(np.sqrt(np.trace(ref_codebook_gram @ opt_codebook_sdr,
                                                     axis1=-2, axis2=-1).real), axis=0))
    return norm2/ref_norm2


def feasible_phased_array(codebook):
    """
    Projection of codebook to phased array constraints, according to LoneSTAR scaling.
    It cannot be used for CISSIR!
    :param codebook: Input codebook
    :return: Phased-array codebook
    """
    return codebook/np.abs(codebook)


def feasible_tapered(codebook):
    """
    Projection of codebook to tapered constraints, according to LoneSTAR scaling.
    It cannot be used for CISSIR!
    :param codebook: Input codebook
    :return: Tapered codebook
    """
    n_ant = codebook.shape[0]
    return codebook * np.sqrt(n_ant)/np.linalg.norm(codebook, axis=0, keepdims=True)


class codebookOptimizer(ABC):
    @abstractmethod
    def __init__(self, si_channel, tx_codebook, rx_codebook, si_threshold_power, beta=0.5, phased=True):
        self._beta = beta
        self._si_threshold = si_threshold_power
        self._si_channel = si_channel
        self._phased = phased

        self.tx_codebook = tx_codebook
        self.rx_codebook = rx_codebook

    @abstractmethod
    def solve(self, tx, *args, **kwargs):
        pass

    @property
    def si_threshold_power(self):
        return self._si_threshold

    @si_threshold_power.setter
    def si_threshold_power(self, value):
        self._si_threshold = value
        self._update_eps()

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self._update_eps()

    @property
    def si_channel(self):
        return self._si_channel

    @si_channel.setter
    def si_channel(self, value):
        self._si_channel = value
        self._update_si_channel()

    @abstractmethod
    def _update_eps(self):
        pass

    @abstractmethod
    def _update_si_channel(self):
        pass


class SDRcodebook(codebookOptimizer):
    def __init__(self, si_channel, tx_codebook, rx_codebook, si_threshold_power, beta=0.5, phased=True):
        self._beta = beta
        self._si_threshold = si_threshold_power
        self._si_channel = si_channel

        self.tx_codebook = tx_codebook
        self.rx_codebook = rx_codebook

        self._prob_tx = self._create_problem(tx_codebook, phased)
        self._prob_rx = self._create_problem(rx_codebook, phased)

        self._update_eps()
        self._update_si_channel()

    def solve(self, tx, return_sdr=False, solver='SCS', warm_start=True, **kwargs):
        kwargs.update({'solver': solver, 'warm_start': warm_start})
        if tx:
            ref_cb, prob = self.tx_codebook, self._prob_tx
        else:
            ref_cb, prob = self.rx_codebook, self._prob_rx

        x_var = prob.var_dict['X']
        ref_param = prob.param_dict['R']

        cb_gram = self.gram_codebook(ref_cb)
        x_sdr = np.zeros_like(cb_gram)

        for i, ref_sdp in enumerate(cb_gram):
            ref_param.value = ref_sdp
            x_var.value = ref_sdp
            prob.solve(**kwargs)
            if prob.status == 'infeasible':
                side = "TX" if tx else "RX"
                raise RuntimeError(f"The {side} problem is infeasible, try increasing the target SI")
            else:
                x_sdr[i,] = x_var.value

        x_rank1 = rank1mat2vec(x_sdr).T
        x_sol = self._rotate_codebook(x_rank1, ref_cb)

        if return_sdr:
            return x_sol, x_sdr
        else:
            return x_sol

    def _create_problem(self, ref_cb, phased):
        n_ant, l_beam = ref_cb.shape
        cb_gram = self.gram_codebook(ref_cb)

        eps_pow = cp.Parameter(pos=True, name="eps2")
        g_si = cp.Parameter(shape=(n_ant, n_ant), name="G", hermitian=True)
        ref_sdp = cp.Parameter(shape=(n_ant, n_ant), value=cb_gram[0,], name="R", hermitian=True)
        x_beam = cp.Variable(shape=(n_ant, n_ant), name="X", hermitian=True)

        if phased:
            constraints = [cp.real(d) == 1 / n_ant for d in cp.diag(x_beam)]
        else:
            constraints = [self._inner_prod(x_beam) == 1.0]

        constraints += [x_beam >> cp.Constant(0.0)]
        constraints += [self._inner_prod(x_beam, g_si) <= eps_pow]

        objective = cp.Maximize(self._inner_prod(x_beam, ref_sdp))
        return cp.Problem(objective, constraints=constraints)

    def _update_eps(self):
        eps_tx, eps_rx = split_si_threshold(self._si_threshold, self._beta)
        self._prob_tx.param_dict['eps2'].value = eps_tx
        self._prob_rx.param_dict['eps2'].value = eps_rx

    def _update_si_channel(self):
        si_rx, si_tx = si_split(self._si_channel)
        self._prob_rx.param_dict['G'].value = si_rx
        self._prob_tx.param_dict['G'].value = si_tx

    @staticmethod
    def _rotate_codebook(opt_cb, ref_cb):
        phase_cb = np.sum(np.conjugate(opt_cb) * ref_cb, axis=0, keepdims=True)
        phase_cb /= np.abs(phase_cb)
        return opt_cb * phase_cb

    @staticmethod
    def gram_codebook(ref_codebook):
        ref_cb = ref_codebook.T[..., None]
        return ref_cb @ herm(ref_cb)

    @staticmethod
    def _inner_prod(a, b=None):
        if b is None:
            return cp.trace(cp.real(a))
        else:
            return cp.trace(cp.real(b @ a))


class SOCPcodebook(codebookOptimizer):
    def __init__(self, si_channel, tx_codebook, rx_codebook, si_threshold_power, beta=0.5, phased=True):
        self._beta = beta
        self._si_threshold = si_threshold_power
        self._si_channel = si_channel
        self._phased = phased

        self.tx_codebook = tx_codebook
        self.rx_codebook = rx_codebook

        self._prob_tx = self._create_problem(tx_codebook, phased)
        self._prob_rx = self._create_problem(rx_codebook, phased)

        self._update_eps()
        self._update_si_channel()

    def solve(self, tx, solver='SCS', warm_start=True, **kwargs):
        kwargs.update({'solver': solver, 'warm_start': warm_start})
        if tx:
            ref_cb, prob = self.tx_codebook, self._prob_tx
        else:
            ref_cb, prob = self.rx_codebook, self._prob_rx

        x_var = prob.var_dict['x']
        ref_param = prob.param_dict['r']

        x_cb = np.zeros_like(ref_cb)

        for i, ref_vec in enumerate(ref_cb.T):
            ref_param.value = ref_vec
            x_var.value = ref_vec
            prob.solve(**kwargs)
            x_cb[:, i] = x_var.value

        return x_cb

    @staticmethod
    def _create_problem(ref_cb, phased):
        n_ant, l_beam = ref_cb.shape
        eps_amp = cp.Parameter(pos=True, name="eps_amp")
        g_chol = cp.Parameter(shape=(n_ant, n_ant), name="G_sqrt", complex=True)
        ref_vec = cp.Parameter(shape=n_ant, value=ref_cb[:, 0], name="r", complex=True)
        x_beam = cp.Variable(shape=n_ant, name="x", complex=True)

        if phased:
            constraints = [cp.SOC(1/np.sqrt(n_ant), e.T @ x_beam) for e in np.eye(n_ant, dtype=complex)[..., None]]
        else:
            constraints = [cp.SOC(1.0, x_beam)]
        constraints += [cp.SOC(eps_amp, g_chol @ x_beam)]

        objective = cp.Maximize(cp.real(ref_vec.H @ x_beam))
        return cp.Problem(objective, constraints=constraints)

    def _update_eps(self):
        eps_tx, eps_rx = split_si_threshold(self._si_threshold, self._beta)
        self._prob_tx.param_dict['eps_amp'].value = np.sqrt(eps_tx)
        self._prob_rx.param_dict['eps_amp'].value = np.sqrt(eps_rx)

    def _update_si_channel(self):
        si_rx, si_tx = si_split(self._si_channel)
        self._prob_rx.param_dict['G_sqrt'].value = herm(la.cholesky(si_rx))
        self._prob_tx.param_dict['G_sqrt'].value = herm(la.cholesky(si_tx))

    def feasible_gap(self, x_sol):
        n_ant, l_beam = x_sol.shape
        if self._phased:
            return np.max(1/np.sqrt(n_ant) - np.abs(x_sol))
        else:
            return np.max(np.ones(l_beam) - la.norm(x_sol, ord=2, axis=0))


def maxSdrError(cb_sdr, cb_sol):
    cb_vectors = cb_sol.T[..., None]
    max_err = np.max(np.abs(cb_vectors @ herm(cb_vectors) - cb_sdr))
    return max_err


def spectral_codebook(ref_cb, tgt_si, g_psd, si_tol=1e-2, real_tol=1e-2):
    eigvals, eigvecs = la.eigh(g_psd)
    si_levels = np.diagonal(herm(ref_cb) @ g_psd @ ref_cb).real

    n_eye = np.eye(len(eigvals))[None,]

    opt_mask = si_levels > (tgt_si * (1 + si_tol))
    poly_coeffs_cb = build_poly_codebook(ref_cb, tgt_si, eigvecs, eigvals)

    cb_roots = np.ones((poly_coeffs_cb.shape[0], poly_coeffs_cb.shape[-1]-1), dtype=complex)
    for i, poly_coeffs in enumerate(poly_coeffs_cb):
        if opt_mask[i]:     # Find polynomial roots only if optimization is needed
            cb_roots[i, :] = Polynomial(poly_coeffs).roots()
    real_mask = np.abs(cb_roots.imag / cb_roots.real) < real_tol
    nu_vec = np.where(real_mask, cb_roots, -1e9).real.max(axis=-1)

    d_mat = np.reciprocal(eigvals[None,] + nu_vec[:, None])[..., None] * n_eye
    qdq = eigvecs @ d_mat @ herm(eigvecs)

    qdq = np.where(opt_mask.reshape((-1, 1, 1)), qdq, n_eye)

    x_opt_cb = np.transpose(qdq @ ref_cb.T[..., None]).squeeze()

    return x_opt_cb/la.norm(x_opt_cb, ord=2, axis=0, keepdims=True)


def sigma_poly_coeffs_fft(eigvals, eps):
    p_weight = (eigvals - eps)
    n = eigvals.size    # Num. antennas P
    coeff_order = 2 * (n - 1)
    fft_size = coeff_order + 1

    ord2coeffs = np.stack([np.power(eigvals, 2), 2 * eigvals, np.ones_like(eigvals)],
                          axis=1)[None,]        # (\sigma_q^2 + 2\sigma_q\nu + \nu^2)

    p_coeff = np.concatenate([p_weight[:, None], np.zeros(shape=(n, 2))],
                             axis=1)[None,]     # (\sigma_p-\epsilon)

    ord2coeffs = np.where(np.eye(len(eigvals))[..., None], p_coeff, ord2coeffs)  # Substitute p-th coeff with eps factor
    ord2coeffs = np.concatenate([ord2coeffs, np.zeros(shape=(n, n, fft_size - 3))], axis=-1)    # Coeff padding

    coeff_conv = irfft(np.prod(rfft(ord2coeffs, ord2coeffs.shape[-1],   # FFT-based batch convolution to avoid loops
                                    axis=-1), axis=-2), ord2coeffs.shape[-1], axis=-1)[..., :coeff_order + 2]
    return coeff_conv   # Shape (P, 2(P-1) + 1)


def build_poly_codebook(ref_cb, eps, eigvecs, eigvals):
    sigma_coeffs = sigma_poly_coeffs_toeplitz(eigvals, eps)
    q_r2 = np.abs(herm(eigvecs) @ ref_cb) ** 2
    poly_coeffs_cb = np.sum(q_r2.T[..., None] * sigma_coeffs[None,], axis=1)    # Shape (Num. vectors L, (2P-1) +1)

    return poly_coeffs_cb / poly_coeffs_cb[..., -1:]    # Normalize the highest poly degree to 1


def largest_real_root(poly_coeffs, real_tol=1e-2):
    poly_nu = Polynomial(poly_coeffs)
    roots = poly_nu.roots()
    roots = roots[np.abs(roots.imag / roots.real) < real_tol]
    return max(roots.real)


def feasible_spectral(ref_cb, g_psd):
    eigvals, eigvecs = la.eigh(g_psd)
    q_r2 = np.abs(herm(eigvecs) @ ref_cb) ** 2
    weighted_eigval = np.sum(q_r2 / eigvals[:, None], axis=0)/np.sum(q_r2 / np.power(eigvals[:, None], 2), axis=0)
    return weighted_eigval


@jit(nopython=True)
def sigma_poly_coeffs_toeplitz(eigvals, eps):
    n = eigvals.size  # Num. antennas P
    coeff_order = 2 * (n - 1)
    c_ordp1 = coeff_order + 1

    ord2coeffs = np.stack((np.power(eigvals, 2), 2 * eigvals, np.ones_like(eigvals)), axis=1)
    ord2coeffs = np.concatenate((ord2coeffs, np.zeros((n, coeff_order - 2))), axis=1)

    coeffs_mat = np.concatenate((np.ones((1, n)), np.zeros((coeff_order, n))), axis=0)

    coeffs_acc = coeffs_mat[:, 0].copy()

    o2c: np.ndarray
    for i, o2c in enumerate(ord2coeffs):
        ip1 = i + 1
        i_coeff = min(c_ordp1, 2 * i + 3)
        coeffs_mat[:i_coeff, i] = coeffs_acc[:i_coeff].copy()

        o2c_toeplitz = np.zeros((i_coeff, i_coeff))
        for j in range(i_coeff):
            o2c_toeplitz[j:, j] = o2c[:i_coeff-j]
        coeffs_mat[:i_coeff, :ip1] = o2c_toeplitz @ coeffs_mat[:i_coeff, :ip1].copy()

        coeffs_acc[:i_coeff], coeffs_mat[:i_coeff, i] = coeffs_mat[:i_coeff, i].copy(), coeffs_acc[:i_coeff].copy()

    coeffs_mat *= np.expand_dims(eigvals, axis=0) - eps

    return np.transpose(coeffs_mat)


def compile_coeff_code():
    """
    Pre-compile `sigma_poly_coeffs_toeplitz` function with numba.
    :return:
    """

    print("Compiling `sigma_poly_coeffs_toeplitz` function...")
    start_time = time.perf_counter()
    _ = sigma_poly_coeffs_toeplitz(np.array([1.0, 1.0]), 1.0)
    end_time = time.perf_counter()
    print(f"Function compiled. time elapsed: {end_time - start_time:.3f} s")

    return
