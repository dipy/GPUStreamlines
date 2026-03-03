"""Shared utilities for bootstrap direction getters (CUDA and Metal).

Extracts DIPY model matrices (H, R, delta_b, delta_q, sampling_matrix)
for OPDT and CSA models.  Both backends need the same matrices — only
the GPU dispatch differs.
"""

from dipy.reconst import shm


def prepare_opdt(gtab, sphere, sh_order_max=6, full_basis=False,
                 sh_lambda=0.006, min_signal=1):
    """Build bootstrap matrices for the OPDT model.

    Returns dict with keys: model_type, min_signal, H, R, delta_b,
    delta_q, sampling_matrix, b0s_mask.
    """
    sampling_matrix, _, _ = shm.real_sh_descoteaux(
        sh_order_max, sphere.theta, sphere.phi,
        full_basis=full_basis, legacy=True,
    )
    model = shm.OpdtModel(
        gtab, sh_order_max=sh_order_max, smooth=sh_lambda,
        min_signal=min_signal,
    )
    delta_b, delta_q = model._fit_matrix

    H, R = _hat_and_lcr(gtab, model, sh_order_max)

    return dict(
        model_type="OPDT", min_signal=min_signal,
        H=H, R=R, delta_b=delta_b, delta_q=delta_q,
        sampling_matrix=sampling_matrix, b0s_mask=gtab.b0s_mask,
    )


def prepare_csa(gtab, sphere, sh_order_max=6, full_basis=False,
                sh_lambda=0.006, min_signal=1):
    """Build bootstrap matrices for the CSA model.

    Returns dict with keys: model_type, min_signal, H, R, delta_b,
    delta_q, sampling_matrix, b0s_mask.
    """
    sampling_matrix, _, _ = shm.real_sh_descoteaux(
        sh_order_max, sphere.theta, sphere.phi,
        full_basis=full_basis, legacy=True,
    )
    model = shm.CsaOdfModel(
        gtab, sh_order_max=sh_order_max, smooth=sh_lambda,
        min_signal=min_signal,
    )
    delta_b = model._fit_matrix
    delta_q = model._fit_matrix

    H, R = _hat_and_lcr(gtab, model, sh_order_max)

    return dict(
        model_type="CSA", min_signal=min_signal,
        H=H, R=R, delta_b=delta_b, delta_q=delta_q,
        sampling_matrix=sampling_matrix, b0s_mask=gtab.b0s_mask,
    )


def _hat_and_lcr(gtab, model, sh_order_max):
    """Compute hat matrix H and leveraged centered residuals matrix R."""
    dwi_mask = ~gtab.b0s_mask
    x, y, z = model.gtab.gradients[dwi_mask].T
    _, theta, phi = shm.cart2sphere(x, y, z)
    B, _, _ = shm.real_sh_descoteaux(sh_order_max, theta, phi, legacy=True)
    H = shm.hat(B)
    R = shm.lcr_matrix(H)
    return H, R
