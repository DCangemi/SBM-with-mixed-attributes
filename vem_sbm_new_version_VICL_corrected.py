
"""
Variational EM for an SBM with optional Gaussian (continuous) and Categorical (discrete) node attributes,
plus vICL model-selection score.

Conventions:
- Undirected network A (n x n), with A[i,j] in {0,1} and A symmetric, diagonal ignored.
- Responsibilities r: shape (n, K), rows sum to 1.
- pi: shape (K,)
- Theta: shape (K, K), symmetric for undirected networks.
- Xg: optional (n, d) continuous features
- Xc: optional (n, M) integer categorical features, coded as 1..C_m (IMPORTANT).
- phi_list: list length M, each phi_list[m] is shape (K, C_m), rows sum to 1.

Changes from the original R translation:
- M-step for Theta: correctly accumulates both cross-terms (r_ik*r_jl + r_il*r_jk)
  for the undirected case, instead of accumulating one term and post-hoc averaging.
- vICL: uses the PDF formula (higher is better), excludes mixing-proportion parameters
  from the penalty term as prescribed by the theoretical derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def logsumexp(v: np.ndarray) -> float:
    """Stable log(sum(exp(v)))."""
    v = np.asarray(v, dtype=float)
    m = np.max(v)
    return float(m + np.log(np.sum(np.exp(v - m))))


def softmax_row(v: np.ndarray) -> np.ndarray:
    """Stable softmax for 1D array."""
    v = np.asarray(v, dtype=float)
    m = np.max(v)
    ex = np.exp(v - m)
    s = np.sum(ex)
    if s <= 0:
        # fallback: uniform
        return np.ones_like(v) / len(v)
    return ex / s


def _is_pd(S: np.ndarray) -> bool:
    """Check positive definiteness (symmetric)."""
    try:
        # Cholesky is the quickest robust check
        np.linalg.cholesky(S)
        return True
    except np.linalg.LinAlgError:
        return False


def log_mvnorm(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, jitter: float = 1e-6) -> float:
    """
    Log-density of multivariate normal N(mu, Sigma) at x.
    Mirrors R: adds jitter if Sigma is not PD.
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)

    d = x.shape[0]
    S = Sigma
    if S.shape != (d, d):
        raise ValueError(f"Sigma must be ({d},{d}); got {S.shape}")

    if not _is_pd(S):
        S = S + np.eye(d) * jitter

    sign, logdet = np.linalg.slogdet(S)
    if sign <= 0:
        # last resort: add more jitter
        S = S + np.eye(d) * (10 * jitter)
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            raise np.linalg.LinAlgError("Sigma is not positive definite even after jitter.")

    Sinv = np.linalg.inv(S)
    diff = (x - mu).reshape(-1, 1)
    qf = float((diff.T @ Sinv @ diff)[0, 0])
    return float(-(d / 2) * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * qf)


def entropy_q(r: np.ndarray, eps: float = 1e-15) -> float:
    """Entropy of mean-field q(Z)=prod_i Cat(r_i). Returns -sum r log r."""
    r = np.asarray(r, dtype=float)
    return float(-np.sum(r * np.log(np.maximum(r, eps))))


def count_params(K: int, dg: int, Cvec: Sequence[int]) -> Dict[str, float]:
    """
    Degrees of freedom, matching the R code:
      df_pi = K-1
      df_theta = K*(K+1)/2  (undirected, symmetric Theta)
      df_attr = K * (df_gauss_perk + df_cat_perk)
    """
    df_pi = K - 1
    df_theta = K * (K + 1) / 2.0
    df_gauss_perk = dg + dg * (dg + 1) / 2.0
    df_cat_perk = float(np.sum(np.asarray(Cvec, dtype=float) - 1.0)) if len(Cvec) else 0.0
    df_attr = K * (df_gauss_perk + df_cat_perk)
    return dict(df_pi=df_pi, df_theta=df_theta, df_attr=df_attr, df_tot=df_pi + df_theta + df_attr)


def vicl_score(ELBO: float, Hq: float, n: int, K: int, dg: int, Cvec: Sequence[int]) -> float:
    """
    vICL as defined in the PDF (Section 4.7), higher is better:

      vICL(K) = E_q[log p(A, Xg, Xc, Z | Theta_hat)]
               - 0.5 * ( p_net * log(n(n-1)/2) + (p_cont + p_cat) * log(n) )

    Note:
    - E_q[log p(...)] = ELBO - Hq  (ELBO includes entropy; subtract it back out).
    - p_net  = K(K+1)/2
    - p_cont = K * (d + d(d+1)/2)
    - p_cat  = K * sum_m (C_m - 1)
    """
    pars = count_params(K, dg, Cvec)
    Npairs = n * (n - 1) / 2.0
    Eq_logp = ELBO - Hq  # expected complete-data log-likelihood under q
    penalty = 0.5 * (pars["df_theta"] * np.log(Npairs) + (pars["df_attr"] + pars["df_pi"]) * np.log(n))
    return float(Eq_logp - penalty)


def elbo_value(
    A: np.ndarray,
    Xg: Optional[np.ndarray],
    Xc: Optional[np.ndarray],
    r: np.ndarray,
    pi: np.ndarray,
    Theta: np.ndarray,
    mu_list: List[np.ndarray],
    Sigma_list: List[np.ndarray],
    phi_list: List[np.ndarray],
    eps_r: float = 1e-15,
) -> float:
    """
    Direct translation of R's elbo_value (loop-based, O(n^2 K^2)).
    ELBO here includes +entropy term (term_Hq).
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    pi = np.asarray(pi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)
    n, K = r.shape

    term_net = 0.0
    logT = np.log(Theta)
    log1mT = np.log(1.0 - Theta)

    for i in range(n - 1):
        for j in range(i + 1, n):
            aij = A[i, j]
            for k in range(K):
                rik = r[i, k]
                for l in range(K):
                    Fkl = logT[k, l] if aij == 1 else log1mT[k, l]
                    term_net += rik * r[j, l] * Fkl

    term_pi = float(np.sum(r * np.log(pi.reshape(1, -1))))

    term_g = 0.0
    if Xg is not None:
        Xg = np.asarray(Xg, dtype=float)
        for i in range(n):
            for k in range(K):
                term_g += r[i, k] * log_mvnorm(Xg[i, :], mu_list[k], Sigma_list[k])

    term_c = 0.0
    if Xc is not None:
        Xc = np.asarray(Xc, dtype=int)
        M = Xc.shape[1]
        for i in range(n):
            for k in range(K):
                s = 0.0
                for m in range(M):
                    cval = int(Xc[i, m])
                    # R indexes categories starting at 1
                    s += np.log(phi_list[m][k, cval - 1])
                term_c += r[i, k] * s

    term_Hq = float(-np.sum(r * np.log(np.maximum(r, eps_r))))
    return float(term_net + term_pi + term_g + term_c + term_Hq)


def E_step(
    A: np.ndarray,
    Xg: Optional[np.ndarray],
    Xc: Optional[np.ndarray],
    r: np.ndarray,
    pi: np.ndarray,
    Theta: np.ndarray,
    mu_list: List[np.ndarray],
    Sigma_list: List[np.ndarray],
    phi_list: List[np.ndarray],
) -> np.ndarray:
    """
    Translation of R's E_step:
      psi[i,k] = log(pi_k) + log_attr[i,k] + net_part[i,k]
      r_new[i,:] = softmax(psi[i,:])
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    pi = np.asarray(pi, dtype=float)
    Theta = np.asarray(Theta, dtype=float)

    n = A.shape[0]
    K = pi.shape[0]

    logT = np.log(Theta)
    log1mT = np.log(1.0 - Theta)

    net_part = np.zeros((n, K), dtype=float)
    for i in range(n):
        for k in range(K):
            s = 0.0
            for j in range(n):
                if j == i:
                    continue
                aij = A[i, j]
                for l in range(K):
                    Fkl = logT[k, l] if aij == 1 else log1mT[k, l]
                    s += r[j, l] * Fkl
            net_part[i, k] = s

    log_attr = np.zeros((n, K), dtype=float)

    if Xg is not None:
        Xg = np.asarray(Xg, dtype=float)
        for i in range(n):
            for k in range(K):
                log_attr[i, k] += log_mvnorm(Xg[i, :], mu_list[k], Sigma_list[k])

    if Xc is not None:
        Xc = np.asarray(Xc, dtype=int)
        M = Xc.shape[1]
        for i in range(n):
            for k in range(K):
                s = 0.0
                for m in range(M):
                    cval = int(Xc[i, m])
                    s += np.log(phi_list[m][k, cval - 1])
                log_attr[i, k] += s

    psi = np.log(pi.reshape(1, -1)) + log_attr + net_part

    r_new = np.empty_like(r)
    for i in range(n):
        r_new[i, :] = softmax_row(psi[i, :])

    return r_new


def M_step(
    A: np.ndarray,
    Xg: Optional[np.ndarray],
    Xc: Optional[np.ndarray],
    r: np.ndarray,
    clip: Dict[str, float] = {"theta": 1e-5, "phi": 1e-9},
) -> Dict[str, Any]:
    """
    Translation of R's M_step.
    - Updates pi, Theta (symmetrized), mu_list/Sigma_list (if Xg), phi_list (if Xc).
    """
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    n, K = r.shape

    N_k = np.sum(r, axis=0)  # shape (K,)
    pi = N_k / n

    # Undirected network: accumulate both cross-terms per the PDF (Section 2.7)
    #   C_kl = sum_{i<j} (r_ik*r_jl + r_il*r_jk) * A_ij
    #   W_kl = sum_{i<j} (r_ik*r_jl + r_il*r_jk)
    #   theta_kl = C_kl / W_kl,  for k <= l
    # This yields a symmetric Theta without post-hoc averaging.
    C_kl = np.zeros((K, K), dtype=float)
    W_kl = np.zeros((K, K), dtype=float)

    for i in range(n - 1):
        for j in range(i + 1, n):
            aij = A[i, j]
            for k in range(K):
                for l in range(K):
                    w = r[i, k] * r[j, l] + r[i, l] * r[j, k]
                    W_kl[k, l] += w
                    C_kl[k, l] += w * aij

    Theta = np.zeros((K, K), dtype=float)
    for k in range(K):
        for l in range(k, K):
            Theta[k, l] = C_kl[k, l] / W_kl[k, l] if W_kl[k, l] > 0 else 0.5
            Theta[l, k] = Theta[k, l]

    epsT = float(clip.get("theta", 1e-5))
    Theta = np.clip(Theta, epsT, 1.0 - epsT)

    mu_list: List[np.ndarray] = [np.array([], dtype=float) for _ in range(K)]
    Sigma_list: List[np.ndarray] = [np.array([[]], dtype=float) for _ in range(K)]

    if Xg is not None:
        Xg = np.asarray(Xg, dtype=float)
        d = Xg.shape[1]
        mu_list = []
        Sigma_list = []
        for k in range(K):
            if N_k[k] > 0:
                muk = (r[:, k].reshape(-1, 1) * Xg).sum(axis=0) / N_k[k]
                mu_list.append(muk.astype(float))
                Sk = np.zeros((d, d), dtype=float)
                for i in range(n):
                    diff = (Xg[i, :] - muk).reshape(-1, 1)
                    Sk += r[i, k] * (diff @ diff.T)
                Sk = Sk / N_k[k]
                if not _is_pd(Sk):
                    Sk = Sk + np.eye(d) * 1e-6
                Sigma_list.append(Sk)
            else:
                mu_list.append(np.zeros(d, dtype=float))
                Sigma_list.append(np.eye(d, dtype=float))

    phi_list: List[np.ndarray] = []
    if Xc is not None:
        Xc = np.asarray(Xc, dtype=int)
        M = Xc.shape[1]
        epsP = float(clip.get("phi", 1e-9))
        for m in range(M):
            C_m = int(np.max(Xc[:, m]))
            phi_km = np.zeros((K, C_m), dtype=float)
            for k in range(K):
                for c in range(1, C_m + 1):
                    phi_km[k, c - 1] = np.sum(r[:, k] * (Xc[:, m] == c))
                if N_k[k] > 0:
                    phi_km[k, :] = phi_km[k, :] / N_k[k]
                else:
                    phi_km[k, :] = 1.0 / C_m
            # clip + renormalize each row (as in R)
            phi_km = np.maximum(phi_km, epsP)
            phi_km = phi_km / phi_km.sum(axis=1, keepdims=True)
            phi_list.append(phi_km)

    return dict(pi=pi, Theta=Theta, mu_list=mu_list, Sigma_list=Sigma_list, phi_list=phi_list)


def VEM_SBM(
    A: np.ndarray,
    Xg: Optional[np.ndarray] = None,
    Xc: Optional[np.ndarray] = None,
    K: int = 2,
    max_iter: int = 200,
    tol: float = 1e-6,
    verbose: bool = True,
    init: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Variational EM main loop, translation of the R function VEM_SBM.

    init may contain:
      r, pi, Theta, mu_list, Sigma_list, phi_list
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]

    if init is None:
        init = {}

    # init r
    if init.get("r") is None:
        r = np.random.rand(n, K)
        r = r / r.sum(axis=1, keepdims=True)
    else:
        r0 = np.asarray(init["r"], dtype=float)
        r = r0 / r0.sum(axis=1, keepdims=True)


    # init pi
    if init.get("pi") is None:
        pi = np.full((K,), 1.0 / K, dtype=float)
    else:
        pi = np.asarray(init["pi"], dtype=float)

    # init Theta
    if init.get("Theta") is None:
        Theta = np.full((K, K), 0.5, dtype=float)
        np.fill_diagonal(Theta, 0.6)
    else:
        Theta = np.asarray(init["Theta"], dtype=float)

    # init gaussian params
    if Xg is not None:
        Xg = np.asarray(Xg, dtype=float)
        d = Xg.shape[1]
        if init.get("mu_list") is None:
            mu_list = [Xg.mean(axis=0) for _ in range(K)]
        else:
            mu_list = [np.asarray(v, dtype=float) for v in init["mu_list"]]
        if init.get("Sigma_list") is None:
            # R uses diag(var(as.vector(Xg))+1e-3, d)
            v = float(np.var(Xg.reshape(-1), ddof=1)) + 1e-3
            Sigma_list = [np.eye(d) * v for _ in range(K)]
        else:
            Sigma_list = [np.asarray(S, dtype=float) for S in init["Sigma_list"]]
    else:
        mu_list = [np.array([], dtype=float) for _ in range(K)]
        Sigma_list = [np.array([[]], dtype=float) for _ in range(K)]

    # init categorical params
    phi_list: List[np.ndarray] = []
    if Xc is not None:
        Xc = np.asarray(Xc, dtype=int)
        M = Xc.shape[1]
        if init.get("phi_list") is None:
            for m in range(M):
                C_m = int(np.max(Xc[:, m]))
                phi_list.append(np.full((K, C_m), 1.0 / C_m, dtype=float))
        else:
            phi_list = [np.asarray(P, dtype=float) for P in init["phi_list"]]

    elbo_hist: List[float] = []

    for it in range(1, max_iter + 1):
        r = E_step(A, Xg, Xc, r, pi, Theta, mu_list, Sigma_list, phi_list)
        upd = M_step(A, Xg, Xc, r)
        pi, Theta = upd["pi"], upd["Theta"]
        mu_list, Sigma_list = upd["mu_list"], upd["Sigma_list"]
        phi_list = upd["phi_list"]

        cur_elbo = elbo_value(A, Xg, Xc, r, pi, Theta, mu_list, Sigma_list, phi_list)
        elbo_hist.append(cur_elbo)

        if verbose:
            print(f"Iter {it} | ELBO={cur_elbo:.6f}")

        if it > 1:
            prev = elbo_hist[-2]
            if abs(cur_elbo - prev) < tol * (1.0 + abs(prev)):
                if verbose:
                    print("Convergenza.")
                break

    ELBO_last = elbo_hist[-1]
    Hq = entropy_q(r)

    dg = 0 if Xg is None else int(Xg.shape[1])
    Cvec = [] if Xc is None else [int(np.max(Xc[:, m])) for m in range(Xc.shape[1])]
    vICL = vicl_score(ELBO_last, Hq, n, K, dg, Cvec)

    return dict(
        r=r,
        pi=pi,
        Theta=Theta,
        mu_list=mu_list,
        Sigma_list=Sigma_list,
        phi_list=phi_list,
        ELBO=np.array(elbo_hist, dtype=float),
        iters=len(elbo_hist),
        vICL=vICL,
    )


# ---- Example (same as in the R file) ----
if __name__ == "__main__":
    np.random.seed(1)
    A = np.zeros((3, 3), dtype=float)
    A[0, 1] = A[1, 0] = 1
    A[1, 2] = A[2, 1] = 1

    Xg = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    Xc = np.array([[1, 2, 1], [2, 1, 2], [1, 2, 2]], dtype=int)[:, :2]  # example with M=2

    fit = VEM_SBM(A, Xg=Xg, Xc=Xc, K=2, max_iter=50, tol=1e-8, verbose=True)
    print("\n=== RISULTATI FINALI ===")
    print(np.round(fit["r"], 3))
    print("pi =", np.round(fit["pi"], 3))
    print("Theta =\n", np.round(fit["Theta"], 3))
    print("vICL =", fit["vICL"])
