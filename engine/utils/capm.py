def blend_mu(
    historical_mu_d: np.ndarray,
    capm_mu_d: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Shrink historical daily mu toward CAPM daily mu.
 
        blended = alpha * historical + (1 - alpha) * capm
 
    Default alpha=0.5 gives equal weight (50-50 blend).
 
    Parameters
    ----------
    alpha
        Weight on historical. 0.5 = equal blend. 1.0 = pure historical.
    """
    h = np.asarray(historical_mu_d, dtype=float)
    c = np.asarray(capm_mu_d, dtype=float)
    if h.shape != c.shape:
        raise ValueError(
            f"Shape mismatch: historical {h.shape} vs capm {c.shape}"
        )
    blended = alpha * h + (1.0 - alpha) * c
    logger.info(
        "Blended mu (alpha=%.2f): hist_mean=%.6f  capm_mean=%.6f  blend_mean=%.6f",
        alpha,
        float(h.mean()),
        float(c.mean()),
        float(blended.mean()),
    )
    return blended
 