"""
Stanford CS336 — Lectures 9 & 11: Scaling Laws
================================================
Spring 2026. Instructor: Tatsu Hashimoto.

Topics:
- Kaplan et al. (2020): "Scaling Laws for Neural Language Models"
- Chinchilla / Hoffmann et al. (2022): "Training Compute-Optimal LLMs"
- Key finding: model size and data should scale equally
- Fitting scaling laws: power-law regression

Key equations:
    L(N, D) = E + A / N^α + B / D^β
    N_opt(C) ∝ C^a,  D_opt(C) ∝ C^b  where C = 6ND (FLOPs)

    Kaplan:  a ≈ 0.73, b ≈ 0.27  (model-heavy)
    Chinchilla: a ≈ 0.50, b ≈ 0.50  (equal scaling)
"""
import numpy as np
from scipy.optimize import minimize


# ── Scaling law function ────────────────────────────────────────────────

def scaling_loss(
    params: np.ndarray,
    n_params: np.ndarray,
    n_tokens: np.ndarray,
) -> np.ndarray:
    """
    Compute predicted loss given scaling law parameters.

    L(N, D) = E + A / N^α + B / D^β

    params = [E, A, B, alpha, beta]
    n_params = model sizes (N)
    n_tokens = training tokens (D)
    """
    E, A, B, alpha, beta = params
    return E + A / n_params**alpha + B / n_tokens**beta


def fit_scaling_law(
    n_params: np.ndarray,
    n_tokens: np.ndarray,
    losses: np.ndarray,
) -> np.ndarray:
    """Fit scaling law parameters to observed data."""
    def objective(p):
        pred = scaling_loss(p, n_params, n_tokens)
        return np.mean((pred - losses) ** 2)

    # Initial guess: [E=1.0, A=100, B=100, alpha=0.5, beta=0.5]
    init = np.array([1.0, 100.0, 100.0, 0.5, 0.5])
    result = minimize(objective, init, bounds=[
        (0.0, None),   # E > 0
        (0.0, None),   # A > 0
        (0.0, None),   # B > 0
        (0.01, 1.0),   # alpha in (0, 1]
        (0.01, 1.0),   # beta in (0, 1]
    ])
    return result.x  # [E, A, B, alpha, beta]


# ── Compute-optimal allocation ──────────────────────────────────────────

def compute_optimal(
    compute_budget: float,  # total FLOPs available
    E: float, A: float, B: float, alpha: float, beta: float,
) -> tuple:
    """
    Given a compute budget C and scaling law params,
    find N_opt and D_opt that minimize L(N, D) subject to C ≈ 6ND.
    """
    # From the Lagrangian:
    # N_opt = (αA / βB)^{1/(α+β)} * (C/6)^{α/(α+β)}
    # D_opt = (βB / αA)^{1/(α+β)} * (C/6)^{β/(α+β)}

    ratio = (alpha * A) / (beta * B)
    n_opt = ratio ** (1 / (alpha + beta)) * (compute_budget / 6) ** (alpha / (alpha + beta))
    d_opt = (1 / ratio) ** (1 / (alpha + beta)) * (compute_budget / 6) ** (beta / (alpha + beta))
    return n_opt, d_opt


# ── Demo with synthetic data ────────────────────────────────────────────

def demo_scaling():
    """Generate synthetic scaling data and fit the law."""
    print("── Synthetic scaling law fitting ──\n")

    # True parameters (hidden)
    true_E, true_A, true_B, true_alpha, true_beta = 1.5, 400, 300, 0.4, 0.6

    # Generate "experiments": various (N, D) combos
    np.random.seed(42)
    n_experiments = 30
    n_params = 10 ** np.random.uniform(7, 10, n_experiments)  # 10M to 10B
    n_tokens = 10 ** np.random.uniform(8, 11, n_experiments)   # 100M to 100B

    true_loss = scaling_loss(
        np.array([true_E, true_A, true_B, true_alpha, true_beta]),
        n_params, n_tokens,
    )
    # Add noise
    observed_loss = true_loss + np.random.normal(0, 0.02, n_experiments)

    # Fit
    fitted = fit_scaling_law(n_params, n_tokens, observed_loss)
    labels = ["E (irreducible loss)", "A", "B", "alpha", "beta"]
    print("Parameter       True     Fitted")
    for name, true_val, fit_val in zip(labels,
                                        [true_E, true_A, true_B, true_alpha, true_beta],
                                        fitted):
        print(f"  {name:16s} {true_val:.3f}     {fit_val:.3f}")

    pred_loss = scaling_loss(fitted, n_params, n_tokens)
    mae = np.mean(np.abs(pred_loss - observed_loss))
    print(f"\n  Mean absolute error: {mae:.4f}")

    # Compute-optimal allocation
    print("\n── Compute-optimal allocation ──\n")
    for tflops in [1e18, 1e21, 1e24]:
        n_opt, d_opt = compute_optimal(tflops, *fitted)
        print(f"  C = {tflops:.0e} FLOPs ({tflops / 1e21:.1f} ZFLOPs)")
        print(f"    N_opt = {n_opt:.2e} params ({n_opt / 1e9:.1f}B)")
        print(f"    D_opt = {d_opt:.2e} tokens ({d_opt / 1e9:.1f}B)")
        print(f"    Verify C ≈ 6ND = {6 * n_opt * d_opt:.2e}")
        print()


# ── Key insights ────────────────────────────────────────────────────────

def key_insights():
    print("""
    ═══════════════ Scaling Laws: Key Insights ═══════════════

    1. Power-law form:
       L(N, D) = E + A/N^α + B/D^β
       Loss decreases predictably with more params and data.

    2. Chinchilla (Hoffmann et al., 2022):
       α ≈ β ≈ 0.5 → equal scaling of model and data.
       For every 2× more compute, increase N by ~1.4× and D by ~1.4×.

    3. Most models are "under-trained":
       A 70B model trained on 2T tokens? Chinchilla says you need
       ~10T tokens for compute-optimality at that scale.

    4. The "Chinchilla tax":
       Training compute-optimally requires much more data than most
       teams have. Hence the rise of data pipelines (lectures 13-14).

    5. Beyond Chinchilla:
       Repeated data (epochs > 1) has diminishing returns.
       Data quality can shift the entire curve down (reduce E).
    """)


if __name__ == "__main__":
    np.random.seed(42)
    demo_scaling()
    key_insights()
