"""
Integration Guide
=================
Shows the minimal changes needed to plug the new metrics into your
existing simulation scripts.  The key idea:

  1.  After prediction, compute training-set residual sigma (for CRPS / CI).
  2.  Call  compute_all_metrics(y_pred, y_true, sigma1, sigma2).
  3.  Append the returned dict to a list; summarise at the end.

Below is a trimmed version of your non-Gaussian + covariates loop
with the new metrics wired in.  The nonstationary script follows the
same pattern — just swap the data-loading section.
"""

# ─── Imports (add these to your existing imports) ────────────────
# from evaluation_metrics import (
#     compute_all_metrics,
#     estimate_sigma_from_residuals,
#     print_summary,
# )

# ─── Replace your results dict with a list-of-dicts ─────────────
# OLD:
#   results = {
#       'standard': {'mse_var1': [], 'mse_var2': []},
#       'weighted': {'mse_var1': [], 'mse_var2': []}
#   }
#
# NEW:
#   results_standard = []   # list of dicts, one per simulation
#   results_weighted  = []


# ─── Inside the simulation loop, AFTER predictions ──────────────
#
# The block below replaces the old per-variable mse(...) calls.
# Paste it twice: once for the standard model, once for the weighted.

"""
# ── Example for the STANDARD MSE model ──────────────────────────

# Step A: get training-set predictions to estimate sigma
y_pred_train_std = model_std.predict(phi_train, verbose=0)
y_pred_train_std[:, 0] = y_pred_train_std[:, 0] * np.sqrt(variance_var1) + mean1
y_pred_train_std[:, 1] = y_pred_train_std[:, 1] * np.sqrt(variance_var2) + mean2
y_train_orig = np.column_stack([
    df_train["var1"].values,
    df_train["var2"].values
])

sigma1_std = estimate_sigma_from_residuals(y_pred_train_std[:, 0], y_train_orig[:, 0])
sigma2_std = estimate_sigma_from_residuals(y_pred_train_std[:, 1], y_train_orig[:, 1])

# Step B: compute all metrics on the TEST set
metrics_std = compute_all_metrics(y_pred_std, y_test, sigma1_std, sigma2_std)
results_standard.append(metrics_std)

# (repeat for the weighted model → results_weighted.append(metrics_wt))


# ── At the end, after the simulation loop ───────────────────────

print_summary(results_standard, results_weighted, num_sim)

# Save the full metric tables to CSV
import pandas as pd
pd.DataFrame(results_standard).to_csv("metrics_standard.csv", index=False)
pd.DataFrame(results_weighted).to_csv("metrics_weighted.csv", index=False)
"""
