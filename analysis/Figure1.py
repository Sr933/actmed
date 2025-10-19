import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from matplotlib import gridspec
from matplotlib.lines import Line2D
import os
import seaborn as sns

set2_colors = sns.color_palette("Set2")
# Add the parent folder to the system path
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')
os.makedirs(plot_folder, exist_ok=True)

def calculate_kl_divergence(prior, posterior):
    epsilon = 1e-10
    prior = np.clip(prior, epsilon, 1 - epsilon)
    posterior = np.clip(posterior, epsilon, 1 - epsilon)
    kl_values = posterior * np.log(posterior / prior) + (1 - posterior) * np.log((1 - posterior) / (1 - prior))
    return np.mean(kl_values)

# Set high-quality figure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "legend.fontsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.figsize": (14, 4),
    "text.usetex": False
})

# Create subplots
fig = plt.figure(figsize=(14, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])

# X-axis for all plots
x = np.linspace(0.01, 0.99, 300)

# --- Panel A: Prior vs. True Distribution ---
prior_a, prior_b = 2, 5
true_a, true_b = 6, 2  # True posterior biased toward 1

prior = beta.pdf(x, prior_a, prior_b)
true = beta.pdf(x, true_a, true_b)

ax0.plot(x, prior, color=set2_colors[2], lw=2)
ax0.plot(x, true, color='black', lw=2, linestyle='--')
ax0.set_title("A: Prior vs True")
ax0.set_xlabel("Probability")
ax0.set_ylabel("Density")
ax0.set_ylim(0, 3)

# --- Panel B: Sampled Posterior Distributions ---
np.random.seed(42)
num_samples = 10
post_ab = np.random.randint(2, 8, size=(num_samples, 2))
posteriors = [beta.pdf(x, a, b) for a, b in post_ab]

# Compute KL divergence from prior
kl_divs = [calculate_kl_divergence(p, prior) for p in posteriors]
best_idx = np.argmax(kl_divs)
best_posterior = posteriors[best_idx]

for i, ps in enumerate(posteriors):
    ax1.plot(x, ps, color='gray', alpha=0.4)
ax1.plot(x, prior, color=set2_colors[2], lw=2)
ax1.set_title("B: Sampled Posteriors")
ax1.set_xlabel("Probability")
ax1.set_ylim(0, 3)
# --- Panel C: Final Selection Based on KL Divergence ---
# Add a hypothetical final posterior (closer to the true distribution)
final_posterior = beta.pdf(x, 5, 2.5)  # more refined estimate

ax2.plot(x, prior, color=set2_colors[2], lw=2)
ax2.plot(x, true, color='black', lw=2, linestyle='--')
ax2.plot(x, best_posterior, color=set2_colors[0], lw=2)
ax2.plot(x, final_posterior, color=set2_colors[1], lw=2)
ax2.set_title("C: Posterior Selection")
ax2.set_xlabel("Probability")
ax2.set_ylim(0, 3)
# --- Unified Legend (updated) ---
custom_lines = [
    Line2D([0], [0], color=set2_colors[2], lw=2, label='Prior Belief'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', label='True Distribution'),
    Line2D([0], [0], color='gray', lw=2, alpha=0.4, label='Sampled Posterior'),
    Line2D([0], [0], color=set2_colors[0], lw=2, label='Selected Posterior (max KL)'),
    Line2D([0], [0], color=set2_colors[1], lw=2, label='Final Posterior after Testing')
]

fig.legend(handles=custom_lines, loc='center right', bbox_to_anchor=(1.28, 0.5), borderaxespad=0.0)
plt.tight_layout(rect=[0, 0, 0.93, 1])
save_path=os.path.join(plot_folder, "Figure1.pdf")
plt.savefig(save_path, bbox_inches='tight')
plt.show()
