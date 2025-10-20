import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

# ----- NeurIPS-Style Plot Settings -----
sns.set(style="whitegrid", font_scale=1.3, rc={"axes.titlesize": 14})

# ----- Feature name mapping (kidney only) -----
kidney_feature_descriptions = {
    "sg": "Specific gravity",
    "al": "Albumin levels in urine",
    "su": "Sugar levels in urine",
    "rbc": "Red blood cells",
    "pc": "Pus cells",
    "pcc": "Pus cell clumps",
    "ba": "Bacteria in urine",
    "bgr": "Blood glucose random",
    "bu": "Blood urea",
    "sc": "Serum creatinine",
    "sod": "Sodium levels",
    "pot": "Potassium levels",
    "hemo": "Hemoglobin levels",
    "pcv": "Packed cell volume",
    "wbcc": "White blood cell count",
    "rbcc": "Red blood cell count",
    "bp": "Blood pressure",
    "rc": "Red blood cell count",
}

# ----- Configuration -----
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
plot_folder = os.path.join(parent_folder, 'plots')
results_dir = os.path.join(parent_folder, 'results', 'main')
os.makedirs(plot_folder, exist_ok=True)

# Datasets, models, seeds, and selection methods to analyze
datasets = ["hepatitis", "diabetes", "kidney"]
models = ["gpt-4o-mini", "gpt-4o"]
seeds = [0, 42, 100, 123, 456]
methods = ["ACTMED", "Implicit", "Random"]

# Create a custom colormap (light blue to light red)
set2_colors = sns.color_palette("Set2")
blue = set2_colors[2]  # light blue
red = set2_colors[1]   # light red
custom_cmap = LinearSegmentedColormap.from_list("blue_red_gradient", [blue, red], N=256)

# ----- Load Data -----
dataframes = {}
for model in models:
    model_data = {}
    for ds in datasets:
        dataset_results = []
        for seed in seeds:
            file_path = os.path.join(results_dir, f"{ds}_experiment_results_{model}_{seed}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df["Seed"] = seed
                dataset_results.append(df)
            else:
                print(f"Warning: File not found for {ds}, seed {seed}, model {model}")
        if dataset_results:
            model_data[ds] = pd.concat(dataset_results, ignore_index=True)
        else:
            model_data[ds] = pd.DataFrame()  # Empty DataFrame if no data found
    dataframes[model] = model_data

# ----- Compute Selection Proportions -----
heatmap_data = {ds: {} for ds in datasets}

for ds in datasets:
    for method in methods:
        method_counts = {}
        totals = {}
        for model in models:
            df_model = dataframes.get(model, {}).get(ds, None)
            if df_model is None or df_model.empty:
                continue

            # Count how many times each feature was selected
            counts = {}
            total = 0
            feature_col = "BayesianFeature" if method == "ACTMED" else f"{method}Feature"
            for idx, row in df_model.iterrows():
                value = row.get(feature_col)
                if pd.notna(value):
                    total += 1
                    try:
                        val = eval(value)
                        if isinstance(val, (list, set)):
                            for feat in val:
                                counts[feat] = counts.get(feat, 0) + 1
                        else:
                            counts[val] = counts.get(val, 0) + 1
                    except Exception:
                        feat = value
                        counts[feat] = counts.get(feat, 0) + 1
            method_counts[model.upper()] = counts
            totals[model.upper()] = total

        # Make a DataFrame of selection percentages
        all_features = set()
        for cnt in method_counts.values():
            all_features |= cnt.keys()
        all_features = sorted(all_features)

        df_heat = pd.DataFrame(index=all_features, columns=[m.upper() for m in models])
        for model in [m.upper() for m in models]:
            for feat in all_features:
                count = method_counts.get(model, {}).get(feat, 0)
                total = totals.get(model, 0)
                df_heat.loc[feat, model] = (count / total * 100) if total > 0 else np.nan

        heatmap_data[ds][method] = df_heat

# ----- Plot Heatmaps -----
fig_width, fig_height = 11.69, 8.27  # A4 landscape
n_rows = len(datasets)
n_cols = len(methods)

fig = plt.figure(figsize=(fig_width, fig_height))
gs = GridSpec(nrows=n_rows + 1, ncols=n_cols, height_ratios=[0.15] + [1] * n_rows)
cax = fig.add_subplot(gs[0, :])  # shared colorbar axis at the top
axs = [[fig.add_subplot(gs[i+1, j]) for j in range(n_cols)] for i in range(n_rows)]

vmin, vmax = 0, 100

for row_idx, ds in enumerate(datasets):
    for col_idx, method in enumerate(methods):
        ax = axs[row_idx][col_idx]
        df_heat = heatmap_data[ds].get(method, None)
        if df_heat is None or df_heat.empty:
            ax.text(0.5, 0.5, f"No data for {ds} - {method}", ha='center', va='center')
            ax.axis("off")
            continue

        # Ensure models in correct order and sort features
        df_heat = df_heat.reindex(columns=[m.upper() for m in models])
        df_heat = df_heat.sort_index()

        # Rename kidney features for interpretability
        if ds == "kidney":
            df_heat.index = [kidney_feature_descriptions.get(f, f) for f in df_heat.index]

        sns.heatmap(
            df_heat.astype(float),
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap=custom_cmap,
            vmin=0,
            vmax=33,
            cbar=(row_idx == 0 and col_idx == len(methods) - 1),
            cbar_ax=cax if (row_idx == 0 and col_idx == len(methods) - 1) else None,
            annot_kws={"size": 8},
            linewidths=0.5,
            xticklabels=True,  # ← Changed to True to show model names
            yticklabels=True,
            cbar_kws={
                "label": "Selection Percentage (%)",
                "shrink": 0.08,
                "aspect": 10,
                "pad": 0.02,
                "orientation": "horizontal"
            },
        )

        ax.set_title(f"{ds.capitalize()} - {method}", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

# Style colorbar and layout
cax.tick_params(labelsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.96])

output_path = os.path.join(plot_folder, "all_datasets_feature_selection_heatmaps_NeurIPS.pdf")
with PdfPages(output_path) as pdf:
    pdf.savefig(fig, dpi=300, bbox_inches="tight")

plt.show()
print(f"Saved combined heatmap to: {output_path}")
