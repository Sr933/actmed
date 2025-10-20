import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Configuration -----
datasets = ["hepatitis", "diabetes", "kidney"]
seeds = [0, 100, 42, 123, 456]
models = ["gpt-4o-mini", "gpt-4o"]
methods = ["ACTMED", "Implicit", "Random"]

feature_column_suffix = {
    "ACTMED": "BayesianFeature",
    "Implicit": "ImplicitFeature",
    "Random": "RandomFeature",
}
prediction_columns = ["RiskRandom"]
label_column = "Label"

# ----- Kidney Feature Mapping (for display only) -----
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

# ----- Paths -----
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
results_dir = os.path.join(parent_folder, 'results', 'main')
plot_folder = os.path.join(parent_folder, 'plots')
os.makedirs(plot_folder, exist_ok=True)

# ----- NeurIPS-Style Plot Settings (similar to Figure3.py) -----
sns.set(style="whitegrid", font_scale=1, rc={"axes.titlesize": 18})

# ----- Safe Feature Parser -----
def parse_feature_entry(entry):
    try:
        parsed = ast.literal_eval(entry)
        if isinstance(parsed, list):
            return parsed
        else:
            return [str(parsed)]
    except:
        return [entry]

# -----------------------------------------------------------------------------
# ACCURACY PER SEED
# For each (model, dataset, seed), we gather feature counts and correctness,
# then compute accuracy. We store these per-feature accuracies so that
# we can later compute the mean ± std across seeds.
# -----------------------------------------------------------------------------
# Structure:
# accuracy_by_model[model][dataset][feature] = list of accuracies (one per seed)
# -----------------------------------------------------------------------------
accuracy_by_model = {
    model: {dataset: {} for dataset in datasets}
    for model in models
}

# Loop over each model/dataset/seed to calculate each feature's accuracy in that seed.
for model in models:
    for dataset in datasets:
        # We'll store partial counts to eventually compute per-seed accuracy.
        # Per seed, feature_stats[feat] = [correct_count, total_count].
        feature_stats_per_seed = {seed: {} for seed in seeds}

        for seed in seeds:
            file_path = os.path.join(results_dir, f"{dataset}_experiment_results_{model}_{seed}.csv")
            if not os.path.exists(file_path):
                print(f"Missing file: {file_path}")
                continue

            df = pd.read_csv(file_path)
            # Verify required columns
            if not all(col in df.columns for col in prediction_columns + [label_column, "Try", "PatientID"]):
                continue

            # Round label column and compute correctness
            df[label_column] = df[label_column].round()
            correct_mask = pd.Series(False, index=df.index)
            for pred_col in prediction_columns:
                if pred_col in df.columns:
                    correct_mask |= (df[pred_col].round() == df[label_column])
            df["Correct"] = correct_mask

            # Group by PatientID
            grouped = df.groupby("PatientID")
            for _, patient_df in grouped:
                # Determine patient's outcome:
                # Use try 3 if available; else fallback
                try3 = patient_df[patient_df["Try"] == 3]
                if not try3.empty:
                    patient_correct = try3["Correct"].iloc[0]
                else:
                    patient_correct = patient_df["Correct"].iloc[0]

                # Gather union of features from any try in any method
                patient_features = set()
                for meth in methods:
                    feature_col = feature_column_suffix[meth]
                    for try_idx in [1, 2, 3]:
                        try_row = patient_df[patient_df["Try"] == try_idx]
                        if not try_row.empty and feature_col in try_row.columns:
                            entry = try_row[feature_col].values[0]
                            feats = parse_feature_entry(entry)
                            patient_features.update(feats)

                # Update counters in feature_stats_per_seed
                for feat in patient_features:
                    if feat not in feature_stats_per_seed[seed]:
                        feature_stats_per_seed[seed][feat] = [0, 0]
                    feature_stats_per_seed[seed][feat][1] += 1  # total
                    if patient_correct:
                        feature_stats_per_seed[seed][feat][0] += 1  # correct

        # Now compute per-seed accuracy for each feature
        # Then save it to accuracy_by_model[model][dataset][feature]
        for seed in seeds:
            for feat, (correct, total) in feature_stats_per_seed[seed].items():
                if total > 0:
                    acc = correct / total
                else:
                    acc = np.nan

                if feat not in accuracy_by_model[model][dataset]:
                    accuracy_by_model[model][dataset][feat] = []
                accuracy_by_model[model][dataset][feat].append(acc)

# -----------------------------------------------------------------------------
# Compute mean ± std for each feature, map kidney names, and store sorted results
# -----------------------------------------------------------------------------
ranking_results = {
    model: {dataset: [] for dataset in datasets}
    for model in models
}

for model in models:
    for dataset in datasets:
        feature_accuracies = accuracy_by_model[model][dataset]  # feat -> list of accuracies
        # Remove empty features (i.e. never selected in any seed).
        feature_accuracies = {
            feat: vals for feat, vals in feature_accuracies.items() if len(vals) > 0
        }

        # Compute mean ± std
        stats = {}
        for feat, vals in feature_accuracies.items():
            # Filter out all-NaN. If not all NaN, take non-NaN values:
            cleaned = [v for v in vals if not pd.isna(v)]
            if len(cleaned) == 0:
                # no valid seeds
                avg, std = np.nan, np.nan
            else:
                avg, std = np.mean(cleaned), np.std(cleaned)
            stats[feat] = (avg, std)

        # Map kidney abbreviations if needed
        if dataset == "kidney":
            mapped_stats = {}
            for feat, (avg, std) in stats.items():
                display_name = kidney_feature_descriptions.get(feat, feat)
                mapped_stats[display_name] = (avg, std)
            stats = mapped_stats

        # Sort features by descending mean accuracy
        sorted_items = sorted(stats.items(), key=lambda x: x[1][0], reverse=True)
        # Store in the form (feature, mean, std)
        ranking_results[model][dataset] = sorted_items

# -----------------------------------------------------------------------------
# Plot: 2 rows (one per model) × len(datasets) columns
# With error bars showing ±1 standard deviation.
# -----------------------------------------------------------------------------
n_rows = len(models)
n_cols = len(datasets)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), dpi=150, squeeze=False)

for row_idx, model in enumerate(models):
    for col_idx, dataset in enumerate(datasets):
        ax = axes[row_idx, col_idx]

        ranking = ranking_results[model][dataset]  # list of (feat, (mean, std))
        if not ranking:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            continue

        # Unzip features, means, and stds
        features = [item[0] for item in ranking]
        means = [item[1][0] for item in ranking]
        stds = [item[1][1] for item in ranking]

        # Plot a horizontal bar chart with raw accuracy (0–1 range), plus error bars
        bar_color = sns.color_palette("Set2")[2]
        ax.barh(
            y=features,
            width=means,
            xerr=stds,
            color=bar_color,
            edgecolor='none',
            capsize=5,          # capsize for error bars
            error_kw=dict(
                lw=1.5,         # error bar line width
                capthick=1.5,   # thickness of the caps
                ecolor='black'  # color of the error bars and caps
            )
        )
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{dataset.capitalize()} - {model.upper()}", fontsize=16)
        ax.set_xlabel("Accuracy", fontsize=14)
        ax.set_ylabel("Feature", fontsize=14)
        ax.grid(axis="x", linestyle="--", alpha=0.7)
        ax.invert_yaxis()  # Highest accuracy at top

plt.tight_layout(rect=[0, 0, 1, 0.95])
output_path = os.path.join(plot_folder, "FeatureRankings_ByModel_WithErrorBars.pdf")
plt.savefig(output_path, bbox_inches="tight", dpi=300)
print(f"Saved figure to {output_path}")
plt.show()
