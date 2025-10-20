import os
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_folder = os.path.join(parent_folder, 'lib')
sys.path.append(lib_folder)
plot_folder = os.path.join(parent_folder, 'plots')
os.makedirs(plot_folder, exist_ok=True)  # Ensure the plots folder exists

# Import your dataset classes (adjust if needed)
from datasets import (
    KidneyDataset,
    DiabetesDataset,
    HepatitisDataset
)

def load_preprocessed_data(dataset_name):
    """
    Returns the preprocessed DataFrame using the matching Dataset subclass.
    """
    base_data_path = os.path.join(parent_folder, 'data')
    if dataset_name == "kidney":
        csv_file = os.path.join(base_data_path, "kidney", "ckd.csv")
        dataset_obj = KidneyDataset(csv_file)
    elif dataset_name == "diabetes":
        csv_file = os.path.join(base_data_path, "diabetes", "diabetes.csv")
        dataset_obj = DiabetesDataset(csv_file)
    elif dataset_name == "hepatitis":
        csv_file = os.path.join(base_data_path, "hepatitis", "Hepatitis_subset.csv")
        dataset_obj = HepatitisDataset(csv_file)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset_obj.preprocess_data()
    return dataset_obj.data

def compute_feature_stats(df):
    """
    Returns a dict of {feature -> (min_val, max_val)} for numeric columns.
    """
    numeric_df = df
    overall_stats = {}
    for col in numeric_df.columns:
        try:
            numeric_df[col] = numeric_df[col].astype(float)
            col_min = numeric_df[col].min()
            col_max = numeric_df[col].max()
            overall_stats[col] = (col_min, col_max)
        except ValueError:
            continue
        
    return overall_stats

def analyze_sampling_results():
    # Feature descriptions for the kidney dataset
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

    datasets = ["kidney", "diabetes", "hepatitis"]
    models = ["gpt-4o", "gpt-4o-mini"]
    seeds = [0, 42, 100, 123, 456]
    results_folder = os.path.join(parent_folder, 'results', 'sampling')

    all_rows = []

    for ds in datasets:
        # Load preprocessed data from the custom Dataset class
        original_df = load_preprocessed_data(ds)
        original_df.columns = original_df.columns.astype(str)  # Ensure column names are strings
        overall_stats = compute_feature_stats(original_df)

        for model in models:
            for seed in seeds:
                filename = f"{ds}_sampling_results_{model}_{seed}.csv"
                filepath = os.path.join(results_folder, filename)
                if not os.path.isfile(filepath):
                    print(f"File not found: {filepath}, skipping.")
                    continue

                df = pd.read_csv(filepath)
                if df.empty:
                    print(f"Empty DataFrame: {filepath}, skipping.")
                    continue

                sample_cols = [c for c in df.columns if c.startswith("Sample")]

                for _, row in df.iterrows():
                    feature = row.get("FeatureToSample", None)
                    if feature not in overall_stats:
                        continue
                    if feature == "Insulin":
                        continue

                    patient_id = row.get("PatientID")
                    if patient_id is None:
                        continue

                    try:
                        actual_value = float(original_df.iloc[patient_id][feature])
                    except (ValueError, IndexError, KeyError):
                        continue

                    feat_min, feat_max = overall_stats[feature]
                    numeric_vals = []
                    for col in sample_cols:
                        val_str = row.get(col, None)
                        if val_str is None:
                            continue
                        try:
                            val = float(val_str)
                            numeric_vals.append(val)
                        except ValueError:
                            pass

                    if not numeric_vals:
                        continue

                    in_range_count = sum(feat_min <= x <= feat_max for x in numeric_vals)
                    proportion_in_range = in_range_count / len(numeric_vals)

                    for i in range(1, len(numeric_vals) + 1):
                        subset_vals = numeric_vals[:i]
                        subset_mean = np.mean(subset_vals)

                        if min(subset_vals) <= actual_value <= max(subset_vals):
                            mean_abs_diff = 0
                        else:
                            if actual_value == 0:
                                mean_abs_diff = np.nan
                            else:
                                mean_abs_diff = (
                                    np.min([abs(x - actual_value) for x in subset_vals])
                                    / (actual_value + max(subset_vals))
                                    / 2
                                    * 100
                                )

                        all_rows.append({
                            "Dataset": ds,
                            "Model": model,
                            "Seed": seed,
                            "Feature": feature,
                            "NumSamples": i,
                            "SampleValue": subset_mean,
                            "ActualValue": actual_value,
                            "ProportionInRange": proportion_in_range,
                            "MeanSampleAbsDiff": mean_abs_diff
                        })

    if not all_rows:
        print("No valid data found.")
        return

    results_df = pd.DataFrame(all_rows)

    # Pick the custom colors for "gpt-4o" and "gpt-4o-mini":
    set2_colors = sns.color_palette("Set2")
    colors = {
        "GPT-4O": set2_colors[2],       # color index 2
        "GPT-4O-MINI": set2_colors[1],  # color index 1
    }

    # Set NeurIPS-quality plot settings
    sns.set(style="whitegrid", font_scale=1.5, rc={"figure.figsize": (10, 6)})

    # 1) Separate Subplots for Proportion in Range and Mean Absolute Difference
    combined_stats = results_df.groupby(["Dataset", "Model", "Seed"], as_index=False).agg({
        "ProportionInRange": "mean",
        "MeanSampleAbsDiff": "mean"
    })
    # Convert datasets to uppercase for neat display
    combined_stats["Dataset"] = combined_stats["Dataset"].str.capitalize()
    # Convert model to uppercase
    combined_stats["Model"] = combined_stats["Model"].str.upper()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    print(combined_stats.groupby("Model")["ProportionInRange"].mean())

    # Proportion in Range Plot
    sns.barplot(
        data=combined_stats,
        x="Dataset",
        y="ProportionInRange",
        hue="Model",
        palette=colors,
        hue_order=["GPT-4O", "GPT-4O-MINI"],
        errorbar="se",
        ax=axes[0],
        edgecolor='none',
        errwidth=1.5,
        capsize=0.1,
    )
    axes[0].set_ylim(0.5, 1)
    axes[0].set_ylabel("Proportion in Range")
    axes[0].set_title("Proportion in Range")
    axes[0].legend_.remove() 

    # Mean Absolute Difference Plot
    sns.barplot(
        data=combined_stats,
        x="Dataset",
        y="MeanSampleAbsDiff",
        hue="Model",
        palette=colors,
        hue_order=["GPT-4O", "GPT-4O-MINI"],
        errorbar="se",
        ax=axes[1],
        edgecolor='none',
        errwidth=1.5,
        capsize=0.1,
    )
    axes[1].set_ylabel("Mean Absolute Error (%)")
    axes[1].set_ylim(0, 10)
    axes[1].set_title("Mean Absolute Error (MAE)")
    axes[1].legend_.remove()

    # Consolidated legend above the plots
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    plt.savefig(os.path.join(plot_folder, "separate_proportion_and_mad.pdf"), format="pdf")
    plt.show()

    # 2) Heatmap for MeanSampleAbsDiff
    error_stats = results_df.groupby(["Feature", "Model", "Dataset"], as_index=False)["MeanSampleAbsDiff"].mean()
    error_stats["Model"] = error_stats["Model"].str.upper()

    def map_feature_name(row):
        if row["Dataset"] == "kidney" and row["Feature"] in kidney_feature_descriptions:
            return f"{kidney_feature_descriptions[row['Feature']]} ({row['Dataset']})"
        else:
            return f"{row['Feature']} ({row['Dataset']})"

    error_stats["Feature"] = error_stats.apply(map_feature_name, axis=1)
    error_pivot = error_stats.pivot(index="Feature", columns="Model", values="MeanSampleAbsDiff")

    plt.figure(figsize=(12, 10))
    # Create a custom gradient from set2_colors[2] (light blue) to set2_colors[1] (light red)
    blue = set2_colors[2]
    red = set2_colors[1]
    custom_cmap = LinearSegmentedColormap.from_list("blue_red_gradient", [blue, red], N=256)

    sns.heatmap(
        error_pivot,
        annot=True,
        fmt=".2f",
        cmap=custom_cmap,
        cbar_kws={"label": "Mean Absolute Error (%)"},
        vmin=0,
        vmax=25,
        edgecolor='none',
    )
    plt.xlabel("Model")
    plt.ylabel("Feature (Dataset)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plot_folder, "heatmap_mean_absolute_error.pdf"), format="pdf")
    plt.show()

    # 3) Individual Distribution Plots
    sns.set(style="whitegrid", font_scale=1.6, rc={
        "figure.figsize": (18, 10),
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14
    })

    for ds in results_df["Dataset"].unique():
        ds_subset = results_df[results_df["Dataset"] == ds]
        if ds == "kidney":
            ds_subset["Feature"] = ds_subset["Feature"].map(kidney_feature_descriptions)

        g = sns.FacetGrid(
            ds_subset,
            col="Feature",
            col_wrap=3,
            sharex=False,
            sharey=False,
            height=4,
            aspect=1.2,
            margin_titles=True
        )
        
        # Use just one color (light blue or red) is fine, but let's keep the default from palette
        g.map(sns.histplot, "SampleValue", bins=20, kde=False, color=set2_colors[2])

        g.set_titles(col_template="{col_name}", size=15)
        g.set_axis_labels("Sample Value", "Frequency")

        for ax in g.axes.flat:
            ax.tick_params(labelsize=12)

        plt.subplots_adjust(top=0.9, hspace=0.4)
        g.fig.suptitle(f"Sample Value Distributions for {ds.capitalize()}", fontsize=20)

        output_path = os.path.join(plot_folder, f"{ds}_sample_value_distributions.pdf")
        g.savefig(output_path, format="pdf", bbox_inches="tight")
        plt.close()

    # Pass results_df to the plot_mae_vs_samples function
    plot_mae_vs_samples(results_df, plot_folder, colors)

def plot_mae_vs_samples(results_df, plot_folder, colors):
    """
    Creates a 3-panel plot showing how the Mean Absolute Error (MAE) changes
    as a function of the number of samples for each dataset, with error bars for seeds.
    """
    mae_stats = results_df[["Dataset", "Model", "Seed", "NumSamples", "MeanSampleAbsDiff"]]
    mae_stats["Dataset"] = mae_stats["Dataset"].str.capitalize()
    mae_stats["Model"] = mae_stats["Model"].str.upper()

    sns.set(style="whitegrid", font_scale=1.7)
    g = sns.FacetGrid(
        mae_stats,
        col="Dataset",
        hue="Model",
        hue_order=["GPT-4O", "GPT-4O-MINI"],
        sharey=True,
        sharex=True,
        height=5,
        aspect=1.2,
        palette=colors   # Use the custom color dictionary
    )
    g.map(
        sns.lineplot,
        "NumSamples",
        "MeanSampleAbsDiff",
        errorbar="se",  # Error bars across seeds
        marker="o"
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("Number of Samples", "Mean Absolute Error (%)")
    g.add_legend(title="Model")
    g.tight_layout()

    plt.savefig(os.path.join(plot_folder, "mae_vs_samples.pdf"), format="pdf")
    plt.show()

if __name__ == "__main__":
    analyze_sampling_results()