#!/usr/bin/env python3
import sys
import os
import pandas as pd
import random
import re

# Setup paths
parent_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
lib_folder = os.path.join(parent_folder, 'lib')
sys.path.append(lib_folder)
results_folder = os.path.join(parent_folder, 'results_osce2')
os.makedirs(results_folder, exist_ok=True)

# Import OSCE loader and BED model
from osce import OSCECaseLoader, OSCEBEDModel


def run_osce_experiments(model_name, seed):
    """
    Run OSCE experiments: for each case, predict diagnosis probability.
    """
    # Load and preprocess cases
    data_path = os.path.join(parent_folder, 'data', 'agentclinic2')
    loader = OSCECaseLoader(data_path)
    loader.preprocess_data()
    num_cases = len(loader.data)
    

    results = []
    for idx in range(num_cases):
        # Get feature table and label
        feature_table, label, diagnosis = loader.get_item(idx)
        print(f"Processing case {idx + 1}/{num_cases} for diagnosis: {diagnosis}")
        model = OSCEBEDModel(model_name, diagnosis=diagnosis)
        
        # Known and unknown feature names
        known_feats, unknown_feats = loader.return_feature_names(idx)
        print(f"Known features: {known_feats}, Unknown features: {unknown_feats}")
        # Baseline risk using all known features
        risk_all = model.predict_risk(feature_table)
        # Initialize method-specific lists and tables
        bayesian_known, bayesian_unknown = [], unknown_feats.copy()
        random_known, random_unknown = [], unknown_feats.copy()
        implicit_known, implicit_unknown = [], unknown_feats.copy()
        #Generate tables for each method with known features
        bayesian_table = feature_table[known_feats].copy()
        random_table = feature_table[known_feats].copy()
        implicit_table = feature_table[known_feats].copy()
        
        

        # Global best feature selection (top 3)
        try:
            global_names = model.get_best_global_features(unknown_feats, 3)
            # Map suggested feature names to actual feature columns
            mapped_feats = []
            for feat in global_names:
                norm_feat = re.sub(r'[^a-z0-9]', '', feat.lower())
                for actual in unknown_feats:
                    if re.sub(r'[^a-z0-9]', '', actual.lower()) == norm_feat:
                        mapped_feats.append(actual)
                        break
            # Warn about any unmapped features
            unmapped = [f for f in global_names if f not in mapped_feats]
            if unmapped:
                print(f"Warning: Could not map suggested global features: {unmapped}")
            global_feats = mapped_feats[:3]
        except Exception as e:
            print(f"Error in global feature selection: {e}")
            #Take the first 3 unknown features as a fallback
            global_feats = unknown_feats[:3]
        global_table = feature_table[known_feats + global_feats]
        global_best_risk = model.predict_risk(global_table)

        for trie in range(1, 4):
            # Bayesian feature selection
            bayesian_feat, kl_div = model.select_feature_bayesian(bayesian_table, bayesian_unknown)
            if bayesian_feat in bayesian_unknown:
                bayesian_unknown.remove(bayesian_feat)
            else:
                feat=bayesian_unknown.pop()
                bayesian_known.append(feat)
            bayesian_known.append(bayesian_feat)
            bayesian_risk = model.predict_risk(bayesian_table)
            bayesian_table = feature_table[known_feats + bayesian_known]
           
            # Random feature selection
            random_feat = random.choice(random_unknown)
            if random_feat in random_unknown:
                random_unknown.remove(random_feat)
            else:
                random_feat=random_unknown.pop()
            random_known.append(random_feat)
            random_table = feature_table[known_feats + random_known]
            risk_random = model.predict_risk(random_table)

            # Implicit feature selection
            implicit_feat = model.select_feature_implicit(implicit_table, implicit_unknown)
            if implicit_feat in implicit_unknown:
                implicit_unknown.remove(implicit_feat)
            else:
                implicit_feat=implicit_unknown.pop(0)
            implicit_known.append(implicit_feat)
            implicit_selection_risk = model.predict_risk(implicit_table)
            implicit_table = feature_table[known_feats + implicit_known]

            # Append results for the current try
            result_row = {
                'CaseIndex': idx,
                'Clinical_Diagnosis': diagnosis,
                'Label': label,
                'Try': trie,
                'FullRisk': risk_all,
                'BayesianRisk': bayesian_risk,
                'RiskRandom': risk_random,
                'RiskGlobalBest': global_best_risk,
                'ImplicitSelectionRisk': implicit_selection_risk,
                'ImplicitFeature': implicit_feat,
                'RandomFeature': random_feat,
                'BayesianFeature': bayesian_feat,
                'KL_div': kl_div,
                'GlobalBestFeature': global_feats
            }
            results.append(result_row)

        # Save results
        df = pd.DataFrame(results)
        output_file = os.path.join(results_folder, f'osce_experiment_results_{model_name}_{seed}.csv')
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python osceexperiment.py <model_name> <seed>")
        sys.exit(1)

    model_name = sys.argv[1]
    seed = int(sys.argv[2])

    # Set the random seed for reproducibility
    random.seed(seed)

    print(f"Running OSCE experiments with model {model_name} and seed {seed}...")
    run_osce_experiments(model_name, seed)
