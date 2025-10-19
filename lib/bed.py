import numpy as np
from model import chat_gpt, llama_chat
from helperfunctions import dataframe_to_markdown, hepatitis_clinical_vignette, diabetes_clinical_vignette, stroke_features_to_clinical_vignette, kidney_clinical_vignette, cirrhosis_clinical_vignette, heart_clinical_vignette
from timeseries import process_patient_df
class BEDModel:
    """
    General BEDModel base class providing common functionality.
    The sampling, risk-prediction, implicit feature selection, and global feature selection prompts are adjustable via instance attributes.
    """
    def __init__(self, model_name, sampling_prompt_template=None, risk_prompt_template=None,
                 implicit_selection_prompt_template=None, global_features_prompt_template=None):
        self.model = None
        self.select_model(model_name)

        # Default sampling prompt template if not provided.
        self.sampling_prompt_template = sampling_prompt_template 
        
        # Default risk prediction prompt template if not provided.
        self.risk_prompt_template = risk_prompt_template 

        # Default implicit feature selection prompt template.
        self.implicit_selection_prompt_template = implicit_selection_prompt_template 

        # Default global features selection prompt template.
        self.global_features_prompt_template = global_features_prompt_template 

        # Subclasses should define their own REFERENCE_TABLE.
        self.REFERENCE_TABLE = {}

    def select_model(self, model_name):
        if model_name == 'gpt-4o-mini':
            self.model = chat_gpt
            self.model_name = 'gpt-4o-mini'
        elif model_name == 'gpt-4.1-mini':
            self.model = chat_gpt
            self.model_name = 'gpt-4.1-mini'
        elif model_name == 'gpt-4o':
            self.model = chat_gpt
            self.model_name = 'gpt-4o'
        elif model_name == 'TinyLlama/TinyLlama-1.1B-Chat-v1.0':
            self.model = llama_chat
            self.model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        else:
            raise ValueError("Model not supported")

    def sample_random_variable(self, known_features, feature_to_sample):
        """
        Uses the adjustable sampling prompt to request the model to sample a candidate value.
        """
        extra_info=("\n\nIMPORTANT: Under NO circumstances provide explanations, commentary, or text beyond the single numeric float or string requested."
            " The response MUST be parseable strictly as a float, e.g., 0.512, with no extra words. If a string is requested no float is required.")
        ref_info = self.REFERENCE_TABLE.get(feature_to_sample, {})
        # Format the sampling prompt with the desired feature and its reference information.
        base_instruction = self.sampling_prompt_template.format(feature=feature_to_sample, ref=ref_info)
        
        known_md = self.format_known_data(known_features)
        
        user_prompt = base_instruction + "\n\n" + known_md + "\n\n" + extra_info
        #print(f"Sampling prompt: {user_prompt}")
        response = self.model(user_prompt=user_prompt, model_name=self.model_name)
        try:
            value = float(response.strip())
        except Exception as e:
            value=response
        return value

    def predict_risk(self, known_feature_table, extra_info=""):
        """
        Uses the adjustable risk prediction prompt to request a risk estimate.
        """
        known_str = self.format_known_data(known_feature_table)
        if isinstance(extra_info, list):
            extra_info = ", ".join(extra_info)  # Convert list to comma-separated string
        user_input = self.risk_prompt_template + "\n\n" + known_str + "\n\n" + extra_info
        #print(f"Risk prediction prompt: {user_input}")
        response = self.model(user_prompt=user_input, model_name=self.model_name)
        try:
            risk = float(response.strip())
        except Exception as e:
            print("Error parsing model response, using raw response as float:", response)
            risk = 0.5
        return risk

    def calculate_kl_divergence(self, prior, posterior):
        epsilon = 1e-10
        prior = np.clip(prior, epsilon, 1 - epsilon)
        posterior = np.clip(posterior, epsilon, 1 - epsilon)
        kl_values = posterior * np.log(posterior / prior) + (1 - posterior) * np.log((1 - posterior) / (1 - prior))
        return np.mean(kl_values)

    def select_feature(self, known_features, unknown_features, N=10):
        prior_predictions = [self.predict_risk(known_features) for _ in range(N)]
        kl_divergences = []
        for feature in unknown_features:
            sampled_values = [self.sample_random_variable(known_features, feature) for _ in range(N)]
            posteriors = [self.predict_risk(known_features, f"{feature}: {sample}") for sample in sampled_values]
            kl_div = self.calculate_kl_divergence(prior_predictions, posteriors)
            kl_divergences.append(kl_div)
            print(f"Feature: {feature}, KL Divergence: {kl_div}")
        max_kl_index = np.argmax(kl_divergences)
        feature_to_test = unknown_features[max_kl_index]
        max_kl_div = kl_divergences[max_kl_index]
        print(f"Optimal feature to query: {feature_to_test} with KL Divergence: {kl_divergences[max_kl_index]}")
        return feature_to_test, max_kl_div

    def format_known_data(self, known_features):
        """
        Format the known features into a string for the prompt.
        Subclasses can override this method to use a different formatting function.
        """
        return dataframe_to_markdown(known_features)
        

    def select_feature_implicit(self, known_features, unknown_features):
        """
        Ask the model to pick the next feature to sample based solely on the known features.
        The prompt and data formatting can be customized by subclasses.
        """
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        prompt = self.implicit_selection_prompt_template.format(
            known_data=known_data,
            unknown_features=unknown_str
        )
        #print(f"Implicit feature selection prompt: {prompt}")
        response = self.model(user_prompt=prompt, model_name=self.model_name)
        best_feature = response.strip()
        # Fallback: if the returned feature isn't in our list, try matching by checking if any unknown feature's name is contained in the response.
        if best_feature not in unknown_features:
            for feat in unknown_features:
                if feat.lower() in best_feature.lower():
                    best_feature = feat
                    break
        print(f"Implicitly selected feature: {best_feature}")
        return best_feature

    def get_best_global_features(self, all_features, n):
        """
        Query the model for the best n global features.
        """
        all_features_str = ", ".join(all_features)
        prompt = self.global_features_prompt_template.format(
            all_features_str,
            n
        )
        response = self.model(user_prompt=prompt, model_name=self.model_name)
        try:
            best_features = eval(response.strip())
            if not isinstance(best_features, list):
                best_features = [feat.strip() for feat in response.split(',') if feat.strip()]
        except Exception as e:
            print("Error parsing model response, using raw response as list:", response)
            best_features = [response.strip()]
        print("Suggested best features:", best_features)
        return best_features
    
    
class HepatitisBEDModel(BEDModel):
    """
    HepatitisBEDModel predicts the risk of hepatitis based on liver-related clinical data.
    """

    def __init__(self, model_name, dataframe, N=10):
        super().__init__(model_name)
        self.dataframe = dataframe
        self.N = N
        self.REFERENCE_TABLE = {
            "Age": "Patient age in years.",
            "Sex": "Patient sex; 'm' for male and 'f' for female.",
            "ALB": "Albumin level in g/L.",
            "ALP": "Alkaline phosphatase level (IU/L).",
            "ALT": "Alanine transaminase level (IU/L).",
            "AST": "Aspartate transaminase level (IU/L).",
            "BIL": "Bilirubin level in µmol/L.",
            "CHE": "Cholinesterase level in kU/L.",
            "CHOL": "Cholesterol level in mmol/L.",
            "CREA": "Creatinine level in µmol/L.",
            "GGT": "Gamma-glutamyl transferase level (IU/L).",
            "PROT": "Total protein level in g/L."
        }

    def predict_risk(self, known_feature_table, extra_info=""):
        self.risk_prompt_template = (
            "You are an expert hepatologist. Based on the following clinical data and the patient's history, "
            "please provide an estimate of the patient's risk of being infected with hepatitis C as a single number between 0 and 1. "
            "Consider key laboratory markers and other pertinent values. "
            "When these values indicate liver inflammation or damage— "
            "assign a number closer to 1, indicating a higher probability of hepatitis C infection. "
            "If the laboratory values are within normal ranges, assign a value closer to 0. Return only the number that can be converted to a Python float, without any additional commentary."
        )
        return super().predict_risk(known_feature_table, extra_info)

    def sample_random_variable(self, known_features, feature_to_sample):
        ref_info = self.REFERENCE_TABLE.get(feature_to_sample, "No description available.")
        self.sampling_prompt_template = (
            f"You are an expert hepatologist. Based on the following clinical data and the patient's history, "
            f"please simulate a random draw from the full range of clinically plausible values for {feature_to_sample}. "
            f"Consider the possible range as described: {ref_info}. "
            "Ensure that the value you return is realistic and reflects clinical variability. "
            "Avoid returning the same value repeatedly across multiple draws, and ensure the value varies as if sampled from a plausible distribution. "
            "Introduce randomness by considering edge cases, typical values, and outliers within the plausible range. "
            "Return your answer as a single numeric value that can be converted to a Python float, without any additional commentary. "
            "IMPORTANT: Assume that the patient may or may not have hepatitis C, and your sampling should reflect that uncertainty."
        )
        return super().sample_random_variable(known_features, feature_to_sample)

    def select_feature_implicit(self, known_features, unknown_features):
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        self.implicit_selection_prompt_template = (
            "You are an expert hepatologist. Based solely on the following known clinical data, "
            "determine which additional feature from the list below would be the most informative to sample next for diagnosing hepatitis.\n\n"
            f"Known Data:\n{known_data}\n\n"
            f"Unknown Features: {unknown_str}\n\n"
            "Return only the name of the feature as a string, without any additional commentary."
        )
        return super().select_feature_implicit(known_features, unknown_features)

    def get_best_global_features(self, all_features, n):
        all_features_str = ", ".join(all_features)
        self.global_features_prompt_template = (
            f"You are an expert hepatologist. Based on the following list of features: {all_features_str}, "
            f"please indicate which {n} feature{'s' if n != 1 else ''} you believe are the most informative and critical for diagnosing hepatitis. "
            f"Return your answer as a Python list of exactly {n} feature name{'s' if n != 1 else ''} "
            "(for example, if n is 1, return ['ALT']; if n is 2, return ['ALT', 'AST']), without any additional commentary."
        )
        return super().get_best_global_features(all_features, n)

    def format_known_data(self, known_features):
        return hepatitis_clinical_vignette(known_features)
    
class DiabetesBEDModel(BEDModel):
    """
    DiabetesBEDModel predicts the risk of diabetes based on clinical data.
    """

    def __init__(self, model_name, dataframe, N=10):
        super().__init__(model_name)
        self.dataframe = dataframe
        self.N = N
        self.REFERENCE_TABLE = {
            "Pregnancies": "Number of times pregnant e.g. 0-10.",
            "Glucose": "Plasma glucose concentration a 2 hours in an oral glucose tolerance test (mg/dL).",
            "BloodPressure": "Diastolic blood pressure (mm Hg).",
            "SkinThickness": "Triceps skinfold thickness (mm).",
            "Insulin": "2-Hour serum insulin (10^(-6) U/ml).",
            "BMI": "Body mass index (kg/(m^2)).",
            "DiabetesPedigreeFunction": "Diabetes pedigree function.",
            "Age": "Age (years)."
        }

    def predict_risk(self, known_feature_table, extra_info=""):
        self.risk_prompt_template = (
            "You are an expert endocrinologist. Based on the following clinical data and the patient's history, "
            "provide an estimate of the patient's risk of diabetes as a single number between 0 and 1. "
            "It is known that all patients are females at least 21 years old of Pima Indian heritage. "
            "Focus on key markers. Assign a value closer to 1 if the data indicate high risk, and closer to 0 if within normal limits. "
            "Return only the number that can be converted to a Python float, without any additional commentary."
        )
        return super().predict_risk(known_feature_table, extra_info)

    def sample_random_variable(self, known_features, feature_to_sample):
        ref_info = self.REFERENCE_TABLE.get(feature_to_sample, "No description available.")
        self.sampling_prompt_template = (
            f"You are an expert endocrinologist. Based on the following clinical data and the patient's history, "
            f"please simulate a random draw from the full range of clinically plausible values for {feature_to_sample}. "
            f"Consider the following unit for the sampled value: {ref_info}. "
            "Ensure that the value you return is realistic and reflects clinical variability. "
            "Avoid returning the same value repeatedly across multiple draws, and ensure the value varies as if sampled from a plausible distribution. "
            "Introduce randomness by considering edge cases, typical values, and outliers within the plausible range. "
            "Return your answer as a single numeric value that can be converted to a Python float with no units or additional commentary. "
            "IMPORTANT: Assume that the patient may or may not have diabetes, and your sampling should reflect that uncertainty."
        )
        return super().sample_random_variable(known_features, feature_to_sample)

    def select_feature_implicit(self, known_features, unknown_features):
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        self.implicit_selection_prompt_template = (
            "You are an expert endocrinologist. Based solely on the following known clinical data, "
            "determine which additional feature from the list below would be the most informative to sample next for diagnosing diabetes.\n\n"
            f"Known Data:\n{known_data}\n\n"
            f"Unknown Features: {unknown_str}\n\n"
            "Return only the name of the feature as a string, without any additional commentary."
        )
        return super().select_feature_implicit(known_features, unknown_features)

    def get_best_global_features(self, all_features, n):
        all_features_str = ", ".join(all_features)
        self.global_features_prompt_template = (
            f"You are an expert endocrinologist. Based on the following list of features: {all_features_str}, "
            f"please indicate which {n} feature{'s' if n != 1 else ''} you believe are the most informative and critical for diagnosing diabetes. "
            f"Return your answer as a Python list of exactly {n} feature name{'s' if n != 1 else ''} "
            "(for example, if n is 1, return ['Glucose']; if n is 2, return ['Glucose', 'BMI']), without any additional commentary."
        )
        return super().get_best_global_features(all_features, n)
    
    def format_known_data(self, known_features):
        return diabetes_clinical_vignette(known_features)


class KidneyBEDModel(BEDModel):
    def __init__(self, model_name, dataframe, N=10):
        """
        Initialize the KidneyBEDModel with a model, a dataframe, and the number of prior samples (N).
        """
        super().__init__(model_name)
        self.dataframe = dataframe
        self.N = N  # Number of prior samples
        self.REFERENCE_TABLE = {
            "age": "Patient age in years.",
            "bp": "Diastolic Blood pressure in mm/Hg. ",
            "sg": "Specific gravity of urine (categorical).",
            "al": "Albumin levels in urine (categorical).",
            "su": "Sugar levels in urine (categorical).",
            "rbc": "Red blood cells (binary: normal or abnormal).",
            "pc": "Pus cells (binary: normal or abnormal).",
            "pcc": "Pus cell clumps (binary: 1 or 0 no in between).",
            "ba": "Bacteria in urine (binary: 1 or 0 no in between).",
            "bgr": "Blood glucose random in mg/dL.",
            "sc": "Serum creatinine in mg/dL (continuous).",
            "sod": "Sodium levels in mEq/L.",
            "pot": "Potassium levels in mEq/L (continuous).",
            "hemo": "Hemoglobin levels in g/dL (continuous).",
            "pcv": "Packed cell volume (integer).",
            "wc": "White blood cell count in individual cells/mm^3.",
            "rc": "Red blood cell count in millions/mm^3 (continuous).",
            "htn": "Hypertension (binary: yes or no).",
            "dm": "Diabetes mellitus (binary: 1 or 0 no in between).",
            "cad": "Coronary artery disease (binary: 1 or 0 no in between).",
            "appet": "Appetite (binary: 1 or 0 no in between).",
            "pe": "Pedal edema (binary: 1 or 0 no in between).",
            "ane": "Anemia (binary: 1 or 0 no in between).",
            "class": "Target variable: chronic kidney disease (binary: ckd or not ckd)."
        }

    def predict_risk(self, known_feature_table, extra_info=""):
        """
        Predict the risk of chronic kidney disease using the clinical data from the known feature table.
        """
        self.risk_prompt_template = (
            "You are an expert nephrologist. Based on the following clinical data and the patient's history, "
            "provide an estimate of the patient having chronic kidney disease as a single number between 0 and 1. "
            "Consider key laboratory markers and other pertinent values. "
            "When these values indicate kidney disease or damage— "
            "assign a number closer to 1, indicating a higher probability of chronic kidney disease. "
            "If the laboratory values are within normal ranges, assign a value closer to 0."
            "Return only the number that can be converted to a Python float, without any additional commentary."
        )
        return super().predict_risk(known_feature_table, extra_info)

    def format_known_data(self, known_features):
        """
        Format the known features into a clinical vignette for kidney disease.
        """
        return kidney_clinical_vignette(known_features)

    def select_feature_implicit(self, known_features, unknown_features):
        """
        Ask the model to pick the next feature to sample based solely on the known features.
        """
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        self.implicit_selection_prompt_template = (
            "You are an expert nephrologist. Based solely on the following known clinical data, "
            "determine which additional feature from the list below would be the most informative to sample next for diagnosing chronic kidney disease.\n\n"
            f"Known Data:\n{known_data}\n\n"
            f"Unknown Features: {unknown_str}\n\n"
            "Return only the name of the feature striclty in the form shown in the list as a string, without any additional commentary."
        )
        return super().select_feature_implicit(known_features, unknown_features)

    def get_best_global_features(self, all_features, n):
        """
        Query the model for the best n global features for diagnosing chronic kidney disease.
        """
        all_features_str = ", ".join(all_features)
        self.global_features_prompt_template = (
            f"You are an expert nephrologist. Based on the following list of features: {all_features_str}, "
            f"please indicate which {n} feature{'s' if n != 1 else ''} you believe are the most informative and critical for diagnosing chronic kidney disease. "
            "Return your answer as a Python list of exactly {n} feature name{'s' if n != 1 else ''} "
            "(for example, if n is 1, return ['age']; if n is 2, return ['age', 'bp']), without any additional commentary."
        )
        return super().get_best_global_features(all_features, n)
    def sample_random_variable(self, known_features, feature_to_sample):
        """
        Requests the model to sample a value for a stroke-specific feature.
        """
        ref_info = self.REFERENCE_TABLE.get(feature_to_sample, "No description available.")
        self.sampling_prompt_template = (
            f"You are an expert nephrologist. Based on the following clinical data and the patient's history, "
            f"please simulate a random draw from the full range of clinically plausible values for {feature_to_sample}. "
            f"The value should not simply be the average or a central tendency, but should vary as if sampled at random from a realistic distribution. "
            f"Consider the following description: {ref_info}. "
            "Avoid returning the same value repeatedly across multiple draws, and ensure the value varies as if sampled from a plausible distribution. "
            "Introduce randomness by considering edge cases, typical values, and outliers within the plausible range. "
            "Return your answer as a single numeric value that can be converted to a Python float with no additional commentary or units."
            "IMPORTANT: Assume that the patient may or may not have chronic kidney disease, and your sampling should reflect that uncertainty."
        )
        return super().sample_random_variable(known_features, feature_to_sample)

    
class HepatitisEntropyBEDModel(HepatitisBEDModel):
    """
    HepatitisEntropyBEDModel predicts the risk of hepatitis based on liver-related clinical data
    and selects the test causing the greatest reduction in entropy.
    """

    def calculate_entropy(self, probabilities):
        """
        Calculate entropy for a list of probabilities.
        """
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
        return -np.sum(probabilities * np.log(probabilities))

    def select_feature(self, known_features, unknown_features, N=10):
        prior_predictions = [self.predict_risk(known_features) for _ in range(N)]
        prior_entropy = self.calculate_entropy(prior_predictions)

        entropy_reductions = []
        for feature in unknown_features:
            sampled_values = [self.sample_random_variable(known_features, feature) for _ in range(N)]
            posteriors = [self.predict_risk(known_features, f"{feature}: {sample}") for sample in sampled_values]
            posterior_entropy = self.calculate_entropy(posteriors)
            entropy_reduction = prior_entropy - posterior_entropy
            entropy_reductions.append(entropy_reduction)
            print(f"Feature: {feature}, Entropy Reduction: {entropy_reduction}")

        max_entropy_index = np.argmax(entropy_reductions)
        feature_to_test = unknown_features[max_entropy_index]
        max_entropy_reduction = entropy_reductions[max_entropy_index]
        print(f"Optimal feature to query: {feature_to_test} with Entropy Reduction: {max_entropy_reduction}")
        return feature_to_test, max_entropy_reduction

class DiabetesEntropyBEDModel(DiabetesBEDModel):
    """
    DiabetesEntropyBEDModel predicts diabetes risk and selects features by entropy reduction.
    """
    def calculate_entropy(self, probabilities):
        epsilon = 1e-10
        probs = np.clip(probabilities, epsilon, 1 - epsilon)
        return -np.sum(probs * np.log(probs))

    def select_feature(self, known_features, unknown_features, N=10):
        # Prior entropy
        prior_preds = [self.predict_risk(known_features) for _ in range(N)]
        prior_entropy = self.calculate_entropy(prior_preds)
        # Evaluate entropy reduction for each feature
        reductions = []
        for feature in unknown_features:
            samples = [self.sample_random_variable(known_features, feature) for _ in range(N)]
            post_preds = [self.predict_risk(known_features, f"{feature}: {s}") for s in samples]
            post_entropy = self.calculate_entropy(post_preds)
            reduction = prior_entropy - post_entropy
            reductions.append(reduction)
            print(f"Feature: {feature}, Entropy Reduction: {reduction}")
        idx = np.argmax(reductions)
        best = unknown_features[idx]
        best_red = reductions[idx]
        print(f"Optimal feature to query: {best} with Entropy Reduction: {best_red}")
        return best, best_red

class KidneyEntropyBEDModel(KidneyBEDModel):
    """
    KidneyEntropyBEDModel predicts kidney disease risk and selects features by entropy reduction.
    """
    def calculate_entropy(self, probabilities):
        epsilon = 1e-10
        probs = np.clip(probabilities, epsilon, 1 - epsilon)
        return -np.sum(probs * np.log(probs))

    def select_feature(self, known_features, unknown_features, N=10):
        prior_preds = [self.predict_risk(known_features) for _ in range(N)]
        prior_entropy = self.calculate_entropy(prior_preds)
        reductions = []
        for feature in unknown_features:
            samples = [self.sample_random_variable(known_features, feature) for _ in range(N)]
            post_preds = [self.predict_risk(known_features, f"{feature}: {s}") for s in samples]
            post_entropy = self.calculate_entropy(post_preds)
            reduction = prior_entropy - post_entropy
            reductions.append(reduction)
            print(f"Feature: {feature}, Entropy Reduction: {reduction}")
        idx = np.argmax(reductions)
        best = unknown_features[idx]
        best_red = reductions[idx]
        print(f"Optimal feature to query: {best} with Entropy Reduction: {best_red}")
        return best, best_red
