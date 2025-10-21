import pandas as pd
import os
from glob import glob
from bed import BEDModel

class OSCECaseLoader:
    def __init__(self, folder_path):
        """
        Loads all OSCE case CSVs from a folder into a single DataFrame.
        Args:
            folder_path (str): Path to the folder containing OSCE case CSV files.
        """
        self.folder_path = folder_path
        self.data = self._load_all_cases()

    def _load_all_cases(self):
        """
        Reads all CSV files in the folder and concatenates them into one DataFrame.
        """
        csv_files = glob(os.path.join(self.folder_path, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.folder_path}")
        # Sort all files alphabetically to ensure consistent order by string name
        csv_files.sort()
        
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df["Source_File"] = os.path.basename(file)  # keep track of origin
                #Change all columns name Presenting_Complaint to Presenting_Complaint
                df.columns = [col.replace("Presenting Complaint", "Presenting_Complaint") for col in df.columns]
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

        combined = pd.concat(dfs, ignore_index=True)
        unnamed = [c for c in combined.columns if c.startswith("Unnamed")]
        combined.drop(columns=unnamed, inplace=True, errors="ignore")
        return combined

    def list_cases(self):
        """Return a summary table with just the row number and diagnosis (if present)."""
        if self.data is None:
            return "No cases loaded."
        cols = [c for c in ["Clinical_Diagnosis", "Label", "Source_File"] if c in self.data.columns]
        return self.data[cols].reset_index()

    def fetch_case(self, index):
        """
        Returns a generic case description listing all columns and values.
        Args:
            index (int): Row index (0-based).
        Returns:
            str: Presenting complaint (if present).
            str: Full case description.
            str: Clinical diagnosis (if present).
            int: Label (if present).
        """
        if self.data is None:
            raise ValueError("No cases loaded.")
        if index < 0 or index >= len(self.data):
            raise IndexError(f"Index {index} out of range (0–{len(self.data)-1}).")

        row = self.data.iloc[index]

        # Pull label and diagnosis if present
        diagnosis = row.get("Clinical_Diagnosis", "Unknown")
        label = row.get("Label", None)

        # Pull presenting complaint
        presenting = row.get("Presenting_Complaint", None)
        if pd.isna(presenting):
            presenting = None

        # Build a generic description by listing all other columns
        excluded = {"Clinical_Diagnosis", "Label", "Source_File", "Presenting_Complaint"}
        details = []
        for col, val in row.items():
            # Include only non-excluded and non-NaN columns
            if col not in excluded and not pd.isna(val):
                details.append(f"{col}: {val}")

        description = f"Case details:\n" + "\n".join(details)
        description += f"\n\nSuspected diagnosis: {diagnosis}"

        return presenting, description, diagnosis, label
    
    def preprocess_data(self):
        """
        Preprocess loaded OSCE case data: drop rows with missing labels and reset index.
        """
        if self.data is None:
            print("No data available to preprocess")
            return
        # Ensure Label column exists and drop rows with missing labels
        if 'Label' in self.data.columns:
            self.data = self.data.dropna(subset=['Label']).reset_index(drop=True)
        else:
            print("Warning: 'Label' column not found in OSCE data")
        # Change all columns name Presenting_Complaint to Presenting_Complaint
        self.data.columns = [col.replace("Presenting Complaint", "Presenting_Complaint") for col in self.data.columns]

    def get_item(self, index):
        """
        Get a specific item (row) from the OSCE data.
        Returns:
          - A one-row DataFrame of features (excluding Clinical_Diagnosis, Label, Source_File)
          - The Label value as stored
        """
        if self.data is None or index < 0 or index >= len(self.data):
            print("Index out of range or no data available")
            return None, None
        row = self.data.iloc[index]
        label = row.get('Label', None)
        diagnosis = row.get('Clinical_Diagnosis', None)
        
        # Drop non-feature columns
        features = row.drop(['Clinical_Diagnosis', 'Label', 'Source_File'], errors='ignore')
        return features.to_frame().T, label, diagnosis

    def return_feature_names(self, idx):
        """
        Return known and unknown feature names for OSCE data.
        """
        #Get the specific case data
        if self.data is None or idx < 0 or idx >= len(self.data):
            print("Index out of range or no data available")
            return [], []
        data= pd.DataFrame(self.data.iloc[idx]).T
       
        #Drop all COLUMNS with NAn values in data
        data = data.dropna(axis=1)
        
        # Known features always include Presenting_Complaint if present
        # Unknown features are all other columns minus metadata and known
        known = ["Presenting_Complaint"] 
        all_feats = data.columns.tolist()
        exclude = ['Clinical_Diagnosis', 'Label', 'Source_File', 'Presenting Complaint', "Presenting_Complaint"]
        unknown = [feat for feat in all_feats if feat not in exclude]
        return known, unknown
    

class OSCEBEDModel(BEDModel):
    """
    BEDModel subclass for OSCE cases to predict diagnosis probability.
    """
    def __init__(self, model_name, diagnosis=None):
        super().__init__(model_name)
        self.diagnosis = diagnosis
        

    def format_known_data(self, known_features):
        """
        Format known features into a structured case description for LLM input.
        Accepts a dict or a single-row pandas DataFrame.
        """
        # If DataFrame, extract first row as dict
        if isinstance(known_features, pd.DataFrame):
            if not known_features.empty:
                known_features = known_features.iloc[0].to_dict()
            else:
                known_features = {}
        # If string, return directly
        if isinstance(known_features, str):
            return known_features
        # Build details list
        details = []
        for col, val in known_features.items():
            # Skip missing values
            if pd.isna(val):
                continue
            details.append(f"{col}: {val}")
        return "Case details:\n" + "\n".join(details)

    def sample_random_variable(self, known_features, feature_to_sample):
        """
        Simulate a plausible case value for a given feature in a disease-agnostic way.
        """
        self.sampling_prompt_template = (
            "You are a expert clinician. Based on the following case details, "
            f"simulate a plausible value for {feature_to_sample}. "
            "Return only the value without explanation."
            
            f"{self.format_known_data(known_features)} " # Use formatted known features
        )
        return super().sample_random_variable(known_features, feature_to_sample)

    def predict_risk(self, known_features, extra_info=""):
        """
        Use a generic prompt to predict diagnosis probability.
        """
        self.risk_prompt_template = (
            "You are an expert clinician. Given the following case details, "
            f"estimate a realistic and conservative probability of the suspected diagnosis of {self.diagnosis} as a single number between 0 and 1. "
            "Return only the number that can be converted to a Python float, without any additional commentary."
            
        )
        return super().predict_risk(known_features, extra_info)

    def select_feature_implicit(self, known_features, unknown_features):
        """
        Choose the next most informative feature from unknown_features, generically.
        """
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        self.implicit_selection_prompt_template = (
            f"You are a clinical assistant. Given these case details: {known_data}. "
            f"Which additional feature from the list [{unknown_str}] would be most informative next for the suspected diagnosis of {self.diagnosis}? "
            "Return only the feature name without commentary."
        )
        return super().select_feature_implicit(known_features, unknown_features)

    def get_best_global_features(self, all_features, n):
        """
        Select the top-n most informative features from all_features in a disease-agnostic way.
        """
        features_str = ", ".join(all_features)
        self.global_features_prompt_template = (
            f"You are a clinical assistant. From the following features: {features_str}, "
            f"identify the {n} most informative features for the diagnosis of {self.diagnosis}. "
            "Return ONLY a valid Python list of feature names with exactly the same name and units as shown in the list, e.g., ['Age (years)', 'Heart rate (bpm)', ...], "
            "with no markdown, code fences, or extra commentary."
        )
        response = super().get_best_global_features(all_features, n)
        return self.parse_model_response(response)

    def select_feature_bayesian(self, known_features, unknown_features):
        """
        Choose the next most informative feature from unknown_features using Bayesian reasoning.
        """
        known_data = self.format_known_data(known_features)
        unknown_str = ", ".join(unknown_features)
        self.bayesian_selection_prompt_template = (
            f"You are a clinical assistant. Given these case details: {known_data}, "
            f"which feature from the list [{unknown_str}] would provide the most information next? "
            "Return only the feature name without commentary."
        )
        return super().select_feature(known_features, unknown_features)

    def parse_model_response(self, response):
        """
        Parse the model response to ensure it is correctly interpreted as a Python list.
        """
        if isinstance(response, list):
            return response  # If already a list, return as-is

        try:
            # Remove embedded Python code formatting if present
            response = response.replace('```python', '').replace('```', '').strip()
            # Attempt to evaluate the response as a Python list
            parsed_response = eval(response)
            if isinstance(parsed_response, list):
                return parsed_response
            else:
                raise ValueError("Response is not a valid list.")
        except Exception as e:
            print(f"Error parsing model response: {e}, using raw response as list.")
            # Fallback: Split the response into a list manually
            return [item.strip() for item in response.strip("[]").split(",") if item.strip()]
