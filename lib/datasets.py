import os
import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
class Dataset:
    """
    Base Dataset class providing common functionality for loading, preprocessing, and accessing data.
    Subclasses should override methods as needed for specific datasets.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.load_data()

    def load_data(self):
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
            # Drop any accidental unnamed index columns if present.
            unnamed_cols = [col for col in self.data.columns if col.startswith("Unnamed")]
            if unnamed_cols:
                self.data.drop(columns=unnamed_cols, inplace=True)
        else:
            print(f"File not found: {self.file_path}")

    def preprocess_data(self):
        """
        Preprocess the dataset. Subclasses should override this method as needed.
        """
        pass

    def return_length(self):
        """
        Return the number of rows in the dataset.
        """
        if self.data is not None:
            return len(self.data)
        else:
            print("No data available")
            return 0

    def get_item(self, index):
        """
        Get a specific item (row) from the dataset. Subclasses should override this method.
        """
        pass

    def return_feature_names(self):
        """
        Return known and unknown feature names. Subclasses should override this method.
        """
        return [], []



class HepatitisDataset(Dataset):
    def preprocess_data(self):
        if self.data is not None:
            # Create a new Label column:
            # 0 if Category is "0=Blood Donor", 1 otherwise.
            self.data['Label'] = self.data['Category'].apply(lambda x: 0 if x.strip().startswith("0=") else 1)
        else:
            print("No data to preprocess")

    def get_item(self, index):
        """
        Get a specific item (row) from the dataset.
        Returns a tuple containing:
          - A one-row DataFrame of features (all columns except Category and Label)
          - The Label as an integer
        """
        if self.data is None or index < 0 or index >= len(self.data):
            print("Index out of range or no data available")
            return None, None

        row = self.data.iloc[index]
        label = row['Label']
        # Remove the Category and Label columns to form the feature set
        features = row.drop(["Category", "Label"])
        return features.to_frame().T, label

    def return_feature_names(self):
        """
        Return known and unknown feature names.
        """
        known_features = ["Age", "Sex"]
        all_features = self.data.columns.tolist()
        unknown_features = [feat for feat in all_features if feat not in known_features and feat not in ["Category", "Label"]]
        return known_features, unknown_features


class DiabetesDataset(Dataset):
    def preprocess_data(self):
        if self.data is not None:
            # Convert Outcome column to integer (label)
            if 'Outcome' in self.data.columns:
                self.data['Outcome'] = self.data['Outcome'].astype(int)
            # Remove rows with missing values
            self.data.dropna(inplace=True)
        else:
            print("No data available to preprocess")

    def get_item(self, index):
        """
        Get a specific item (row) from the dataset.
        Returns a tuple containing:
          - A one-row DataFrame of features (all columns except Outcome)
          - The Outcome label as an integer
        """
        if self.data is None or index < 0 or index >= len(self.data):
            print("Index out of range or no data available")
            return None, None

        row = self.data.iloc[index]
        label = row['Outcome']
        # Remove the Outcome column to form the feature set
        features = row.drop("Outcome")
        return features.to_frame().T, label

    def return_feature_names(self):
        """
        Return known and unknown feature names.
        """
        known_features = ["Age"]
        all_features = self.data.columns.tolist()
        unknown_features = [feat for feat in all_features if feat not in known_features and feat != "Outcome"]
        return known_features, unknown_features

class KidneyDataset(Dataset):
    """
    KidneyDataset processes and provides access to kidney-related data.
    """

    def preprocess_data(self):
        # Replace '?' with NaN
        if self.data is not None:
            self.data.replace('?', pd.NA, inplace=True)
            # Convert 'class' column to a label (ensure it's binary)
            if 'class' in self.data.columns:
                # Convert 'class' to integer 0 if condition is 'notckd', 1 if 'ckd'
                self.data['class'] = self.data['class'].astype(str).str.strip()
                self.data['class'] = self.data['class'].replace({'ckd': 1, 'notckd': 0})

            # Remove rows with missing values
            self.data.dropna(inplace=True)
        else:
            print("No data available to preprocess")
    def get_item(self, index):
        """
        Get a specific item (row) from the dataset.
        Returns a tuple containing:
          - A one-row DataFrame of features (all columns except 'stroke')
          - The 'stroke' label as an integer
        """
        if self.data is None or index < 0 or index >= len(self.data):
            print("Index out of range or no data available")
            return None, None

        row = self.data.iloc[index]
        label = row['class']
        # Remove the 'stroke' column to form the feature set
        features = row.drop("class")
        return features.to_frame().T, label
    
    def return_feature_names(self):
        known_features = ['age', 'rbc','pe', 'appet', 'pcv','htn','dm','cad','appet','pe','ane', 'al', 'su']
        features_to_drop = ['class', 'sg', 'pcc', 'pc', 'ba', 'wc', 'pcv' ]
        all_features = self.data.columns.tolist()
        unknown_features = [feat for feat in all_features if feat not in known_features and feat not in features_to_drop]
        return known_features, unknown_features

    