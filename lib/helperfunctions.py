import pandas as pd

def dataframe_to_markdown(df):
    """
    Converts a pandas DataFrame to a Markdown table.
    
    Args:
        df (pd.DataFrame): The DataFrame to convert.
    
    Returns:
        str: A string representing the DataFrame as a Markdown table.
    """
    # Generate the header row
    header = "| " + " | ".join(df.columns) + " |"
    
    
    # Generate the data rows
    rows = "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in df.values
    )
    
    # Combine all parts
    markdown_table = f"{header}\n{rows}"
    return markdown_table

def hepatitis_clinical_vignette(features):
    """
    Converts a feature vector (pd.DataFrame with a single row) from the hepatitis dataset into a clinical vignette.

    Expected columns (if available):
      - Age: Patient's age.
      - Sex: Patient's sex ("m" or "f").
      - Lab tests: ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT.

    Returns:
        str: A clinical vignette summarizing the patient's hepatitis-related lab values.
    """
    # Ensure features is a DataFrame and extract the first row as a dictionary
    if isinstance(features, pd.DataFrame):
        data = features.iloc[0].to_dict()
    else:
        raise ValueError("Expected features to be a pandas DataFrame with a single row.")

    parts = []

    # Describe basic patient info.
    age = data.get("Age")
    sex = data.get("Sex")

    # Ensure age and sex are properly formatted.
    try:
        age = int(float(age)) if age is not None else None
    except (ValueError, TypeError):
        age = None

    if isinstance(sex, dict):  # Handle unexpected nested dictionary
        sex = sex.get(1)  # Extract the value if it's a dictionary
    sex = str(sex).lower() if sex is not None else None

    if age is not None and sex in ["m", "f"]:
        parts.append(f"The patient is a {age}-year-old {'male' if sex == 'm' else 'female'}.")
    elif age is not None:
        parts.append(f"The patient is {age} years old.")
    elif sex in ["m", "f"]:
        parts.append(f"The patient is {'male' if sex == 'm' else 'female'}.")

    # Add lab test results if available.
    lab_tests = {
        "ALB": ("Albumin", "g/L"),
        "ALP": ("Alkaline Phosphatase", "IU/L"),
        "ALT": ("ALT", "IU/L"),
        "AST": ("AST", "IU/L"),
        "BIL": ("Bilirubin", "µmol/L"),
        "CHE": ("Cholinesterase", "kU/L"),
        "CHOL": ("Cholesterol", "mmol/L"),
        "CREA": ("Creatinine", "µmol/L"),
        "GGT": ("GGT", "IU/L"),
        "PROT": ("Total Protein", "g/L")
    }

    for key, (label, unit) in lab_tests.items():
        value = data.get(key)
        if isinstance(value, dict):  # Handle unexpected nested dictionary
            value = value.get(1)  # Extract the value if it's a dictionary
        try:
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            value = None

        if value is not None:
            parts.append(f"{label} was measured at {value} {unit}.")

    return " ".join(parts)

def diabetes_clinical_vignette(features):
    """
    Converts a feature vector (pd.DataFrame with a single row) from the diabetes dataset into a clinical vignette.

    Expected columns (if available):
      - Pregnancies: Number of times pregnant.
      - Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test (mg/dL).
      - BloodPressure: Diastolic blood pressure (mm Hg).
      - SkinThickness: Triceps skinfold thickness (mm).
      - Insulin: 2-Hour serum insulin (mu U/ml).
      - BMI: Body mass index (kg/(m^2)).
      - DiabetesPedigreeFunction: Diabetes pedigree function.
      - Age: Age (years).

    Returns:
        str: A clinical vignette summarizing the patient's diabetes-related features.
    """
    # Ensure features is a DataFrame and extract the first row as a dictionary
    if isinstance(features, pd.DataFrame):
        data = features.iloc[0].to_dict()
    else:
        raise ValueError("Expected features to be a pandas DataFrame with a single row.")

    parts = []

    # Describe basic patient info.
    age = data.get("Age")
    pregnancies = data.get("Pregnancies")

    # Ensure age and pregnancies are properly formatted.
    try:
        age = int(float(age)) if age is not None else None
    except (ValueError, TypeError):
        age = None

    try:
        pregnancies = int(float(pregnancies)) if pregnancies is not None else None
    except (ValueError, TypeError):
        pregnancies = None

    if age is not None and pregnancies is not None:
        parts.append(f"The patient is a {age}-year-old who has been pregnant {pregnancies} time(s).")
    elif age is not None:
        parts.append(f"The patient is {age} years old.")
    elif pregnancies is not None:
        parts.append(f"The patient has been pregnant {pregnancies} time(s).")

    # Add clinical features if available.
    clinical_features = {
        "Glucose": ("Plasma glucose concentration", "mg/dL"),
        "BloodPressure": ("Diastolic blood pressure", "mm Hg"),
        "SkinThickness": ("Triceps skinfold thickness", "mm"),
        "Insulin": ("2-Hour serum insulin", "mu U/ml"),
        "BMI": ("Body mass index", "kg/(m^2)"),
        "DiabetesPedigreeFunction": ("Diabetes pedigree function", ""),
    }

    for key, (label, unit) in clinical_features.items():
        value = data.get(key)
        if isinstance(value, dict):  # Handle unexpected nested dictionary
            value = value.get(1)  # Extract the value if it's a dictionary
        try:
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            value = None

        if value is not None:
            if unit:  # Add unit if available
                parts.append(f"{label} was measured at {value} {unit}.")
            else:  # No unit for DiabetesPedigreeFunction
                parts.append(f"{label} was measured at {value}.")

    return " ".join(parts)


def kidney_clinical_vignette(features):
    """
    Converts a feature vector (pd.DataFrame with a single row) from the kidney dataset into a clinical vignette.

    Expected columns (if available):
      - age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc, htn, dm, cad, appet, pe, ane, class.

    Returns:
        str: A clinical vignette summarizing the patient's kidney-related features.
    """
    # Ensure features is a DataFrame and extract the first row as a dictionary
    if isinstance(features, pd.DataFrame):
        data = features.iloc[0].to_dict()
    else:
        raise ValueError("Expected features to be a pandas DataFrame with a single row.")

    parts = []

    # Describe basic patient info
    age = data.get("age")
    bp = data.get("bp")
    appet = data.get("appet")
    pe = data.get("pe")
    htn = data.get("htn")
    dm = data.get("dm")
    cad = data.get("cad")
    ane = data.get("ane")

    # Ensure age and blood pressure are properly formatted
    try:
        age = int(float(age)) if age is not None else None
    except (ValueError, TypeError):
        age = None

    try:
        bp = int(float(bp)) if bp is not None else None
    except (ValueError, TypeError):
        bp = None

    if age is not None:
        parts.append(f"The patient is {age} years old.")
    if bp is not None:
        parts.append(f"The patient's diastolic blood pressure is {bp} mm/Hg.")

    # Add clinical conditions
    if appet:
        parts.append(f"The patient has a {appet.lower()} appetite.")
    if pe:
        parts.append(f"The patient has {'pedal edema' if pe.lower() == 'yes' else 'no pedal edema'}.")
    if htn:
        parts.append(f"The patient {'has hypertension' if htn.lower() == 'yes' else 'does not have hypertension'}.")
    if dm:
        parts.append(f"The patient {'has diabetes mellitus' if dm.lower() == 'yes' else 'does not have diabetes mellitus'}.")
    if cad:
        parts.append(f"The patient {'has coronary artery disease' if cad.lower() == 'yes' else 'does not have coronary artery disease'}.")
    if ane:
        parts.append(f"The patient {'has anemia' if ane.lower() == 'yes' else 'does not have anemia'}.")

    # Add lab test results
    lab_tests = {
        "sg": ("Specific gravity", ""),
        "al": ("Albumin levels in urine", ""),
        "su": ("Sugar levels in urine", ""),
        "rbc": ("Red blood cells", ""),
        "pc": ("Pus cells", ""),
        "pcc": ("Pus cell clumps", ""),
        "ba": ("Bacteria in urine", ""),
        "bgr": ("Blood glucose random", "mg/dL"),
        "bu": ("Blood urea", "mg/dL"),
        "sc": ("Serum creatinine", "mg/dL"),
        "sod": ("Sodium levels", "mEq/L"),
        "pot": ("Potassium levels", "mEq/L"),
        "hemo": ("Hemoglobin levels", "g/dl"),
        "pcv": ("Packed cell volume", ""),
        "wbcc": ("White blood cell count", "cells/cmm"),
        "rbcc": ("Red blood cell count", "millions/cmm"),
    }

    for key, (label, unit) in lab_tests.items():
        value = data.get(key)
        try:
            value = float(value) if value is not None else None
        except (ValueError, TypeError):
            value = None

        if value is not None:
            if unit:
                parts.append(f"{label} was measured at {value} {unit}.")
            else:
                parts.append(f"{label} was measured at {value}.")

    return " ".join(parts)

    