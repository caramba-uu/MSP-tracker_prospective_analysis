import os
import copy

import numpy as np
import pandas as pd
import tqdm
from collections import Counter


FEATURES = ['edss_score', 'age_at_visit', 'sex_label',
            'no_treatment', 'first_line_DMT', 'second_line_DMT', 'other_drugs', 'relapse_treatment_drugs', 'stem_cell_treatment',
            'eq5d_score', 'age_at_eq5d',
            'sdmt_score', 'age_at_sdmt',
            'mono_on_sum', 'monofocal_sum', 'multi_focal_sum', 'afferent_non_on_sum', 'steroid_treatment_sum', 'is_last_relapse_steroid_treated', 'is_last_relapse_completely_remitted',
            'age_at_relapse', 'revised_debut_age', 'age_at_debut_relapse',
            't2_lesion_catagory', 'brain_barrier_lesion_catagory', 'spinal_barrier_lesion_catagory', 'age_at_mri']

Y_LABEL = ["y_label"]


def print_df_summary(input_df):
    print(len(input_df), len(input_df["patient_code"].unique()), Counter(input_df["y_label"]))


def _read_split(path):
    split_df = pd.read_csv(path)
    split_df = split_df.replace(-1, np.nan)
    split_df["visit_date"] = pd.to_datetime(split_df['visit_date'], format='%Y-%m-%d')
    return split_df


def load_2022_data(data_dir):
    """Loads the 2022 train/valid/test/calib splits. Validation is merged into train."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_dir, "valid.csv"))
    train_df = pd.concat([train_df, valid_df])
    train_df = train_df.replace(-1, np.nan)
    train_df["visit_date"] = pd.to_datetime(train_df['visit_date'], format='%Y-%m-%d')

    calib_df = _read_split(os.path.join(data_dir, "calib.csv"))
    test_df = _read_split(os.path.join(data_dir, "test.csv"))
    return train_df, calib_df, test_df


def load_2025_data(data_dir):
    df_2025 = pd.read_csv(os.path.join(data_dir, "3_cleaned_data.csv"))
    df_2025 = df_2025.sort_values("visit_date")
    df_2025["visit_date"] = pd.to_datetime(df_2025['visit_date'], format='%Y-%m-%d')
    return df_2025


def load_relapse_and_edss(data_dir):
    """Loads the relapse (skov) and EDSS tables used for the Lorscheider criteria."""
    relapse_df = pd.read_csv(os.path.join(data_dir, "english_translated", "skov.csv"))
    relapse_df = relapse_df.sort_values("date")
    relapse_df["date"] = pd.to_datetime(relapse_df['date'], format='%Y-%m-%d')

    edss_df = pd.read_csv(os.path.join(data_dir, "english_translated", "edss.csv"))
    edss_df = edss_df.sort_values("edss date")
    edss_df["edss date"] = pd.to_datetime(edss_df['edss date'], format='%Y-%m-%d')
    edss_df = edss_df.drop_duplicates(subset=["patient_code", "edss date"])

    sub_edss_df = edss_df[["patient_code", "edss date", "edss pyramidal", "calculated score"]]
    return relapse_df, sub_edss_df


def merge_with_edss(input_df, sub_edss_df):
    return pd.merge(input_df, sub_edss_df, how="left",
                    left_on=["patient_code", "visit_date"], right_on=["patient_code", "edss date"])


def get_exlusive_df(inp_df_2025, inp_df_2022):
    """Keeps only the 2025 visits that happen after the last visit seen in 2022."""
    inp_df_2022 = inp_df_2022.sort_values(["age_at_visit"])
    pcodes_2022 = set(inp_df_2022["patient_code"].unique())
    selected_sub_df = []
    for pcode in tqdm.tqdm(inp_df_2025["patient_code"].unique()):
        sub_2025_df = inp_df_2025[inp_df_2025["patient_code"] == pcode]
        if pcode in pcodes_2022:
            sub_2022_df = inp_df_2022[inp_df_2022["patient_code"] == pcode]
            sub_2025_df = sub_2025_df[sub_2025_df["visit_date"] > sub_2022_df["visit_date"].to_list()[-1]]
        selected_sub_df.append(sub_2025_df)
    return pd.concat(selected_sub_df)


def get_equivalent_2025_dataset(inp_df_2025, inp_df_2022, inverse_pcode_selection=False):
    copy_2025_df = copy.deepcopy(inp_df_2025)
    pcodes_2022 = inp_df_2022["patient_code"].unique()
    if not inverse_pcode_selection:
        copy_2025_df = copy_2025_df[copy_2025_df["patient_code"].isin(pcodes_2022)].reset_index(drop=True)
    else:
        copy_2025_df = copy_2025_df[~copy_2025_df["patient_code"].isin(pcodes_2022)].reset_index(drop=True)
    return copy_2025_df


def categorize_patient(labels_set):
    if labels_set == {0}:
        return "RRMS"
    elif labels_set == {1}:
        return "SPMS"
    elif labels_set == {0, 1}:
        return "transition"


def get_patient_groups(input_df):
    """Returns the patient codes of RRMS-only, SPMS-only and transitioning patients."""
    patient_labels = input_df.groupby("patient_code")["y_label"].unique().apply(set)
    patient_groups = patient_labels.apply(categorize_patient)

    rrms_pcodes = patient_groups[patient_groups == "RRMS"].index
    spms_pcodes = patient_groups[patient_groups == "SPMS"].index
    transition_pcodes = patient_groups[patient_groups == "transition"].index
    return rrms_pcodes, spms_pcodes, transition_pcodes


def get_pcodes_with_min_visits(input_df, min_visits=3):
    return input_df.groupby("patient_code")["patient_code"].count().loc[lambda x: x >= min_visits].index
