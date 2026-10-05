import copy

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from collections import Counter

import lorscheider_utils
from data_utils import get_patient_groups, merge_with_edss


def get_df_following_lorschider_criteria(input_df, input_pcodes):
    """Selects patients with 3 or more visits, so that the Lorscheider criteria can be applied."""
    selected_pcodes = []
    for pcode in input_pcodes:
        sub_df = input_df[input_df["patient_code"] == pcode]
        if len(sub_df) >= 3:
            selected_pcodes.append(pcode)
    return input_df[input_df["patient_code"].isin(selected_pcodes)]


def get_table_results_visit_time_diff(visit_diff_list, add_text=""):
    seg_age_visit_freq_dict = dict(Counter(visit_diff_list))
    freq_dict = {k: v for k, v in sorted(seg_age_visit_freq_dict.items(), key=lambda item: item[0])}

    visit_diff_arr = np.array(list(freq_dict.keys()))
    frequency_arr = np.array(list(freq_dict.values()))

    count_early_ident = int(np.sum(frequency_arr[visit_diff_arr < -1]))
    count_late_ident = int(np.sum(frequency_arr[visit_diff_arr > 1]))
    count_correct_ident = int(np.sum(frequency_arr[visit_diff_arr == 0]))
    count_one_visit_deviation_ident = int(np.sum(frequency_arr[(visit_diff_arr == 1) | (visit_diff_arr == -1)]))

    return {"Correct" + str(add_text): count_correct_ident,
            "One visit deviation" + str(add_text): count_one_visit_deviation_ident,
            "Early" + str(add_text): count_early_ident,
            "Late" + str(add_text): count_late_ident}


def _get_group_visit_diffs(group_df, relapse_df, model, ccode):
    if group_df["patient_code"].nunique() == 0:
        return [], []
    lor_added_df, visit_diff_list = lorscheider_utils.plot_lorscheider(group_df, relapse_df, ccode=ccode, do_label_check=False)
    pred_visit_diff_list = lorscheider_utils.plot_visit_diff_from_model(lor_added_df, model, data_ccode=ccode, do_label_check=False)
    return visit_diff_list, pred_visit_diff_list


def get_lorscheider_stats(input_df, model, relapse_df, sub_edss_df, do_lor_specific_cutoff=False, ccode="SE"):
    """Counts the patients per diagnosis group (gt), and the ones flagged as SPMS
    by the Lorscheider criteria (lor) and by the model (pred)."""
    merged_df = merge_with_edss(input_df, sub_edss_df)
    rrms_pcodes, spms_pcodes, transition_pcodes = get_patient_groups(merged_df)

    rrms_df = merged_df[merged_df["patient_code"].isin(rrms_pcodes)]
    spms_df = merged_df[merged_df["patient_code"].isin(spms_pcodes)]
    transition_df = merged_df[merged_df["patient_code"].isin(transition_pcodes)]

    if do_lor_specific_cutoff:
        rrms_df = get_df_following_lorschider_criteria(rrms_df, rrms_pcodes)
        spms_df = get_df_following_lorschider_criteria(spms_df, spms_pcodes)
        transition_df = get_df_following_lorschider_criteria(transition_df, transition_pcodes)

    rrms_visit_diff_list, rrms_pred_visit_diff_list = _get_group_visit_diffs(rrms_df, relapse_df, model, ccode)
    spms_visit_diff_list, spms_pred_visit_diff_list = _get_group_visit_diffs(spms_df, relapse_df, model, ccode)
    transition_visit_diff_list, transition_pred_visit_diff_list = _get_group_visit_diffs(transition_df, relapse_df, model, ccode)

    gt_list = [len(rrms_df["patient_code"].unique()), len(spms_df["patient_code"].unique()), len(transition_df["patient_code"].unique())]
    lor_list = [len(rrms_visit_diff_list), len(spms_visit_diff_list), len(transition_visit_diff_list)]
    pred_list = [len(rrms_pred_visit_diff_list), len(spms_pred_visit_diff_list), len(transition_pred_visit_diff_list)]

    columns = ["rrms", "spms", "transition"]
    model_lor_gt_df = pd.DataFrame({ccode: gt_list}, index=columns).T
    model_lor_lor_df = pd.DataFrame({ccode: lor_list}, index=columns).T
    model_lor_pred_df = pd.DataFrame({ccode: pred_list}, index=columns).T

    stats_df = pd.merge(model_lor_gt_df, model_lor_lor_df, how="right", left_index=True, right_index=True, suffixes=("_gt", "_lor"))
    stats_df = pd.merge(stats_df, model_lor_pred_df, how="right", left_index=True, right_index=True, suffixes=("", "_pred"))
    return stats_df


def plot_the_lor_comparison_plot(for_model_df,
                                 for_lor_df,
                                 relapse_df,
                                 sub_edss_df,
                                 do_label_check=True,
                                 plot_spms=False,
                                 model="",
                                 plot_only_model=False,
                                 title="Time difference to transition plot",
                                 offset_right=0, fig_title="", return_axis=False, ax1=None, ccode="SE"):
    """Plots the visit difference between the diagnosed transition and the
    transition found by the Lorscheider criteria (green) and the model (orange)."""
    merged_lor_df = merge_with_edss(for_lor_df, sub_edss_df)

    if ax1 is None:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))

    ax1.grid(False)
    ax1.set_facecolor('white')
    for spine in ax1.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1)

    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 22})
    lor_added_df, visit_diff_list = lorscheider_utils.plot_lorscheider(merged_lor_df, relapse_df, ccode=ccode, ax=ax1, color="green",
                                                                       alpha=1, fig_text=False, progression_position=0,
                                                                       do_label_check=do_label_check, plot_spms=plot_spms, offset_right=offset_right)

    if plot_only_model:
        ax1.clear()
    pred_visit_diff_list = lorscheider_utils.plot_visit_diff_from_model(for_model_df, model, data_ccode=ccode, ax=ax1, color="orange",
                                                                        alpha=0.7, fig_text=False, progression_position=1,
                                                                        do_label_check=do_label_check, plot_spms=plot_spms, offset_right=offset_right)

    if not plot_only_model:
        colors = {"Lorscheider criteria": "green", "Model's predictions": "orange"}
        labels = list(colors.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
        ax1.legend(handles, labels, loc="upper right", fontsize=16)

    if not return_axis:
        plt.title(title, fontsize=15)
        if fig_title != "":
            plt.savefig(fig_title, dpi=1000, bbox_inches='tight')
        plt.show()

    gt_table = get_table_results_visit_time_diff(visit_diff_list)
    pred_table = get_table_results_visit_time_diff(pred_visit_diff_list, add_text="_pred")

    cat_table = {"Country": str(ccode), "Total": len(for_model_df["patient_code"].unique())}
    cat_table.update(gt_table)
    cat_table.update(pred_table)
    pred_gt_visit_diff_table_df = pd.DataFrame([cat_table])

    if not return_axis:
        return pred_gt_visit_diff_table_df
    else:
        return pred_gt_visit_diff_table_df, ax1


def extend_lor_progression(lor_df):
    """The Lorscheider criteria mark only the first progression visit,
    mark every visit after it as progressed as well."""
    the_lor_result_df = copy.deepcopy(lor_df)
    for pcode in tqdm.tqdm(the_lor_result_df["patient_code"].unique()):
        sub_df = the_lor_result_df[the_lor_result_df["patient_code"] == pcode]
        if 1 in sub_df["lor_progression"].to_list():
            first_pred_spms_date = sub_df[sub_df["lor_progression"] == 1]["visit_date"].to_list()[0]
            index_to_change = sub_df[sub_df["visit_date"] >= first_pred_spms_date].index.to_list()
            the_lor_result_df.loc[index_to_change, "lor_progression"] = 1
    return the_lor_result_df
