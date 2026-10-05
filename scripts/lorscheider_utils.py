import os
import sys
import tqdm
import glob
import copy
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from collections import Counter
import math
from matplotlib.ticker import MaxNLocator

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, MarginErrFunc, ClassificationErrFunc, InverseProbabilityErrFunc
from nonconformist.base import ClassifierAdapter
from nonconformist.acp import AggregatedCp, BootstrapSampler

plt.style.use("default")
import warnings
warnings.filterwarnings("ignore")


x_features = ['edss_score','age_at_visit' ,'sex_label',
                'no_treatment', 'first_line_DMT', 'second_line_DMT', 'other_drugs', 'relapse_treatment_drugs','stem_cell_treatment',
                'eq5d_score','age_at_eq5d',
               'sdmt_score','age_at_sdmt',
                'mono_on_sum','monofocal_sum','multi_focal_sum','afferent_non_on_sum','steroid_treatment_sum','is_last_relapse_steroid_treated','is_last_relapse_completely_remitted',
           'age_at_relapse','revised_debut_age', 'age_at_debut_relapse',
                't2_lesion_catagory', 'brain_barrier_lesion_catagory', 'spinal_barrier_lesion_catagory','age_at_mri']
y_label = "y_label"

def get_patient_data_from_pcode(pcode, input_df, relapse_df):
    return input_df[input_df["patient_code"] == pcode], relapse_df[relapse_df["patient_code"] == pcode]


def get_lorscheider_progression_index(pcode, input_df, relapse_df, force_fs=True):

    sub_df, sub_relapse_df = get_patient_data_from_pcode(pcode, input_df, relapse_df)

    all_edss = sub_df["edss_score"].to_list()
    all_visit_date = sub_df["visit_date"].to_list()
    if force_fs == True:
        all_kfs1 = sub_df["edss pyramidal"].to_list()

    if len(all_edss) < 3:
        return None, None

    for i, current_edss in enumerate(all_edss):
        current_date = all_visit_date[i]

        if force_fs == True:
            kfs_score = all_kfs1[i]
            if kfs_score < 2 and str(kfs_score) != "nan": # if kfs criteria is not met and is not nan. If nan ignore kfs and check EDSS
                continue

        if i + 2 >= len(all_edss):
            break

        next_edss = all_edss[i+1]
        next_date = all_visit_date[i+1]

        if next_edss > current_edss:
            if current_edss <= 5.5:
                step_edss = 1
            if current_edss > 5.5:
                step_edss = 0.5

            if all_edss[i+1] < 4: # if edss criteria is not met
                continue

            if next_edss - current_edss >= step_edss:
                #if not any(abs(next_date - sub_relapse_df["date"])/np.timedelta64(1, 'D') <= 3 * 30): # if there is no relapse within 3 months, proceed
                if not any(abs(next_date - sub_relapse_df["date"])/np.timedelta64(1, 'D') <= 1 * 30): # if there is no relapse within 1 months, proceed
                    for j in range(i+2, len(all_edss), 1):
                        if j >= len(all_edss):
                            break

                        confirmation_edss = all_edss[j]
                        confirmation_date = all_visit_date[j]
                        if (confirmation_date - next_date)/np.timedelta64(1, 'D') >= 3 * 30: # if the confirmation date is above 3 months
                            if not any(abs(confirmation_date - sub_relapse_df["date"])/np.timedelta64(1, 'D') <= 1 * 30): # if there is no relapse within 1 months, proceed
                                if confirmation_edss -  current_edss >= step_edss:
                                    return i + 1, int(sub_df.index[i+1])
                                else:
                                    break

    return None, None

def plot_lorscheider(sub_edss_df, sub_relapse_df, ccode="", ax=None, color="", alpha=0.1, fig_text="", progression_position=0, do_label_check=True,plot_spms=False,offset_right=0):

    if do_label_check: # only use the ones with transition
        inconsistent_patients_method1 = sub_edss_df.groupby("patient_code")["y_label"].nunique()
        patients_with_both_labels = inconsistent_patients_method1[inconsistent_patients_method1==2].index.tolist()
    else:
        patients_with_both_labels = sub_edss_df["patient_code"].unique()

    count = 0
    copy_edss_df = copy.deepcopy(sub_edss_df)
    copy_edss_df["lor_progression"] = 0

    for entry in patients_with_both_labels:
        prog_index, df_prog_index = get_lorscheider_progression_index(entry, copy_edss_df, sub_relapse_df, force_fs=True)
        if prog_index is not None:
            copy_edss_df.loc[df_prog_index, "lor_progression"] = 1
            count += 1

    lor_pcodes = copy_edss_df[copy_edss_df["lor_progression"] == 1]["patient_code"].to_list()
    lor_pcodes = copy_edss_df["patient_code"].to_list()
    #print (len(patients_with_both_labels))
    if do_label_check or plot_spms:
        visit_diff_list = []
        year_diff_list = []
        no_progression_found = 0
        for pcode in patients_with_both_labels:
            sub_df = copy_edss_df[copy_edss_df["patient_code"] == pcode]
            real_visit_prog = sub_df["y_label"].to_list().index(1)
            if 1 in sub_df["lor_progression"].to_list():
                lor_visit_prog = sub_df["lor_progression"].to_list().index(1)
                visit_diff_list.append(lor_visit_prog - real_visit_prog)
                year_diff_list.append((sub_df["visit_date"].to_list()[lor_visit_prog] - sub_df["visit_date"].to_list()[real_visit_prog]) / np.timedelta64(365, "D"))
            else:
                no_progression_found += 1

        make_pred_gt_visit_diff_plot(visit_diff_list, no_progression_found=no_progression_found, ax=ax, color=color,
                                     alpha=alpha, fig_text=fig_text, progression_position=progression_position,offset_right=offset_right)

        return copy_edss_df, visit_diff_list

    else:
        visit_diff_list = []
        no_progression_found = 0
        for pcode in patients_with_both_labels:
            sub_df = copy_edss_df[copy_edss_df["patient_code"] == pcode]
            if 1 in sub_df["lor_progression"].to_list():
                lor_visit_prog = sub_df["lor_progression"].to_list().index(1)
                visit_diff_list.append(lor_visit_prog) #lor_visit_prog
            else:
                no_progression_found += 1
        return copy_edss_df, visit_diff_list

def lambda_function(x):
    return x[1]


def get_predicted_bool(model, input_df, ccode, x_features, y_label):
    #with open("/".join(model_path.split("/")[:-1]) + "/" + str(ccode) + "_sign_used.txt", "r") as f:
    #    significance = float(f.read())
    #print(significance)
    significance = 0.07
    icp=model
    #icp = joblib.load(model_path)
    #print(input_df)
    test_pred_bool = icp.predict(input_df[x_features].values, significance)
    return test_pred_bool

def check_if_jumping_pred(preds): # not using this since this adds longitudinal info for the model. so returning False
    return False
    first_jump, first_jump_back = False, False
    for entry in preds:
        if entry == 1:
            first_jump = True
        if first_jump == True and entry == 0:
            first_jump_back = True
        if first_jump_back == True and entry == 1:
            second_jump = True
            return True
    return False #, False


def non_jumpy_progression_index(sub_df):
    sub_preds = sub_df["pred"].to_list()
    sub_age_at_visit = sub_df["age_at_visit"].to_list()
    for i, single_pred in enumerate(sub_preds):
        if len(sub_preds) == i + 1 and single_pred == 1:
            return i

        if single_pred == 1:
            for j in range(i+1, len(sub_preds), 1):
                next_pred = sub_preds[j]
                if next_pred == 0:
                    break

                if next_pred == 1:
                    current_age = sub_age_at_visit[i]
                    next_age = sub_age_at_visit[j]

                    if (next_age - current_age) * 365 >= 3 * 30:
                        return i
    #print (sub_preds,i)
    return None



def plot_visit_diff_from_model(input_df, model, data_ccode, ax=None, color="", alpha=1, fig_text=True, progression_position=0, do_label_check=True,plot_spms=False,offset_right=0):

    if do_label_check: # only use the ones with transition
        inconsistent_patients_method1 = input_df.groupby("patient_code")["y_label"].nunique()
        patients_with_both_labels = inconsistent_patients_method1[inconsistent_patients_method1>1].index.tolist()
    else:
        patients_with_both_labels = input_df["patient_code"].unique()

    current_df = copy.deepcopy(input_df[input_df["patient_code"].isin(patients_with_both_labels)])
    current_df = current_df.sort_values(by="age_at_visit")
    current_df = current_df.reset_index()

    pred_bool = get_predicted_bool(model, current_df, data_ccode, x_features, y_label)

    for entry, df_index in zip(pred_bool, current_df.index.to_list()):
        if np.sum(entry) != 2 and np.sum(entry) != 0:
            label = int(np.argmax(entry))
        if np.sum(entry) == 2:
            label = 2
        if np.sum(entry) == 0:
            label = 3
        current_df.loc[df_index, "pred"] = label
    #return current_df
    if do_label_check  or plot_spms: # only use the ones with transition
        pred_visit_diff_list = []
        pred_year_diff_list = []
        pred_no_progression_found = 0
        for pcode in patients_with_both_labels:
            sub_df = current_df[current_df["patient_code"] == pcode]
            real_visit_prog = sub_df["y_label"].to_list().index(1)
            if 1 in sub_df["pred"].to_list():
                if check_if_jumping_pred(sub_df["pred"].to_list()): # set to false, wont enter loop
                    pred_visit_prog = non_jumpy_progression_index(sub_df)
                    if pred_visit_prog is None:
                        pred_no_progression_found += 1
                        continue
                else:
                    pred_visit_prog = sub_df["pred"].to_list().index(1)

                pred_visit_diff_list.append(pred_visit_prog - real_visit_prog)
                # if abs(pred_visit_prog - real_visit_prog) > 50:
                #    print(pcode)
                #    break
                pred_year_diff_list.append(sub_df["age_at_visit"].to_list()[pred_visit_prog] - sub_df["age_at_visit"].to_list()[real_visit_prog])
            else:
                pred_no_progression_found += 1

        make_pred_gt_visit_diff_plot(pred_visit_diff_list, no_progression_found=pred_no_progression_found, ax=ax, color=color,
                                     alpha=alpha, fig_text=fig_text, progression_position=progression_position,offset_right=offset_right)

        return pred_visit_diff_list
    else:
        pred_visit_diff_list = []
        pred_no_progression_found = 0
        for pcode in patients_with_both_labels:
            sub_df = current_df[current_df["patient_code"] == pcode]
            if 1 in sub_df["pred"].to_list():
                if check_if_jumping_pred(sub_df["pred"].to_list()):
                    pred_visit_prog = non_jumpy_progression_index(sub_df)
                    if pred_visit_prog == None:
                        pred_no_progression_found += 1
                        continue
                else:
                    pred_visit_prog = sub_df["pred"].to_list().index(1)

                pred_visit_diff_list.append(pred_visit_prog)
            else:
                pred_no_progression_found += 1
        return pred_visit_diff_list


def make_pred_gt_visit_diff_plot(visit_diff_list, no_progression_found=0, ax=None, color="", alpha=1, fig_text=True, progression_position=0,offset_right=0):
    plt.style.use("ggplot")
    plt.rcParams.update({'font.size': 22})

    seg_age_visit_freq_dict = Counter(visit_diff_list)
    freq_dict = {k: v for k, v in sorted(seg_age_visit_freq_dict.items(), key=lambda item: item[0])}
    #print (freq_dict)

    if color == "":
        color = (0.2, 0.4, 0.2, 1)

    if type(ax) == list:
        fig, ax = plt.subplots()
        fig.set_size_inches(21, 7)

    # fontsize = 25
    fontsize = 20


    ax.set_ylabel("Number of patients", fontsize=fontsize)
    ax.set_xlabel("Follow-up visits", fontsize=fontsize)
    ax.set_title("")


    pps = ax.bar(freq_dict.keys(), freq_dict.values(), color=color, width=1, align="center", alpha=alpha)


    y_ax_limit = int(max(list(freq_dict.values())) + 3)
    if y_ax_limit < ax.get_ylim()[1]:
        y_ax_limit = ax.get_ylim()[1]


    if y_ax_limit > 1000:
        y_ax_limit += 100
    elif y_ax_limit > 100:
        y_ax_limit += 10


    if fig_text:
        for p in pps:
            height = p.get_height()
            ax.annotate('{}'.format(height),
                        xy=(p.get_x() + p.get_width()/2, height),
                        xytext=(0, 3), # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #print(freq_dict)
    #print(ax.get_xlim()[0], ax.get_xlim()[1])
    # if ax.get_xlim()[0] < -1 and ax.get_xlim()[1] < 1 and ax.get_xlim()[1] > 0:

    #ticks_list = np.arange(-1, 0, 1)
    #else:
    ticks_list = np.arange(math.ceil(ax.get_xlim()[0]), math.floor(ax.get_xlim()[1]+1), 1)

    #print(ticks_list)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_xticks(ticks_list, ticks_list, rotation=90, fontsize=fontsize)

    #print(ticks_list)

    #ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 3)

    ax.set_ylim(ax.get_ylim()[0], y_ax_limit)

    props = dict(boxstyle="round", facecolor="wheat", alpha=1.0)

    if progression_position == 0:
        prog_text = "Criteria - missed transitions = "
        #prog_text = "Criteria - missed SPMS = "
    else:
        prog_text = "Model - missed transitions = "
        #prog_text = "Model - missed SPMS = "

    ax.text(0.05+offset_right, 0.95 - (progression_position/9),prog_text+str(no_progression_found), transform=ax.transAxes, fontsize=fontsize, verticalalignment="top", bbox=props)
    #ax.text(0.05+offset_right, 0.95 - (progression_position/15),prog_text+str(no_progression_found), transform=ax.transAxes, fontsize=fontsize, verticalalignment="top", bbox=props)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    #ax.set_ylim(0,ax.get_ylim()[1]+5)
    ax.set_ylim(0,y_ax_limit)

    if type(ax) == list:
        plt.show()
