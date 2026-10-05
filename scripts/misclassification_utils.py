import copy
import math
import itertools

import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.power import NormalIndPower


# ----------------------------------------------------------------------------
# Building per-visit arrays
# ----------------------------------------------------------------------------
def get_per_patient_arrays(input_df, value_columns, time_basis="years"):
    """For every visit returns its position relative to the patient's most recent visit
    (``time_basis="years"``: years before the last visit, ``"visits"``: number of visits
    before the last visit) together with the values of ``value_columns``.
    ``input_df`` is expected to be sorted on visit_date."""
    time_array = []
    value_arrays = {column: [] for column in value_columns}
    for pcode in tqdm.tqdm(input_df["patient_code"].unique()):
        sub_df = input_df[input_df["patient_code"] == pcode]
        if time_basis == "years":
            time_array.extend(((sub_df["visit_date"].to_list()[-1] - sub_df["visit_date"]) / pd.Timedelta(days=365.25)).to_list())
        elif time_basis == "visits":
            time_array.extend([i for i in range(len(sub_df) - 1, -1, -1)])
        else:
            raise ValueError("time_basis should be 'years' or 'visits'")
        for column in value_columns:
            value_arrays[column].extend(sub_df[column].to_list())

    time_array = np.array(time_array)
    value_arrays = {column: np.array(values) for column, values in value_arrays.items()}
    return time_array, value_arrays


def filter_by_max_time(time_array, value_arrays, max_time):
    condition = time_array <= max_time
    return time_array[condition], {column: values[condition] for column, values in value_arrays.items()}


def get_mislabelled_mask(diagnosis_arr, pred_arr):
    return ((diagnosis_arr == 1) & (pred_arr != 1)) | ((diagnosis_arr == 0) & (pred_arr != 0))


# ----------------------------------------------------------------------------
# Counting per bin
# ----------------------------------------------------------------------------
def get_count_of_mis_per_visit(visit_array_mis, visits_order_array):
    max_visits = math.ceil(max(visits_order_array))
    return [len(visit_array_mis[visit_array_mis == i]) for i in range(0, max_visits)]


def get_count_of_mis_per_year(year_array_mis, years_array):
    max_years = math.ceil(max(years_array))
    return [len(year_array_mis[(year_array_mis >= i) & (year_array_mis < i + 1)]) for i in range(0, max_years)]


def get_total_count_per_visit(visits_order_array):
    return np.array([len(visits_order_array[visits_order_array == i]) for i in range(math.ceil(max(visits_order_array)))])


def get_total_count_per_year(years_array):
    return np.array([len(years_array[(years_array >= i) & (years_array < i + 1)]) for i in range(math.ceil(max(years_array)))])


def get_proportion_of_mis_vs_correct(year_array_mis, year_array_cor, years_array):
    max_years = math.ceil(max(years_array))
    proportion_mislabelled = []
    for i in range(0, max_years):
        count_mis = len(year_array_mis[(year_array_mis >= i) & (year_array_mis < i + 1)])
        count_corr = len(year_array_cor[(year_array_cor >= i) & (year_array_cor < i + 1)])
        if count_mis + count_corr != 0:
            proportion_mislabelled.append(count_mis / (count_mis + count_corr))
        else:
            proportion_mislabelled.append(0)
    return proportion_mislabelled


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def p_to_stars(p):
    return "ns" if p >= 0.05 else ("*" if p >= 0.01 else ("**" if p >= 0.001 else ("***" if p >= 1e-4 else "****")))


def add_sig(ax, x1, x2, y, h, p, fontsize=12):
    """Draws a significance bracket from x1 to x2 at y (data coords)."""
    star = p_to_stars(p)
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color="black")
    ax.text((x1 + x2) / 2, y + h, star, ha="center", va="bottom", fontsize=fontsize)


def _style_top_axis_grid(ax, ax_top):
    ax_top.set_axisbelow(True)       # grid below artists on ax_top
    ax_top.grid(True, alpha=0.4, color="black", linestyle="--")
    ax_top.set_zorder(0)             # put the twin axis behind the main axis
    ax.set_zorder(1)                 # ensure the data axis (and bars) are above
    ax_top.patch.set_visible(False)  # no opaque face that could cover bars
    ax.patch.set_visible(False)


def plot_years_mislabelled(input_mislabelled_array, input_per_year_count_array, y_lim=0.3, inp_title="", fig_title=""):
    """Misclassification rate (with Wilson 95% CI) per year before the most recent visit."""
    k = np.asarray(copy.deepcopy(input_mislabelled_array), dtype=float)       # mislabel counts per bin
    n = np.asarray(copy.deepcopy(input_per_year_count_array), dtype=float)    # total samples per bin

    labels = [f"[{i}-{i+1})" for i in range(len(k))]
    labels[-1] = "[" + str(len(k) - 1) + "-" + str(len(n)) + ")"

    rate = np.where(n > 0, k / n, np.nan)
    ci_lo, ci_hi = proportion_confint(k, n, method="wilson")
    yerr = np.vstack([rate - ci_lo, ci_hi - rate])

    bar_width = 0.7
    x = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rate, yerr=yerr, capsize=5, width=bar_width)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.set_ylim(0, y_lim)
    ax.set_ylabel("Misclassification rate")
    ax.set_xlabel("Years since recent recorded visit")

    # Bonferroni adjusted significance thresholds for all pairwise bin comparisons
    count_combinations = len(list(itertools.combinations(range(len(k)), 2)))
    print("Num combinations =", count_combinations)
    for alpha in [0.05, 0.01, 0.001, 0.0001]:
        print("Adjusted p-values for", alpha, alpha / count_combinations)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    centers = x + (bar_width / 7)
    if x.size > 0:
        ax_top.set_xticks(centers - 0.1)
        ax_top.set_xticklabels([f"{int(c)}" for c in n], rotation=45)
    else:
        ax_top.set_xticks([])
    ax_top.set_xlabel("Number of visits during each time interval")

    _style_top_axis_grid(ax, ax_top)
    ax.set_title(inp_title, fontsize=16)
    ax.grid(None)
    plt.tight_layout()
    if fig_title != "":
        plt.savefig(fig_title, dpi=600, bbox_inches='tight')
    plt.show()


def plot_grouped_mislabel_bars(pred_count_mis, lor_count_mis, total_count, bottom_labels,
                               pred_errorbar_offset=0.0, y_lim=None, bar_width=0.2):
    """Side by side misclassification rate of the model and the Lorscheider criteria per bin."""
    data = [pred_count_mis / total_count, lor_count_mis / total_count]
    labels = ["Misclassified prediction", "Misclassified lorscheider criteria"]

    x = np.arange(max(len(lst) for lst in data))
    fig, ax = plt.subplots(figsize=(15, 5))
    for i, lst in enumerate(data):
        ax.bar(x + i * bar_width, lst, width=bar_width, label=labels[i])

    centers = x + bar_width * (len(data) - 1) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(bottom_labels, rotation=45, ha='right')

    ax.set_xlabel('Years (bins)')
    ax.set_ylabel('Values')
    ax.set_title('Bar Chart for Two Lists with Top Axis Year Counts')
    ax.legend()

    ci_low_pred, ci_high_pred = proportion_confint(pred_count_mis, total_count, method='wilson')
    ci_low_lor, ci_high_lor = proportion_confint(lor_count_mis, total_count, method='wilson')

    rate_pred = pred_count_mis / total_count
    ax.errorbar(x + pred_errorbar_offset, rate_pred, yerr=[rate_pred - ci_low_pred, ci_high_pred - rate_pred],
                fmt='o', capsize=5, label='Rate ± 95% CI', color="orange")

    rate_lor = lor_count_mis / total_count
    ax.errorbar(x + bar_width, rate_lor, yerr=[rate_lor - ci_low_lor, ci_high_lor - rate_lor],
                fmt='o', capsize=5, label='Lorscheider rate ±95% CI', color="orange")

    # top axis: number of visits per bin
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    if len(total_count) > 0:
        if len(total_count) == len(centers):
            top_tick_positions = centers
        else:
            top_tick_positions = np.linspace(centers[0], centers[-1], len(total_count))
        ax_top.set_xticks(top_tick_positions)
        ax_top.set_xticklabels([f"{c}" for c in total_count], rotation=45)
    else:
        ax_top.set_xticks([])
    ax_top.set_xlabel("Year (count)")

    if y_lim is not None:
        ax.set_ylim(0, y_lim)
    plt.tight_layout()
    plt.show()


def plot_rate_with_significance(count_mis, total_count, n_compared_bins=7,
                                xlabel="Visit order", top_xlabel="Visit (count)", center_offset=0.1):
    """Mislabel rate per bin with Wilson 95% CI, and two-sided z-test significance
    brackets between the first ``n_compared_bins`` bins."""
    k = np.asarray(count_mis, dtype=float)
    n = np.asarray(total_count, dtype=float)
    labels = [f"{i}" for i in range(len(k))]

    rate = np.where(n > 0, k / n, np.nan)
    ci_lo, ci_hi = proportion_confint(k, n, method="wilson")
    yerr = np.vstack([rate - ci_lo, ci_hi - rate])

    x = np.arange(len(k))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, rate, yerr=yerr, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha='right')
    ax.set_ylim(0, 1.05 * np.nanmax(ci_hi))
    ax.set_ylabel("Mislabel rate")
    ax.set_xlabel(xlabel)

    pairs = list(itertools.combinations(range(n_compared_bins), 2))

    offset = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])  # vertical spacing
    levels = {}  # stack brackets if they overlap
    for (i, j) in pairs:
        if n[i] == 0 or n[j] == 0:
            continue
        stat, pval = proportions_ztest([k[i], k[j]], [n[i], n[j]], alternative="two-sided")
        if pval > 0.05:
            continue

        top_ij = max(ci_hi[i], ci_hi[j])
        level = levels.get((min(i, j), max(i, j)), 0)
        for (a, b), L in list(levels.items()):
            if (min(i, j) <= b and max(i, j) >= a) and L >= level:
                level = L + 1
        levels[(min(i, j), max(i, j))] = level

        y = top_ij + offset * (1.8 + level * 3.2)
        add_sig(ax, i, j, y, h=offset * 0.8, p=pval)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())

    centers = x + center_offset
    ax.set_xticks(centers)
    if x.size > 0:
        ax_top.set_xticks(centers - 0.1)
        ax_top.set_xticklabels([f"{c}" for c in total_count], rotation=45)
    else:
        ax_top.set_xticks([])
    ax_top.set_xlabel(top_xlabel)

    _style_top_axis_grid(ax, ax_top)
    ax.grid(None)
    plt.tight_layout()
    plt.show()


def plot_per_label_misclassification(data, labels, years_sub_array_rrms, years_sub_array_spms, years_array, bar_width=0.2):
    """Grouped bars of the per label misclassification proportion per year bin.
    Top axis shows the (RRMS, SPMS) visit counts per bin."""
    n_bins = math.ceil(max(years_array))
    count_rrms = [int(len(years_sub_array_rrms[(years_sub_array_rrms >= i) & (years_sub_array_rrms < i + 1)])) for i in range(n_bins)]
    count_spms = [int(len(years_sub_array_spms[(years_sub_array_spms >= i) & (years_sub_array_spms < i + 1)])) for i in range(n_bins)]

    max_len = max(len(lst) for lst in data)
    padded_data = [list(lst) + [0] * (max_len - len(lst)) for lst in data]
    x = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, lst in enumerate(padded_data):
        ax.bar(x + i * bar_width, lst, width=bar_width, label=labels[i])

    centers = x + bar_width * (len(padded_data) - 1) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels([f'{i}-{i+1}' for i in range(max_len)], rotation=45, ha='right')

    ax.set_xlabel('Years (bins)')
    ax.set_ylabel('Values')
    ax.set_title('Bar Chart for Two Lists with Top Axis Year Counts')
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    if n_bins > 0:
        if n_bins == len(centers):
            top_tick_positions = centers
        else:
            top_tick_positions = np.linspace(centers[0], centers[-1], n_bins)
        ax_top.set_xticks(top_tick_positions)
        ax_top.set_xticklabels([f"{c_rr, c_sp}" for c_rr, c_sp in zip(count_rrms, count_spms)], rotation=45)
    else:
        ax_top.set_xticks([])
    ax_top.set_xlabel("Year (count)")

    plt.tight_layout()
    plt.show()


def print_pairwise_power(count_mis, total_count, alpha=0.05):
    """Statistical power of the pairwise two-proportion comparison between bins."""
    analysis = NormalIndPower()
    for i in range(len(count_mis)):
        for j in range(i + 1, len(count_mis)):
            n1 = total_count[i]
            n2 = total_count[j]
            if n1 > 0 and n2 > 0:
                p1 = count_mis[i] / n1
                p2 = count_mis[j] / n2
                eff = sm.stats.proportion_effectsize(p1, p2)
                pw = analysis.solve_power(eff, nobs1=min(n1, n2), ratio=n2 / n1, alpha=alpha)
                print(f"{i}-{j}: power={pw:.2f}")
