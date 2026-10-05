import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(tp, fp, tn, fn, plot_lor=True, ax=None, fontsize=30):
    confusion_matrix = np.array([[tp, fp],
                                 [fn, tn]])

    if plot_lor:
        title = "Lorscheider criteria"
        y_label = "Assigned label"
    else:
        title = "Model prediction"
        y_label = "Predicted label"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['SPMS', 'RRMS'],
                yticklabels=['SPMS', 'RRMS'], annot_kws={"size": fontsize})
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel(y_label, fontsize=fontsize)
    ax.set_xlabel('Diagnosis', fontsize=fontsize)
    return ax


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator != 0 else None


def calculate_metrics(tp, fp, tn, fn):
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    f1 = None
    if precision is not None and sensitivity is not None:
        f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)

    metrics = {"Sensitivity (Recall)": sensitivity,
               "Specificity": specificity,
               "Precision/PPV": precision,
               "NPV": npv,
               "Accuracy": accuracy,
               "F1": f1}
    for name, value in metrics.items():
        if value is None:
            print(f"{name}: Cannot be calculated, denominator is zero.")
        else:
            print(f"{name}: {value:.4f}")
    return metrics


def get_cm_from_lor_stats_df(input_df, ax1=None, ax2=None):
    """Confusion matrices (SPMS as the positive class) for the Lorscheider criteria (ax1)
    and the model (ax2), from the output of ``get_lorscheider_stats``."""
    all_neg = input_df["rrms_gt"].values.sum()
    all_pos = np.sum([input_df["spms_gt"].values.sum(), input_df["transition_gt"].values.sum()])

    lor_tp = np.sum([input_df["spms_lor"].values.sum(), input_df["transition_lor"].values.sum()])
    lor_tn = all_neg - input_df["rrms_lor"].values.sum()
    lor_fp = all_neg - lor_tn
    lor_fn = all_pos - lor_tp

    ax1 = plot_confusion_matrix(lor_tp, lor_fp, lor_tn, lor_fn, ax=ax1)
    print("Criteria")
    calculate_metrics(lor_tp, lor_fp, lor_tn, lor_fn)

    model_tp = np.sum([input_df["spms"].values.sum(), input_df["transition"].values.sum()])
    model_tn = all_neg - input_df["rrms"].values.sum()
    model_fp = all_neg - model_tn
    model_fn = all_pos - model_tp

    ax2 = plot_confusion_matrix(model_tp, model_fp, model_tn, model_fn, plot_lor=False, ax=ax2)
    print("Model")
    calculate_metrics(model_tp, model_fp, model_tn, model_fn)
    return ax1, ax2
