import numpy as np
import matplotlib.pyplot as plt

# Requires pharmbio plot_utils (https://github.com/pharmbio/plot_utils) on the PYTHONPATH
from pharmbio.cp import metrics, plotting


def plot_cp_plots(model, input_df, x_columns, y_label, significance=0.07):
    """Overview of conformal prediction metrics: calibration, confusion matrices and label distribution."""
    p_values = model.predict(input_df[x_columns].values)
    fig, axes = plt.subplots(3, 2, figsize=(12, 17))
    sign = str(significance)
    class_labels = ['RRMS', 'SPMS']
    true_labels = input_df[y_label].values.T.flatten()

    plotting.plot_calibration_clf(true_labels, p_values, ax=axes[0, 0], labels=class_labels)
    # 70-95% confidence only
    plotting.plot_calibration_clf(true_labels, p_values, ax=axes[0, 1], sign_vals=np.arange(0.05, 0.31, 0.01), chart_padding=.025, labels=class_labels)
    plotting.plot_confusion_matrix_bubbles(metrics.confusion_matrix(true_labels, p_values, significance, labels=class_labels), title="Bubble plot (sign=" + sign + ")", ax=axes[1, 0])
    plotting.plot_confusion_matrix_heatmap(metrics.confusion_matrix(true_labels, p_values, significance, labels=class_labels, normalize_per_class=True), vmin=0, vmax=1, ax=axes[1, 1], title="Normalized heatmap (sign=" + sign + ")")
    plotting.plot_label_distribution(y_true=true_labels, p_values=p_values, ax=axes[2, 0])
    plotting.plot_label_distribution(y_true=true_labels, p_values=p_values, ax=axes[2, 1], display_incorrect=True, title='Label distribution')
    fig.suptitle('Metrics plots', fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
