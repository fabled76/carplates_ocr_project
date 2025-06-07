from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(y_true, y_pred, class_names):
    """
    Evaluate model performance: Accuracy, Classification Report, and Confusion Matrix
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names)


def plot_confusion_matrix(cm, class_names):
    """
    Plot confusion matrix using heatmap.
    """
    mask = np.eye(cm.shape[0], dtype=bool)
    max_misclassification = np.max(cm[~mask]) if np.sum(~mask) > 0 else 1

    # Normalize off-diagonal elements for better visualization of confusion
    cm_display = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    cm_display = np.round(cm_display, decimals=2)

    # Create figure
    plt.figure(figsize=(12, 10))

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Plot heatmap
    sns.heatmap(
        cm_display,
        annot=cm,
        fmt='d',
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count'},
        annot_kws={'size': 10},
        square=True,
        linewidths=0.5,
        linecolor='white'
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0.1 * max_misclassification:
                plt.text(j + 0.5, i + 0.5, cm[i, j],
                         ha='center', va='center',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title("Confusion Matrix", pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()
