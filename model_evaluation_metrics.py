import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curve(y_test, y_probs):
    """
    Hercodes-ux: Professional Metrics Visualization.
    Generates an ROC-AUC curve to evaluate classification performance.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Real News marked as Fake)')
    plt.ylabel('True Positive Rate (Fake News correctly caught)')
    plt.title('Receiver Operating Characteristic (ROC) - Hercodes Analysis')
    plt.legend(loc="lower right")
    
    # Save the proof for GitHub
    plt.savefig("roc_curve_output.png")
    print("✅ SUCCESS: ROC Curve generated and saved as 'roc_curve_output.png'")
    plt.show()

# --- THE EXECUTION BLOCK (This was missing!) ---
if __name__ == "__main__":
    # We create fake data to test if the chart draws correctly
    # 0 = Real, 1 = Fake
    y_test_mock = [0, 0, 1, 1] 
    # High probabilities for fake news (1)
    y_probs_mock = [0.1, 0.4, 0.35, 0.8] 
    
    plot_roc_curve(y_test_mock, y_probs_mock)