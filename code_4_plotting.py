import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
    plt.title('Model Loss vs. Epochs')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph_loss.png')
    print("Saved graph_loss.png")
    plt.show()

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, s=10)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title('Predicted vs Actual RUL')
    plt.xlabel('Actual RUL (Cycles)')
    plt.ylabel('Predicted RUL (Cycles)')
    plt.grid(True)
    plt.savefig('graph_predictions.png')
    print("Saved graph_predictions.png")
    plt.show()

def plot_roc_pr_curves(y_true, y_pred, threshold=30):
    binary_true = (y_true <= threshold).astype(int)
    
    if len(np.unique(binary_true)) < 2:
        print("Warning: Only one class present in test set. ROC/PR curves skipped.")
        return
    
    max_rul_val = 130 
    failure_score = max_rul_val - y_pred
    
    fpr, tpr, _ = roc_curve(binary_true, failure_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (Fail Threshold={threshold})')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('graph_roc.png')
    print("Saved graph_roc.png")
    plt.show()
    
    precision, recall, _ = precision_recall_curve(binary_true, failure_score)
    avg_precision = average_precision_score(binary_true, failure_score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (Fail Threshold={threshold})')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig('graph_pr_curve.png')
    print("Saved graph_pr_curve.png")
    plt.show()
