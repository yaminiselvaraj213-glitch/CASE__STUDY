import numpy as np
from sklearn.metrics import mean_squared_error
import sys
import os

sys.path.append(r"c:\Users\YAMINI\Downloads\archive\CMaps")

from code_1_preprocessing import prepare_data
from code_2_model import build_model
from code_4_plotting import plot_loss, plot_predictions, plot_roc_pr_curves

BATCH_SIZE = 64
EPOCHS = 10

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    h = y_pred - y_true
    s_score = np.sum(np.where(h < 0, np.exp(-h/13)-1, np.exp(h/10)-1))
    
    eol_indices = np.where(y_true <= 20)[0]
    if len(eol_indices) > 0:
        eol_error = y_pred[eol_indices] - y_true[eol_indices]
        reliability_std = np.std(eol_error)
    else:
        reliability_std = 0
        
    cost_late = 500
    cost_early_per_cycle = 10
    costs = np.where(h < 0, np.abs(h) * cost_early_per_cycle, cost_late)
    total_model_cost = np.sum(costs)
    total_baseline_cost = len(y_true) * cost_late
    cb_ratio = (total_baseline_cost - total_model_cost) / total_baseline_cost
    
    return rmse, s_score, reliability_std, cb_ratio

if __name__ == "__main__":
    print("--- STEP 1: Preparing Data ---")
    X_train, y_train, X_test, y_test, window_size, n_features = prepare_data()
    
    print("\n--- STEP 2: Building Model ---")
    model = build_model((window_size, n_features))
    
    print("\n--- STEP 3: Training ---")
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, verbose=1)
    
    plot_loss(history)
    
    print("\n--- STEP 4: Evaluation ---")
    y_pred = model.predict(X_test).flatten()
    
    plot_predictions(y_test, y_pred)
    
    plot_roc_pr_curves(y_test, y_pred, threshold=30)
    
    rmse, s, rel_std, cb = calculate_metrics(y_test, y_pred)
    
    print(f"\nfinal RESULTS:")
    print(f"RMSE: {rmse:.4f}")
    print(f"S-Score: {s:.4f}")
    print(f"Reliability (StdDev @ EOL): {rel_std:.4f}")
    print(f"Cost-Benefit Ratio: {cb:.4f}")
    
    with open('results_modular.txt', 'w') as f:
        f.write(f"RMSE: {rmse:.4f}\nS-Score: {s:.4f}\nReliability: {rel_std:.4f}\nCB-Ratio: {cb:.4f}")
    print("Results saved to results_modular.txt")
