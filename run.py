import os
import joblib
import numpy as np
import pandas as pd

def run(split_dir, GLOBAL_SEED, data, use_model, params, grid_search=False):
    X_train, X_val, X_test, y_train, y_val, y_test, train_idx, val_idx, test_idx = split_data(data, random_state=GLOBAL_SEED)

    save_splits(data, train_idx, val_idx, test_idx, split_dir)
    print(f"Split datasets saved to {split_dir}/")

    model, param_grid = get_model(GLOBAL_SEED, use_model)

    if grid_search:

        end_model, grid_params, pred_train, pred_val = grid_search_model(model, X_train, y_train, X_val, y_val, use_model, param_grid)

    else:
        end_model, tuned_params, pred_train, pred_val = train_with_params(model, X_train, y_train, X_val, y_val, use_model, params)

    test_metrics, pred_test = eval_model(end_model, X_test, y_test)

    print \
        (f"\n{use_model} - Test Set: R²={test_metrics['r2']:.4f}, MSE={test_metrics['mse']:.4f}, RMSE={test_metrics['rmse']:.4f}, MAE={test_metrics['mae']:.4f}")

    df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df_train['label'] = y_train
    df_train['pred'] = pred_train
    df_train.to_csv(os.path.join(result_dir, f"{use_model}_train_grid_{grid_search}.csv"), index=False)

    df_val['label'] = y_val
    df_val['pred'] = pred_val
    df_val.to_csv(os.path.join(result_dir, f"{use_model}_val_grid_{grid_search}.csv"), index=False)

    df_test['label'] = y_test
    df_test['pred'] = pred_test
    df_test.to_csv(os.path.join(result_dir, f"{use_model}_test_grid_{grid_search}.csv"), index=False)

if __name__ == "__main__":
    # Global settings and seeds
    GLOBAL_SEED = 850558
    np.random.seed(GLOBAL_SEED)

    output_dir = "model_results"
    split_dir = os.path.join(output_dir, "split_data")# Save splits under model_results/split_data
    result_dir = os.path.join(output_dir, "result")# Save result under model_results/result
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    MODEL_NAMES = {k: k for k in ['Ridge Regression', 'Lasso Regression', 'ElasticNet Regression',
                                  'Bayesian Ridge Regression', 'K-Nearest Neighbors', 'Random Forest',
                                  'AdaBoost', 'XGBoost']}
    METRIC_COLS = ['train_r2', 'val_r2', 'test_r2', 'train_mse', 'val_mse', 'test_mse',
                   'train_rmse', 'val_rmse', 'test_rmse', 'train_mae', 'val_mae', 'test_mae']

    METRIC_NAMES = {k: k.replace('_', ' ').title() for k in METRIC_COLS}
    data = pd.read_csv(r"./S1_Data.csv")

    fixed_params = {'Ridge Regression': {'alpha': np.float64(1000.0), 'solver': 'sparse_cg'},
                    'Lasso Regression': {'alpha': np.float64(0.001), 'selection': 'random'},
                    'ElasticNet Regression': {'alpha': np.float64(0.001), 'l1_ratio': np.float64(1.0), 'selection': 'random'},
                    'Bayesian Ridge Regression': {'alpha_1': np.float64(100.0), 'alpha_2': np.float64(1.0), 'lambda_1': np.float64(1.0), 'lambda_2': np.float64(0.001)},
                    'K-Nearest Neighbors': {'algorithm': 'auto', 'n_neighbors': 11, 'p': 1, 'weights': 'distance'},
                    'Random Forest': {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 8, 'criterion': 'squared_error'},
                    'AdaBoost': {'estimator': DecisionTreeRegressor(max_depth=5, random_state=850558), 'learning_rate': np.float64(0.1), 'loss': 'exponential', 'n_estimators': 100},
                    'XGBoost': {'colsample_bytree': 0.6, 'gamma': 1, 'learning_rate': np.float64(0.1), 'max_depth': 3, 'n_estimators': 200, 'reg_alpha': 1, 'reg_lambda': 0.1, 'subsample': 0.6}
                    }

    use_model = "Random Forest" # model name in "MODEL_NAMES"

    run(split_dir, GLOBAL_SEED, data, use_model, fixed_params[use_model], grid_search=False)