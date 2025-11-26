import os
import numpy as np

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ParameterGrid


class GridSearch:

    def __init__(self, estimator, param_grid, scoring='neg_mean_squared_error'):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.results_ = []
        self.best_params_ = None
        self.best_estimator_ = None
        self.best_score_ = -np.inf
        self.best_estimator_ = None

    def fit(self, X_train, y_train, X_val, y_val):

        param_combinations = list(ParameterGrid(self.param_grid))

        for i, params in enumerate(param_combinations):

            model = self.estimator.__class__(**{**self.estimator.get_params(), **params})

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)

            train_mse = mean_squared_error(y_train, y_train_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_val, y_val_pred)

            if self.scoring == 'neg_mean_squared_error':
                main_score = -val_mse
            elif self.scoring == 'r2':
                main_score = val_r2
            else:
                main_score = -val_mae

            result = {
                'params': params,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'main_score': main_score,
                'model': model
            }

            self.results_.append(result)

            if main_score > self.best_score_:
                self.best_score_ = main_score
                self.best_params_ = params
                self.best_estimator_ = model

            if (i + 1) % 10 == 0:
                print(f" {i + 1}/{len(param_combinations)} is Done!")

        return self

    def get_best_results(self):

        return {
            'best_params': self.best_params_,
            'best_score': self.best_score_,
            'best_estimator': self.best_estimator_
        }


def generate_scaffold(smiles, include_chirality=False):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)


def eval_model(model, X, y):

    y_pred = model.predict(X)
    return {
        'r2': r2_score(y, y_pred),
        'mse': mean_squared_error(y, y_pred),
        'rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred)
    }, y_pred


def grid_search_model(model, X_train, y_train, X_val, y_val, name, param_grid):

    model_default = clone(model)

    grid_search = GridSearch(
        estimator=model_default,
        param_grid=param_grid,
        scoring='r2'
    )

    grid_search.fit(X_train, y_train, X_val, y_val)

    model_grid = grid_search.best_estimator_
    train_metrics, pred_train = eval_model(model_grid, X_train, y_train)
    val_metrics, pred_val = eval_model(model_grid, X_val, y_val)

    print("=" * 50)
    print(f"\n{name} - best Parameters:")
    print(grid_search.best_params_)
    print(f"  Train Set: R²={train_metrics['r2']:.4f}, MSE={train_metrics['mse']:.4f}, "
          f"RMSE={train_metrics['rmse']:.4f}, MAE={train_metrics['mae']:.4f}")
    print(f"  Validation Set: R²={val_metrics['r2']:.4f}, MSE={val_metrics['mse']:.4f}, "
          f"RMSE={val_metrics['rmse']:.4f}, MAE={val_metrics['mae']:.4f}")

    return model_grid, grid_search.best_params_, pred_train, pred_val


def get_model(GLOBAL_SEED, use_model):

    n_cores = max(1, (os.cpu_count() or 1) // 2)
    models = [
        (Ridge(random_state=GLOBAL_SEED), 'Ridge Regression', {
            'alpha': np.logspace(-3, 3, 7),
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }),
        (Lasso(random_state=GLOBAL_SEED), 'Lasso Regression', {
            'alpha': np.logspace(-3, 3, 7),
            'selection': ['cyclic', 'random']
        }),
        (ElasticNet(random_state=GLOBAL_SEED), 'ElasticNet Regression', {
            'alpha': np.logspace(-3, 3, 7),
            'l1_ratio': np.linspace(0.1, 1.0, 10),
            'selection': ['cyclic', 'random']
        }),
        (BayesianRidge(), 'Bayesian Ridge Regression', {
            'alpha_1': np.logspace(-6, 2, 9),
            'alpha_2': np.logspace(-6, 2, 9),
            'lambda_1': np.logspace(-6, 2, 9),
            'lambda_2': np.logspace(-6, 2, 9)
        }),
        (KNeighborsRegressor(), 'K-Nearest Neighbors', {
            'n_neighbors': list(range(1, 21, 2)),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }),
        (RandomForestRegressor(random_state=GLOBAL_SEED, n_jobs=n_cores), 'Random Forest',
         {
             'n_estimators': list(range(100, 1001, 100)),
             'criterion': ['squared_error', 'absolute_error'],
             'max_depth': list(range(1, 11)),
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'max_features': ['sqrt', 'log2']
         }),
        (AdaBoostRegressor(random_state=GLOBAL_SEED), 'AdaBoost', {
            'n_estimators': list(range(100, 1001, 100)),
            'learning_rate': np.logspace(-3, 0, 4),
            'loss': ['linear', 'square', 'exponential'],
            'estimator': [DecisionTreeRegressor(max_depth=i) for i in [1, 3, 5]]
        }),
        (xgb.XGBRegressor(random_state=GLOBAL_SEED, n_jobs=n_cores), 'XGBoost', {
            'n_estimators': list(range(100, 1001, 100)),
            'learning_rate': np.logspace(-3, 0, 4),
            'max_depth': list(range(1, 11)),
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3, 0.4],
            'reg_alpha': [0, 0.1, 1, 10],
            'reg_lambda': [0, 0.1, 1, 10]
        })
    ]
    model_idx = [i for i, (_, name, _) in enumerate(models) if name == use_model][0]

    return models[model_idx][0], models[model_idx][2]


def train_with_params(base_model, X_train, y_train, X_val, y_val, name, fixed_params):

    model_tuned = clone(base_model)
    model_tuned.set_params(**fixed_params)
    model_tuned.fit(X_train, y_train)

    tuned_train, pred_train = eval_model(model_tuned, X_train, y_train)
    tuned_val, pred_val = eval_model(model_tuned, X_val, y_val)

    print(f" {name} Model Fixed Parameters: {fixed_params}")
    print(
        f"  Fixed Train: R²={tuned_train['r2']:.4f}, MSE={tuned_train['mse']:.4f}, RMSE={tuned_train['rmse']:.4f}, MAE={tuned_train['mae']:.4f}")
    print(
        f"  Fixed Validation: R²={tuned_val['r2']:.4f}, MSE={tuned_val['mse']:.4f}, RMSE={tuned_val['rmse']:.4f}, MAE={tuned_val['mae']:.4f}")

    return model_tuned, str(fixed_params), pred_train, pred_val