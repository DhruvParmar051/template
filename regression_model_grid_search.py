from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

X_test = [] , X_train = [], y_train = [], y_test =[]

reg_models = {
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, eval_metric="rmse")
}

reg_models = {
    
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, eval_metric="rmse") # tree_method='gpu_hist' predictor='gpu_predictor',
}
reg_param_grids = {

    "GradientBoosting": {
        "n_estimators": [200, 250,300],
        "learning_rate": [0.05, 0.1],        
        "max_depth": [3, 4],                  
        "subsample": [0.8, 1.0],          
    },

    "XGBoost": {
        "n_estimators": [200, 250,300],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
}



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

results_reg = {}

for name, model in reg_models.items():
    print(f"\nRunning {name}...")
    grid = GridSearchCV(
    model,
        reg_param_grids[name],
        cv=3,
        scoring="r2",
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
        
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
        
    metrics = {
        "r2_score": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mean_squared_error(y_test, y_pred)
    }
        
    results_reg[name] = {
        "best_params": grid.best_params_,
        "metrics": metrics
    }
        
    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Metrics: {metrics}")
