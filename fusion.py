# 模型融合
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from catboost import CatBoostRegressor
import time 
from sklearn.metrics import mean_absolute_error
import numpy as np

def fit_catboost(x_train, y_train, x_val=None, y_val=None):
    model = CatBoostRegressor(iterations=10000,
                             depth=3,
                             learning_rate=0.02,
                             loss_function='MAE',
                             early_stopping_rounds=50,
                             verbose=False)
    if x_val is not None and y_val is not None:
        model.fit(x_train, 
                  y_train, 
                  eval_set=(x_val, y_val), 
                  use_best_model=True,)
    else:
        model.fit(x_train, 
                  y_train)
    model.fit(x_train, y_train)
    return model

def fit_ets(x_train, y_train):
    model = ExtraTreesRegressor(n_estimators=1000, 
                                n_jobs=-1)
    model.fit(x_train, y_train)
    return model

def fit_rf(x_train, y_train):
    """拟合随机森林模型
    Parameters
    ----------
    x_train: array_like
    y_train: array_like
    
    Returns
    -------
    model: object, 拥有predict方法的模型对象
    """
    model = RandomForestRegressor(n_estimators=1000, 
                                  n_jobs=-1)
    model.fit(x_train, y_train)
    return model

def fit_xgb(x_train, y_train, x_val, y_val):
    params = {'learning_rate': 0.02,
              'n_estimators': 10000,
              'early_stopping_rounds': 50,
              'verbosity': 0, }
    model = XGBRegressor(**params)
    if x_val is not None and y_val is not None:
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)
    else:    
        model.fit(x_train, y_train)
    return model

def fit_lgbm(x_train, y_train, x_val=None, y_val=None):
    params = {'learning_rate': 0.02,
              'n_estimators': 10000,
              'early_stopping_round': 50,
              'verbosity': -1, }
    model = LGBMRegressor(**params)
    if x_val is not None and y_val is not None:
        model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=-1)
    else:
        model.fit(x_train, y_train)
    return model

def train_predict_pipeline(x, y, cv, fit_func, ):
    """
    Parameters
    ----------
    x: array_like
    y: array_like
    cv: list[array_like], shape (#folds, )
    fit_func: function, 函数对象
        参数必须要能够接受x_train, y_train, 
        并且返回模型, 模型可以调用predict方法

    Returns
    -------
    mae: float, mean absolute error in validation set
    y_pred: array_like of shape (#training samples, ), prediction in validation set
    model_list: List[object], model list
    """
    print(fit_func.__name__)
    model_list = []
    begin_time = time.time()
    x = np.array(x)
    y = np.array(y).reshape(-1)
    # y_log = np.log(y)
    y_pred = np.empty_like(y)
    # y_pred_log = np.empty_like(y_log)   # 将标签进行对数变换
    for n_fold, (train_idx,val_idx) in enumerate(cv):
        x_train = x[train_idx]
        x_val = x[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        if 'xgb' in fit_func.__name__ or 'lgbm' in fit_func.__name__:
            model = fit_func(x_train, y_train, x_val, y_val)
        else:
            model = fit_func(x_train, y_train)
        y_pred[val_idx] = model.predict(x_val)
        model_list.append(model)
        print(f"训练完第{n_fold}折")
    # y_pred = np.exp(y_pred)
    mae = mean_absolute_error(y, y_pred)
    end_time = time.time()
    print(f"elapsed time:\t{end_time-begin_time:.2f}" )
    print(f"mae:\t{mae:.4f}" )
    return mae, y_pred, model_list

def fusion_func(x_train, y_train, ):
    pass