# 模型融合
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import time 
from sklearn.metrics import mean_absolute_error
import numpy as np

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