# 脚本 一键运行
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
import joblib
import pandas as pd
import numpy as np
from fusion import fit_lgbm, fit_xgb, fit_rf, fit_catboost, fit_ets, train_predict_pipeline
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # 读取特征工程后的数据
    testfile = "data/used_car_testB_engineering_0410.csv"
    trainfile = 'data/used_car_train_engineering_0410.csv'

    train_df = pd.read_csv(trainfile, sep=',', index_col=0)
    test_df = pd.read_csv(testfile, sep=',', index_col=0)
    train_df.head()

    # 拆分特征和标签
    feature_names = train_df.columns
    label_columns = ['price']
    feature_names = np.setdiff1d(train_df.columns, label_columns)

    x_train_df = train_df[feature_names]
    y_train_df = train_df[label_columns]
    x_test_df = test_df[feature_names]

    # # 交叉验证划分
    # rskf = RepeatedStratifiedKFold(n_repeats=1, n_splits=5, random_state=666)
    # cv = [(t,v) for (t,v) in rskf.split(x_train_df, marks)]

    # 直接读取cv缓存文件-5折
    with open("./cache/cv.pkl", 'rb') as handle:
        cv = joblib.load(handle)

    # 训练
    mae_list = []
    y_pred_val_list = []
    model_list = []
    for fit_func in [fit_lgbm, fit_xgb, fit_ets]:
        mae, y_pred_val, models = train_predict_pipeline(x_train_df, y_train_df, cv, fit_func)
        mae_list.append(mae)
        y_pred_val_list.append(y_pred_val)
        model_list.append(models)
    error_mat = np.abs(np.transpose(y_pred_val_list) - y_train_df.values)
    # 验证集误差
    print(f'mae in validation set: {np.mean(mae_list):.4f}')

    # 预测测试集
    y_pred_test_list = []
    for models in model_list:
        y_pred_test = 0
        for model in models:
            y_pred_test += model.predict(x_test_df)
        y_pred_test /= len(models)
        y_pred_test_list.append(y_pred_test)
    y_pred_test_list = np.transpose(y_pred_test_list)

    # 计算测试集样本的K近邻训练样本, 并获得这些样本的索引
    from sklearn.neighbors import NearestNeighbors
    K = 10
    Neigh = NearestNeighbors(n_neighbors=K)
    Neigh.fit(x_train_df)
    idx_neighours = Neigh.kneighbors(x_test_df, n_neighbors=K, return_distance=False)

    # 为测试集样本计算权重
    from sklearn.preprocessing import normalize
    def get_mean_error(idx_kneighbors, error_mat):
        """计算样本K个近邻样本在验证集上的平均误差
        Parameters
        ----------
        idx_kneighbors: ndarray of shape (#neighbors, )
        error_mat: ndarray of shape (#samples, #estimators)

        Returns
        -------
        mean_error: ndarray of shape (#estimators, )
        """
        mean_errors = np.mean(error_mat[idx_kneighbors,:], axis=0)
        return mean_errors
    mean_error_mat = np.apply_along_axis(get_mean_error, axis=1, arr=idx_neighours, error_mat=error_mat)
    weights_mat = normalize(mean_error_mat, 'l1')

    # 加权预测
    weighted_y_pred_test = np.sum(y_pred_test_list*weights_mat, axis=1)

    # 保存结果
    columns_res = ['SaleID','price']
    pd.DataFrame(
        np.transpose([x_test_df.index, weighted_y_pred_test]), 
        columns=columns_res,
        ).to_csv('./result/used_car_submit_0410.csv')
    
    # 保存模型
    import joblib 
    with open ("./data/models_list.pkl", "wb") as handle:
        joblib.dump(model_list, handle)
    