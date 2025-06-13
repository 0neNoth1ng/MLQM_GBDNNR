import time, os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from model.gbdnnr import DeepRegressor

start_time = time.time()

# 特征缩放
scaler = StandardScaler()

# 加载数据集
train_dataSet = pd.read_csv(".\真实值\modified_数据集Time_Series661.dat")
test_dataSet = pd.read_csv(".\不含真实值\modified_数据集Time_Series662.dat")
print("训练集列名:", train_dataSet.columns.tolist())
print("测试集列名:", test_dataSet.columns.tolist())
# 列定义
columns = ['T_SONIC', 'CO2_density', 'CO2_density_fast_tmpr', 'H2O_density', 'H2O_sig_strgth', 'CO2_sig_strgth']
noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                 'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
CL = columns + noise_columns

# 查看数据缺失情况
data = train_dataSet[CL]
missingDf = data.isnull().sum().sort_values(ascending=False).reset_index()
missingDf.columns = ['feature', 'miss_num']
missingDf['miss_percentage'] = missingDf['miss_num'] / data.shape[0]
print("缺失值比例")
print(missingDf)

# 填补缺失值
train_dataSet[CL] = train_dataSet[CL].fillna(train_dataSet[CL].median())

# 异常值检测与处理
for column in CL:
    z_scores = np.abs(stats.zscore(train_dataSet[column].dropna()))
    outliers = (z_scores > 2)
    train_dataSet[column] = train_dataSet[column].mask(outliers, train_dataSet[column].median())

# 数据采样（加速训练）
train_dataSet_sampled = train_dataSet.sample(frac=0.0001, random_state=217)  # 采样50%数据

# 划分数据集
X_train_full = train_dataSet[noise_columns]
y_train_full = train_dataSet[columns]
X_test = test_dataSet[noise_columns]

# 创建验证集
X_train = X_train_full
y_train = y_train_full

# 特征缩放
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 查看训练集和测试集列名及数据类型
print("训练集结构:")
print(train_dataSet.info())
print("\n测试集结构:")
print(test_dataSet.info())

# 输出示例：
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 10000 entries, 0 to 9999
# Columns: 12 entries, T_SONIC to Error_CO2_sig_strgth
# dtypes: float64(12)
# memory usage: 937.6 KB

# 描述性统计（数值列）
print("训练集统计摘要:")
print(train_dataSet[CL].describe().T)
print("\n测试集统计摘要:")
print(test_dataSet[noise_columns].describe().T)


model = DeepRegressor(random_state=123)
model.fit(X_train_scaled, y_train)



# 使用最佳模型预测
y_predict = model.predict(X_test_scaled)
import joblib
# 保存预测结果
results = []
for Predicted_Value in y_predict:
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    results.append([formatted_predicted_value])

result_df = pd.DataFrame(results, columns=['Predicted_Value'])
result_df.to_csv("result_LightGBM.csv", index=False)

print("预测完成，结果已保存至 result_LightGBM.csv")

end_time = time.time()
print(f"总耗时：{end_time - start_time:.3f}秒")

# 保存训练集的统计信息
train_stats = {
    'medians': train_dataSet[noise_columns].median(),
    'means': train_dataSet[noise_columns].mean(),
    'stds': train_dataSet[noise_columns].std()
}

import shutil

# 在训练结束后，保存模型
model_dir = 'gbdnnr_model'

# 如果目录已存在，先删除
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

# 保存模型
model.save(model_dir)

# 保存预处理对象和统计信息
joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(train_stats, os.path.join(model_dir, 'train_stats.pkl'))