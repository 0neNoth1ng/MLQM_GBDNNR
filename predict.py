import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler

# 从原始文件导入模型类
from model.gbdnnr import DeepRegressor  # 确保导入路径正确

def preprocess_data(data, scaler, train_stats):
    """预处理数据（与训练时相同）"""
    # 列定义
    noise_columns = ['Error_T_SONIC', 'Error_CO2_density', 
                    'Error_CO2_density_fast_tmpr', 'Error_H2O_density',
                    'Error_H2O_sig_strgth', 'Error_CO2_sig_strgth']
    
    # 选择特征列
    X = data[noise_columns]
    
  
    
    # 特征缩放
    X_scaled = scaler.transform(X)
    
    return X_scaled

# 加载模型和相关对象
model_dir = 'gbdnnr_model'
model = DeepRegressor.load(model_dir)
scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
train_stats = joblib.load(os.path.join(model_dir, 'train_stats.pkl'))

# 加载测试数据
test_dataSet = pd.read_csv(r".\不含真实值\modified_数据集Time_Series662.dat")

# 预处理测试数据
X_test_scaled = preprocess_data(test_dataSet, scaler, train_stats)

# 使用模型进行预测
y_predict = model.decision_function(X_test_scaled)

# 保存预测结果
results = []
for Predicted_Value in y_predict:
    formatted_predicted_value = ' '.join(map(str, Predicted_Value))
    results.append([formatted_predicted_value])

result_df = pd.DataFrame(results, columns=['Predicted_Value'])
result_df.to_csv("predictions.csv", index=False)

print("预测完成，结果已保存至 predictions.csv")