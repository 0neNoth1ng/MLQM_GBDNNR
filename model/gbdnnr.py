"""GB-DNNR - Gradient Boosted - Deep Neural Network Regression"""

# Author: Seyedsaman Emami
# Author: Gonzalo Martínez-Muñoz

# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)


import numpy as np
from Base._base import BaseEstimator
from Base._loss import squared_loss
import os
import joblib
import tensorflow as tf

class DeepRegressor(BaseEstimator):
    def __init__(
        self,
        iter=200,
        eta=1,
        learning_rate=1e-2,
        total_nn=300,
        num_nn_step=100,
        batch_size=128,
        early_stopping=10,
        random_state=None,
        l2=0.1,
        dropout=0.1,
        record=False,
        freezing=True,
    ):
        super().__init__(
            iter,
            eta,
            learning_rate,
            total_nn,
            num_nn_step,
            batch_size,
            early_stopping,
            random_state,
            l2,
            dropout,
            record,
            freezing,
        )

    def _validate_y(self, y):
        self._loss = squared_loss()
        self.n_classes = y.shape[1] if len(y.shape) == 2 else 1
        return y

    def predict(self, X):
        return self.decision_function(X)

    def predict_stage(self, X):
        preds = np.ones_like(self._models[0].predict(X)) * self.intercept

        for model, step in zip(self._models, self.steps):
            preds += model.predict(X) * step
            yield preds

    def score(self, X, y):
        """Returns the average of RMSE of all outputs."""
        pred = self.predict(X)
        output_errors = np.mean((y - pred) ** 2, axis=0)

        return np.mean(np.sqrt(output_errors))

    def save(self, path):
        """保存模型到指定路径"""
        # 确保路径存在
        os.makedirs(path, exist_ok=True)
        
        # 保存模型参数
        params = self.get_params()
        joblib.dump(params, os.path.join(path, 'params.pkl'))
        
        # 保存模型权重 - 确保每个模型有自己的子目录
        for i, model in enumerate(self._models):
            # 为每个模型创建单独的子目录
            model_dir = os.path.join(path, f'model_{i}')
            os.makedirs(model_dir, exist_ok=True)
            
            # 保存整个模型（包括结构）
            model.save(model_dir)
        
        # 保存其他重要属性
        state = {
            'intercept': self.intercept,
            'steps': self.steps,
            'g_history': self.g_history,
            'n_classes': self.n_classes
        }
        joblib.dump(state, os.path.join(path, 'state.pkl'))
    
    @classmethod
    def load(cls, path):
        """从指定路径加载模型"""
        # 加载参数
        params = joblib.load(os.path.join(path, 'params.pkl'))
        
        # 创建模型实例
        model = cls(**params)
        
        # 加载状态
        state = joblib.load(os.path.join(path, 'state.pkl'))
        model.intercept = state['intercept']
        model.steps = state['steps']
        model.g_history = state['g_history']
        model.n_classes = state['n_classes']
        
        # 初始化模型列表和层列表
        model._models = []
        model.layers = []  # 重新构建层列表
        
        # 加载每个子模型
        i = 0
        while os.path.exists(os.path.join(path, f'model_{i}')):
            # 加载整个模型
            model_path = os.path.join(path, f'model_{i}')
            loaded_model = tf.keras.models.load_model(model_path)
            
            # 添加到模型列表
            model._models.append(loaded_model)
            
            # 从加载的模型中提取层信息
            if i > 0:  # 第一个模型没有冻结层
                layer = loaded_model.layers[-2]
                model.layers.append(layer)
            
            i += 1
        
        return model