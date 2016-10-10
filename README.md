# 最大熵程序库

## 模型训练

## 模型评估
加载最大熵模型，并进行打分。
使用方法
```cpp
maxent::LinearModel model;
model.load_model(model_path);
model.predict(feat_arr, feat_len, score_arr, label);
```
