# 最大熵打分库

加载最大熵模型，并进行打分。
使用方法
```cpp
maxent::LinearModel model;
model.load_model(model_path);
model.predict(feat_arr, feat_len, score_arr, label);
```