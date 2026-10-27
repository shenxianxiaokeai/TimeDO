# 输出目录

运行产物集中保存在这里，并由 Git 忽略：

- `runs/<dataset>/pred_<horizon>/<mode>/`：训练/评估结果、指标、运行配置和训练得到的 checkpoint。
- `tuning/<dataset>/pred_<horizon>/`：每组调参任务的日志、候选 checkpoint、汇总表和验证集最优记录。

此目录只存放可再生成的运行产物，不存源码、输入 CSV 或发布用预训练权重。
