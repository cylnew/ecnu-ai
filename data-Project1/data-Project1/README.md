## 项目概述

本项目基于 TF-IDF 特征提取，对新闻文本进行 10 分类任务。实验包含 4 个经典机器学习模型（NaiveBayes、SVM、LogisticRegression、MLP），并在基线、调参、数据预处理、N-gram 扩展、模型融合等维度展开系统对比。

---

## 环境配置

```bash
# 1. 创建 conda 环境
conda create -n project1 python=3.10
conda activate project1

# 2. 安装依赖
conda install numpy pandas scikit-learn matplotlib nltk
# 或使用 pip
pip install numpy pandas scikit-learn matplotlib nltk
```

**Python 版本**：≥ 3.7（推荐 3.10）

**核心依赖**：
- `numpy`, `pandas` — 数据处理
- `scikit-learn` — 机器学习模型与评估
- `matplotlib` — 可视化
- `nltk` — 英文停用词 & Porter 词干化

---

## 运行方法

```bash
cd data-Project1/data-Project1
python draft_main.py
```

程序会自动：
1. 加载 `train_data.csv`（训练集）和 `test_data_unlabeled.csv`（测试集）
2. 按 `VAL_RATIO=0.2` 划分训练集/验证集（`random_state=42`）
3. 执行 4 个模型的基线训练、调参对比、优化实验
4. 在当前目录生成所有结果文件

**程序运行约需 3~5 分钟**（MLP 训练较慢）

---

## 输出文件说明

| 文件 | 说明 |
|------|------|
| `predictions.csv` | **最终提交文件**——测试集预测标签（无表头，共 2457 行，值 0-9） |
| `experiments_results.csv` | 所有实验结果的 CSV 表格 |
| `experiments_results.md` | 实验结果 Markdown 表格 |
| `experiments_results_table.txt` | 实验结果文本表格 |
| `fig1_model_accuracy_compare.png` | 基线/调参后各模型准确率对比柱状图 |
| `fig2_training_time_compare.png` | 各模型训练耗时对比柱状图 |
| `fig3_tuning_improvement.png` | 调参前后准确率变化对比图 |
| `fig4_overall_comparison.png` | 预处理/ngram/融合实验综合对比图 |
| `outputs/figs/fig_mlp_loss_curve.png` | MLP 训练过程 loss 曲线（含验证 acc） |
| `outputs/figs/fig_before_after_comparison.png` | 优化前后效果对比图 |
| `outputs/tables/optimization_results_table.txt` | 优化实验结果表 |

---

## 模型与参数配置

### 全局配置
- `RANDOM_STATE = 42`（所有模型固定，保证可复现）
- 验证集比例：`VAL_RATIO = 0.2`
- TF-IDF：`max_features=10000, ngram_range=(1,1), min_df=2, sublinear_tf=True`

### 各模型调参网格

| 模型 | 调参项 | 候选值 |
|------|--------|--------|
| NaiveBayes | alpha | 0.01, 0.1, 1.0, 10.0 |
| SVM | C, kernel | C: {0.1, 1, 10}, kernel: linear |
| LogisticRegression | C | 0.1, 1, 10 |
| MLP | hidden_layer_sizes, alpha | (50,), (100,), (100,50); alpha: 1e-4, 1e-3 |

### 优化实验
- **4a 预处理优化**：自定义 clean_tokenizer（含去 HTML/URL、Porter Stemming、去停用词）
- **4b N-gram 扩展**：尝试 (1,1) unigram / (1,2) bigram / (1,3) trigram
- **4c 模型融合**：LR + SVM + NB + MLP 软投票（soft voting）

---

## 实验结果摘要

### Part 1：基线 vs 调参（验证集准确率）

| 模型 | Baseline Acc | Tuned Acc | Δ |
|------|-------------|-----------|---|
| NaiveBayes | 0.9003 | 0.9057 | +0.0054 |
| SVM | 0.9240 | 0.9240 | +0.0000 |
| LogisticRegression | 0.9145 | 0.9274 | **+0.0129** |
| MLP | **0.9328** | 0.9315 | -0.0013 |

### Part 2：优化实验（Macro-F1）

| 优化方向 | 配置 | Macro-F1 |
|---------|------|---------|
| 预处理优化 | 自定义清洗（去HTML/URL/Stemming） | 0.9192 |
| N-gram扩展 | bigram (1,2) | 0.9145（↓） |
| 投票融合 | LR+SVM+NB+MLP 软投票 | **0.9338** |

### 最终提交模型
- **使用模型**：LR + SVM + NB + MLP 软投票融合
- **验证集 Macro-F1**：0.9338
- **predictions.csv 对应配置**：见下方"关于 predictions.csv"

---

## 关于 predictions.csv

- **生成方式**：`save_results()` 函数自动将最终模型在**完整训练集**（训练+验证）上训练后，对测试集预测结果写入 `predictions.csv`
- **文件格式**：无表头，单列整数（0-9），共 2457 行
- **提交说明**：本文件为助教评测脚本的输入文件，请确保直接提交此文件，无需修改格式
- **如需备份**：
  ```bash
  copy predictions.csv predictions_backup_0.9338.csv
  ```

---

## 随机种子说明

本实验所有涉及随机性的模型（SVM、LR、MLP、训练集划分）均使用固定随机种子：

```python
RANDOM_STATE = 42  # draft_main.py 第 103 行
```

请勿随意修改此值，否则结果将与实验报告不一致。

---

## 注意事项

1. **NLTK 首次运行会下载停用词库**，若网络不畅可设置 `_STOP_WORDS = set()` 跳过
2. MLP 训练较慢（约 1 分钟/次），耐心等待即可
3. 程序会自动创建 `outputs/` 目录存放图表和表格，无需手动新建
4. 如遇中文字体问题，matplotlib 会自动降级使用英文标签
