# 实验一：新闻文本十分类

## 环境配置

```bash
# 创建并激活环境
conda create -n project1 python=3.10
conda activate project1

# 安装依赖
conda install numpy pandas scikit-learn matplotlib
# 或使用 pip
pip install -r requirements.txt
```

## 运行方式

### 方式一：直接运行完整实验（自动调参 + 生成预测）

```bash
python draft_main.py
```

### 方式二：命令行参数控制

```bash
# 只跑基线对比
python draft_main.py --mode baseline

# 跳过网格搜索（使用默认参数）
python draft_main.py --skip-tuning

# 指定验证集比例
python draft_main.py --val-ratio 0.2
```

## 输出文件

- `predictions.csv` — 测试集预测结果（提交用）
- `experiments_results.csv` — 各模型验证集准确率汇总（用于撰写报告）

## 实验流程

1. 数据加载与分层划分（训练集 80% / 验证集 20%）
2. TF-IDF 特征提取（仅在训练集上 fit）
3. 基线对比：NB / SVM / LR / MLP 默认参数
4. 网格搜索调参（使用验证集评估）
5. 最佳模型用全部训练数据重训
6. 生成 `predictions.csv`

## 随机种子

默认使用 `random_state=42` 保证结果可复现。
