| 模型 | 阶段 | 准确率 | 耗时(s) | TF-IDF特征数 | ngram | 模型参数 |
|------|------|--------|---------|--------------|-------|----------|
| NaiveBayes | baseline | 0.9003 | 0.01 | 10000 | (1, 1) | {'alpha': 1.0} |
| SVM | baseline | 0.9240 | 24.46 | 10000 | (1, 1) | {'C': 1.0, 'kernel': 'linear'} |
| LogisticRegression | baseline | 0.9145 | 0.62 | 10000 | (1, 1) | {'C': 1.0, 'max_iter': 1000} |
| MLP | baseline | 0.9328 | 83.39 | 10000 | (1, 1) | {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 300} |
| NaiveBayes | tuned | 0.9057 | 2.22 | 10000 | (1, 1) | {'classifier__alpha': 0.1} |
| SVM | tuned | 0.9240 | 62.64 | 10000 | (1, 1) | {'classifier__C': 1, 'classifier__kernel': 'linear'} |
| LogisticRegression | tuned | 0.9274 | 2.83 | 10000 | (1, 1) | {'classifier__C': 10, 'classifier__max_iter': 1000} |
| MLP | tuned | 0.9315 | 381.06 | 10000 | (1, 1) | {'classifier__alpha': 0.0001, 'classifier__hidden_layer_sizes': (50,)} |