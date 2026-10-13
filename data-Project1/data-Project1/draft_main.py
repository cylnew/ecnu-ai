"""
实验一：新闻文本十分类
- TF-IDF + 朴素贝叶斯 / SVM / 逻辑回归 / MLP
- 基线对比 + 网格搜索调参
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境也能保存图
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, confusion_matrix)
from sklearn.pipeline import Pipeline

# NLTK 可选（未装则降级）
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    try:
        _STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        nltk.download('stopwords', quiet=True)
        _STOP_WORDS = set(stopwords.words('english'))
    _STEMMER = PorterStemmer()
except Exception:
    _STOP_WORDS = set()
    _STEMMER = None

# 输出目录
OUTPUT_DIR = 'outputs'
FIG_DIR    = os.path.join(OUTPUT_DIR, 'figs')
TABLE_DIR  = os.path.join(OUTPUT_DIR, 'tables')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ============================================================
# 文本预处理工具（数据层优化）
# ============================================================
_HTML_RE    = re.compile(r'<[^>]+>')
_URL_RE     = re.compile(r'https?://\S+|www\.\S+')
_CDATA_RE   = re.compile(r'<!\[CDATA\[|\]\]>')
_NON_ALPHA  = re.compile(r'[^A-Za-z\u4e00-\u9fa5]')


def clean_tokenizer(text):
    """去除 HTML/URL/CDATA → 小写 → 中英文字符 → Porter Stemming → 去停用词"""
    if not isinstance(text, str):
        return []
    t = _CDATA_RE.sub(' ', text)
    t = _HTML_RE.sub(' ', t)
    t = _URL_RE.sub(' ', t)
    t = _NON_ALPHA.sub(' ', t.lower())
    out = []
    for w in t.split():
        if len(w) <= 1:
            continue
        if _STEMMER is not None:
            w = _STEMMER.stem(w)
        if w not in _STOP_WORDS:
            out.append(w)
    return out


def default_tokenizer(text):
    """原始 baseline 的简易 tokenizer"""
    if not isinstance(text, str):
        return []
    return _NON_ALPHA.sub(' ', text.lower()).split()


def make_vectorizer(mode='clean', **kwargs):
    """
    构造 TfidfVectorizer
    mode='clean'   → 使用自定义 clean_tokenizer（含 stemming / 去停用词）
    mode='default' → 使用原始 default_tokenizer（仅小写+非字母切分）
    """
    base = dict(max_features=10000, ngram_range=(1, 1), min_df=2, sublinear_tf=True)
    base.update(kwargs)
    base['tokenizer'] = clean_tokenizer if mode == 'clean' else default_tokenizer
    return TfidfVectorizer(**base)


# ============================================================
# 配置区
# ============================================================
RANDOM_STATE = 42
VAL_RATIO = 0.2
N_JOBS = -1

# TF-IDF 候选参数
TFIDF_PARAMS = {
    'max_features': [5000, 10000],
    'ngram_range': [(1, 1), (1, 2)],
    'min_df': [2, 5]
}

# 模型超参数候选网格
PARAM_GRIDS = {
    'NaiveBayes': {
        'classifier__alpha': [0.01, 0.1, 1.0, 10.0]
    },
    'SVM': {
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear']
    },
    'LogisticRegression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__max_iter': [1000]
    },
    'MLP': {
        'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
        'classifier__alpha': [0.0001, 0.001]
    }
}

# 默认参数（用于基线对比）
DEFAULT_PARAMS = {
    'NaiveBayes': {'alpha': 1.0},
    'SVM': {'C': 1.0, 'kernel': 'linear'},
    'LogisticRegression': {'C': 1.0, 'max_iter': 1000},
    'MLP': {'hidden_layer_sizes': (100,), 'alpha': 0.0001, 'max_iter': 300}
}

# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载训练集和测试集"""
    train_df = pd.read_csv('train_data.csv')
    test_df = pd.read_csv('test_data_unlabeled.csv')
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()
    print(f"[数据] 训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    print(f"[数据] 类别数量: {len(np.unique(y_train))}")
    return X_train, y_train, X_test


def split_data(X, y, val_ratio=VAL_RATIO):
    """划分训练集和验证集（分层划分）"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_ratio,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"[划分] 训练集: {len(X_train)}, 验证集: {len(X_val)}")
    return X_train, X_val, y_train, y_val


# ============================================================
# 模型工厂
# ============================================================
def get_classifier(name, params=None):
    """根据名称返回分类器实例，每个分类器只接受自己支持的参数"""
    if params is None:
        params = {}

    if name == 'NaiveBayes':
        return MultinomialNB(**params)
    elif name == 'SVM':
        return SVC(random_state=RANDOM_STATE, **params)
    elif name == 'LogisticRegression':
        return LogisticRegression(random_state=RANDOM_STATE, **params)
    elif name == 'MLP':
        return MLPClassifier(random_state=RANDOM_STATE, **params)
    else:
        raise ValueError(f"未知模型: {name}")


# ============================================================
# 特征提取
# ============================================================
def create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 1), min_df=2):
    """创建 TF-IDF 向量化器"""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=True
    )


# ============================================================
# 基线对比
# ============================================================
def run_baseline(X_train, X_val, y_train, y_val, tfidf_params):
    """使用默认参数运行所有模型，对比基线性能"""
    print("\n" + "=" * 60)
    print("阶段一：基线对比（默认参数）")
    print("=" * 60)

    # TF-IDF 向量化（仅在训练集上 fit）
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    print(f"[TF-IDF] 特征维度: {X_train_tfidf.shape[1]}")

    results = {}

    for model_name in ['NaiveBayes', 'SVM', 'LogisticRegression', 'MLP']:
        print(f"\n--- 训练 {model_name} ---")
        start_time = time.time()

        clf = get_classifier(model_name, DEFAULT_PARAMS[model_name])
        clf.fit(X_train_tfidf, y_train)

        y_pred = clf.predict(X_val_tfidf)
        acc = accuracy_score(y_val, y_pred)
        elapsed = time.time() - start_time

        macro_f1 = f1_score(y_val, y_pred, average='macro')
        weighted_f1 = f1_score(y_val, y_pred, average='weighted')
        results[model_name] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'time': elapsed,
            'tfidf_params': tfidf_params,
            'model_params': DEFAULT_PARAMS[model_name]
        }

        print(f"[{model_name}] acc={acc:.4f} | macro-F1={macro_f1:.4f} | "
              f"weighted-F1={weighted_f1:.4f} | {elapsed:.2f}s")

    return results, vectorizer


# ============================================================
# 网格搜索调参
# ============================================================
def tune_models(X_train, X_val, y_train, y_val, tfidf_params):
    """对每个模型进行网格搜索调参"""
    print("\n" + "=" * 60)
    print("阶段二：网格搜索调参")
    print("=" * 60)

    # TF-IDF 向量化（仅在训练集上 fit）
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    print(f"[TF-IDF] 特征维度: {X_train_tfidf.shape[1]}")

    tuned_results = {}

    for model_name in ['NaiveBayes', 'SVM', 'LogisticRegression', 'MLP']:
        print(f"\n--- 调参 {model_name} ---")
        start_time = time.time()

        clf = get_classifier(model_name)

        # 创建 Pipeline
        pipeline = Pipeline([
            ('classifier', clf)
        ])

        param_grid = PARAM_GRIDS[model_name]

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=N_JOBS,
            verbose=0
        )

        grid_search.fit(X_train_tfidf, y_train)

        # 在验证集上评估（参考）
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_val_tfidf)
        val_acc = accuracy_score(y_val, y_pred)
        elapsed = time.time() - start_time

        val_macro_f1 = f1_score(y_val, y_pred, average='macro')
        val_weighted_f1 = f1_score(y_val, y_pred, average='weighted')
        tuned_results[model_name] = {
            'best_params': grid_search.best_params_,
            'cv_best_score': grid_search.best_score_,        # CV f1_macro（报出）
            'val_accuracy': val_acc,
            'val_macro_f1': val_macro_f1,
            'val_weighted_f1': val_weighted_f1,
            'time': elapsed,
            'best_model': best_model
        }

        print(f"[{model_name}] 最佳参数: {grid_search.best_params_}")
        print(f"[{model_name}] CV最佳分数: {grid_search.best_score_:.4f}")
        print(f"[{model_name}] acc={val_acc:.4f} | macro-F1={val_macro_f1:.4f} | "
              f"weighted-F1={val_weighted_f1:.4f} | {elapsed:.2f}s")

    return tuned_results, vectorizer


# ============================================================
# 选择最佳模型并重训
# ============================================================
def retrain_best_on_full_data(tuned_results, X_train, X_val, y_train, y_val,
                              X_test, tfidf_params):
    """
    从两个候选方案里挑验证集上 val_accuracy 最高的那个，
    再用 X_train + X_val 全量数据重训，对 X_test 给出预测。

    候选方案：
        ① 网格搜索阶段表现最好的单模型
        ② 投票融合 (LR + SVM + NB + MLP，MLP 参数取自网格搜索)
    """
    print("\n" + "=" * 60)
    print("阶段三：选最佳方案 (单模型 vs 投票融合) 并全量重训")
    print("=" * 60)

    # ---------- ① 单模型（沿用原逻辑） ----------
    best_model_name = max(tuned_results, key=lambda k: tuned_results[k]['val_accuracy'])
    single_val_acc  = tuned_results[best_model_name]['val_accuracy']
    single_params   = tuned_results[best_model_name]['best_params']
    print(f"\n[候选1] 单模型最佳: {best_model_name}, val_acc={single_val_acc:.4f}")

    # ---------- ② 投票融合 ----------
    # 在 X_train 上拟合向量化器，X_val 上评估，得到可比对的 val_acc
    vec_eval = create_tfidf_vectorizer(**tfidf_params)
    Xtr_eval = vec_eval.fit_transform(X_train)
    Xva_eval = vec_eval.transform(X_val)

    # 取出 MLP 网格搜索最优参数
    mlp_best_params = {k.replace('classifier__', ''): v
                       for k, v in tuned_results['MLP']['best_params'].items()}
    mlp_best_params.setdefault('max_iter', 300)
    mlp_best_params.setdefault('solver',   'adam')

    voting_eval = VotingClassifier(
        estimators=[
            ('lr',  LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
            ('svm', SVC(C=1.0, kernel='linear', probability=True, random_state=RANDOM_STATE)),
            ('nb',  MultinomialNB(alpha=1.0)),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=mlp_best_params.get('hidden_layer_sizes', (100,)),
                alpha=mlp_best_params.get('alpha', 0.0001),
                max_iter=mlp_best_params.get('max_iter', 300),
                random_state=RANDOM_STATE,
                solver=mlp_best_params.get('solver', 'adam'),
            )),
        ],
        voting='soft', n_jobs=1   # Windows: 必填 1，否则 joblib + sparse.sort_indices 报 WRITEBACKIFCOPY
    )
    voting_eval.fit(Xtr_eval, y_train)
    voting_val_acc = accuracy_score(y_val, voting_eval.predict(Xva_eval))
    print(f"[候选2] 投票融合 LR+SVM+NB+MLP, val_acc={voting_val_acc:.4f}")

    # ---------- 选赢家 ----------
    use_voting = voting_val_acc >= single_val_acc   # 平票时选投票
    print(f"\n>>> 选用: {'投票融合' if use_voting else f'单模型 {best_model_name}'} "
          f"(val_acc={max(voting_val_acc, single_val_acc):.4f})")

    # ---------- 全量数据重训 ----------
    X_all = X_train + X_val
    y_all = np.concatenate([y_train, y_val])

    vec_full = create_tfidf_vectorizer(**tfidf_params)
    X_all_tfidf  = vec_full.fit_transform(X_all)
    X_test_tfidf = vec_full.transform(X_test)
    print(f"[TF-IDF] 特征维度: {X_all_tfidf.shape[1]}")

    if use_voting:
        chosen_name   = 'VotingEnsemble_LR+SVM+NB+MLP'
        chosen_acc    = voting_val_acc
        chosen_params = {'voting': 'soft', 'tfidf_params': tfidf_params,
                         'mlp_params': mlp_best_params}
        final_model = VotingClassifier(
            estimators=[
                ('lr',  LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
                ('svm', SVC(C=1.0, kernel='linear', probability=True, random_state=RANDOM_STATE)),
                ('nb',  MultinomialNB(alpha=1.0)),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=mlp_best_params.get('hidden_layer_sizes', (100,)),
                    alpha=mlp_best_params.get('alpha', 0.0001),
                    max_iter=mlp_best_params.get('max_iter', 300),
                    random_state=RANDOM_STATE,
                    solver=mlp_best_params.get('solver', 'adam'),
                )),
            ],
            voting='soft', n_jobs=1  # Windows: 必填 1
        )
    else:
        chosen_name   = best_model_name
        chosen_acc    = single_val_acc
        chosen_params = single_params
        classifier_params = {k.replace('classifier__', ''): v
                            for k, v in single_params.items()}
        final_model = get_classifier(best_model_name, classifier_params)

    print(f"\n正在用全部训练数据 ({len(X_all)} 样本) 重训 {chosen_name}...")
    final_model.fit(X_all_tfidf, y_all)
    predictions = final_model.predict(X_test_tfidf)
    print(f"预测完成！测试集样本数: {len(predictions)}")

    return predictions, chosen_name, chosen_acc, chosen_params


# ============================================================
# ============================================================
# 阶段四：优化实验（自动执行）
# ============================================================
def run_optimization_experiments(X_train, X_val, y_train, y_val,
                                 baseline_results, tuned_results, tfidf_params):
    """
    自动跑三组优化实验，结果存入 tables/optimization_results.csv
    优化前后对比图存入 figs/fig_before_after_comparison.png
    """
    import matplotlib.pyplot as _plt
    import matplotlib
    matplotlib.use('Agg')
    _plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                         'PingFang SC', 'Arial Unicode MS', 'DejaVu Sans']
    _plt.rcParams['axes.unicode_minus'] = False

    print("\n" + "=" * 60)
    print("阶段四：优化实验")
    print("=" * 60)

    records = []   # 汇总表
    bar_data = {}  # 画图用

    # ----- 优化实验 1：自定义预处理 vs 原始预处理 -----
    print("\n[优化1] 自定义预处理 vs 原始预处理 (LR)")
    vec_def = make_vectorizer(mode='default')
    vec_cln = make_vectorizer(mode='clean')

    for mode, vec in [('原始预处理', vec_def), ('自定义预处理', vec_cln)]:
        Xtr = vec.fit_transform(X_train)
        Xva = vec.transform(X_val)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
        yp = clf.predict(Xva)
        acc = accuracy_score(y_val, yp)
        mf1 = f1_score(y_val, yp, average='macro')
        wf1 = f1_score(y_val, yp, average='weighted')
        records.append({'优化方法': '预处理优化', '变体': mode,
                         'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1})
        bar_data.setdefault('预处理优化', {})[mode] = mf1
        print(f"  [{mode}] acc={acc:.4f} | macro-F1={mf1:.4f}")

    # ----- 优化实验 2：n-gram 扩展 -----
    print("\n[优化2] n-gram 扩展 (LR, clean tokenizer)")
    for ngram, label in [((1, 1), 'unigram'), ((1, 2), 'bigram'), ((1, 3), 'trigram')]:
        vec = make_vectorizer(mode='clean', ngram_range=ngram)
        Xtr = vec.fit_transform(X_train)
        Xva = vec.transform(X_val)
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(Xtr, y_train)
        yp = clf.predict(Xva)
        acc = accuracy_score(y_val, yp)
        mf1 = f1_score(y_val, yp, average='macro')
        wf1 = f1_score(y_val, yp, average='weighted')
        records.append({'优化方法': 'ngram扩展', '变体': label,
                         'accuracy': acc, 'macro_f1': mf1, 'weighted_f1': wf1})
        bar_data.setdefault('ngram扩展', {})[label] = mf1
        print(f"  [{label}] acc={acc:.4f} | macro-F1={mf1:.4f}")

    # ----- 优化实验 3：投票融合 (LR + SVM + NB + MLP) -----
    # 注意：使用与阶段二网格搜索一致的 TF-IDF 参数，保证特征空间一致，
    # 网格搜索找到的最佳参数才能迁移到这里使用。
    print("\n[优化3] 投票融合 LR+SVM+NB+MLP (与网格搜索一致的 TF-IDF)")
    vec = create_tfidf_vectorizer(**tfidf_params)
    Xtr = vec.fit_transform(X_train)
    Xva = vec.transform(X_val)

    # 基线单模型 (与网格搜索一致的特征)
    best_single = None
    best_single_f1 = 0
    for name, clf_cls, params in [
        ('LR',  LogisticRegression, {'C': 1.0, 'max_iter': 1000}),
        ('SVM', SVC,               {'C': 1.0, 'kernel': 'linear'}),
        ('NB',  MultinomialNB,     {'alpha': 1.0}),
    ]:
        # NB 不支持 random_state，按模型单独处理
        if clf_cls is MultinomialNB:
            clf = clf_cls(**params)
        else:
            clf = clf_cls(**params, random_state=42)
        clf.fit(Xtr, y_train)
        yp = clf.predict(Xva)
        f1 = f1_score(y_val, yp, average='macro')
        print(f"  [{name} 单模型] macro-F1={f1:.4f}")
        if f1 > best_single_f1:
            best_single_f1 = f1
            best_single = name

    # 从网格搜索结果里取出 MLP 的最佳参数（去掉前缀 classifier__）
    mlp_best_params = {k.replace('classifier__', ''): v
                       for k, v in tuned_results['MLP']['best_params'].items()}
    # 补齐必备参数
    mlp_best_params.setdefault('max_iter', 300)
    mlp_best_params.setdefault('random_state', RANDOM_STATE)
    mlp_best_params.setdefault('solver', 'adam')

    # 软投票融合 (LR + SVM + NB + MLP)
    lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    svm = SVC(C=1.0, kernel='linear', probability=True, random_state=RANDOM_STATE)
    nb  = MultinomialNB(alpha=1.0)
    mlp = MLPClassifier(
        hidden_layer_sizes=mlp_best_params.get('hidden_layer_sizes', (100,)),
        alpha=mlp_best_params.get('alpha', 0.0001),
        max_iter=mlp_best_params.get('max_iter', 300),
        random_state=RANDOM_STATE,
        solver=mlp_best_params.get('solver', 'adam'),
    )
    print(f"  [MLP 参数] hidden={mlp_best_params.get('hidden_layer_sizes')} "
          f"alpha={mlp_best_params.get('alpha')}")

    voting = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('nb', nb), ('mlp', mlp)],
        voting='soft', n_jobs=1   # Windows: 必填 1，否则 sparse 报 WRITEBACKIFCOPY
    )
    voting.fit(Xtr, y_train)
    yp_voting = voting.predict(Xva)
    acc_v = accuracy_score(y_val, yp_voting)
    mf1_v = f1_score(y_val, yp_voting, average='macro')
    wf1_v = f1_score(y_val, yp_voting, average='weighted')
    records.append({'优化方法': '投票融合', '变体': 'LR+SVM+NB+MLP软投票',
                     'accuracy': acc_v, 'macro_f1': mf1_v, 'weighted_f1': wf1_v})
    bar_data.setdefault('投票融合', {})['单模型最佳'] = best_single_f1
    bar_data['投票融合']['软投票融合'] = mf1_v
    print(f"  [投票融合] acc={acc_v:.4f} | macro-F1={mf1_v:.4f}")

    # ----- 保存 CSV -----
    df_opt = pd.DataFrame(records)
    csv_path = os.path.join(TABLE_DIR, 'optimization_results.csv')
    df_opt.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[保存] {csv_path}")

    # 同样输出对齐的 TXT 表格（与 experiments_results_table.txt 风格一致）
    txt_path = os.path.join(TABLE_DIR, 'optimization_results_table.txt')
    md_path  = os.path.join(TABLE_DIR, 'optimization_results.md')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(_format_opt_aligned_table(df_opt))
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(_format_opt_markdown_table(df_opt))
    print(f"[保存] {txt_path}")
    print(f"[保存] {md_path}")

    # 控制台同时打印一份
    print("\n" + _format_opt_aligned_table(df_opt))

    # ----- 画优化前后对比图 -----
    _plot_before_after_comparison(bar_data, baseline_results, tuned_results)

    return records


def _format_opt_aligned_table(df):
    """生成 | 分隔、宽度对齐的等宽文本表格（与 experiments_results_table.txt 风格一致）"""
    headers = ['优化方法', '变体', '准确率', 'Macro-F1', 'Weighted-F1']
    keys    = ['优化方法', '变体', 'accuracy', 'macro_f1', 'weighted_f1']

    def fmt(v, kind):
        if kind in ('accuracy', 'macro_f1', 'weighted_f1'):
            return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
        return str(v)

    rows = [[fmt(row[k], k) for k in keys] for _, row in df.iterrows()]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def line(c='-'):
        return '+' + '+'.join(c * (w + 2) for w in widths) + '+'

    def row_line(cells):
        return '| ' + ' | '.join(cells[i].ljust(widths[i]) for i in range(len(headers))) + ' |'

    out = [line('-'), row_line(headers), line('=')]
    for r in rows:
        out.append(row_line(r))
        out.append(line('-'))
    return '\n'.join(out)


def _format_opt_markdown_table(df):
    """生成 Markdown 表格"""
    headers = ['优化方法', '变体', 'accuracy', 'macro_f1', 'weighted_f1']
    disp    = ['优化方法', '变体', '准确率', 'Macro-F1', 'Weighted-F1']
    out = ['| ' + ' | '.join(disp) + ' |',
           '|' + '|'.join('---' for _ in headers) + '|']
    for _, row in df.iterrows():
        out.append('| ' + ' | '.join(str(row[h]) for h in headers) + ' |')
    return '\n'.join(out)


def _plot_before_after_comparison(bar_data, baseline_results, tuned_results):
    """把优化前后的指标画到一张并排对比图"""
    import matplotlib.pyplot as _plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = _plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Optimization: Before vs After Comparison', fontsize=14, fontweight='bold')

    # === 优化汇总：从阶段四各优化方案中取每个模型的最高 macro_f1 ===
    # 这里用 ngram（unigram/bigram/trigram）中最高的当 "Before"
    # 用软投票融合的 macro_f1 当 "After"
    ngram_best = max(bar_data['ngram扩展'].values()) if 'ngram扩展' in bar_data else 0
    voting_best = bar_data['投票融合']['软投票融合'] if '投票融合' in bar_data else 0
    # 为每个模型构造 "优化后" 指标：取 tuned_results 中最高的 + 投票融合（取较大者）
    best_tuned = max((tuned_results[m]['val_macro_f1']
                      for m in tuned_results), default=0)

    # 子图 0 数据准备
    model_order = ['NaiveBayes', 'SVM', 'LogisticRegression', 'MLP']
    model_labels = ['Naive Bayes', 'SVM', 'Logistic Reg', 'MLP']

    # 子图 0：调参前后对比（在 val 上评估）
    # 调参前：默认参数模型在 val 上的 macro-F1
    # 调参后：GridSearchCV 选出的最佳参数模型在 val 上的 macro-F1
    ax0 = axes[0]
    x = range(len(model_order))
    before = [baseline_results[m]['macro_f1'] for m in model_order]
    after  = [tuned_results[m]['val_macro_f1'] for m in model_order]
    width = 0.35
    ax0.bar([i - width/2 for i in x], before, width, label='Before: baseline',
            color='#9ecae1', edgecolor='black', linewidth=0.5)
    ax0.bar([i + width/2 for i in x], after, width, label='After: tuned',
            color='#fdae6b', edgecolor='black', linewidth=0.5)
    ax0.set_xticks(x)
    ax0.set_xticklabels(model_labels, rotation=15)
    ax0.set_ylabel('Macro-F1'); ax0.set_ylim(0.82, 1.0)
    ax0.set_title('1. Hyperparameter Tuning (before vs after)')
    ax0.legend(fontsize=8); ax0.grid(axis='y', linestyle='--', alpha=0.4)
    for i, (b, a) in enumerate(zip(before, after)):
        if b > 0: ax0.text(i - width/2, b + 0.003, f'{b:.3f}', ha='center', va='bottom', fontsize=7)
        ax0.text(i + width/2, a + 0.003, f'{a:.3f}', ha='center', va='bottom', fontsize=7)

    # 子图 1：预处理优化
    ax1 = axes[1]
    pre = bar_data.get('预处理优化', {})
    labels_pre = list(pre.keys())
    vals_pre   = list(pre.values())
    colors_pre = ['#a1d99b', '#31a354']
    ax1.bar(labels_pre, vals_pre, color=colors_pre, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Macro-F1'); ax1.set_ylim(0.82, 1.0)
    ax1.set_title('2. Preprocessing Optimization')
    ax1.grid(axis='y', linestyle='--', alpha=0.4)
    for i, v in enumerate(vals_pre):
        ax1.text(i, v + 0.003, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    if len(vals_pre) == 2:
        delta = vals_pre[1] - vals_pre[0]
        ax1.annotate('', xy=(1, vals_pre[1]), xytext=(0, vals_pre[0]),
                     arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        ax1.text(0.5, (vals_pre[0] + vals_pre[1]) / 2 + 0.01,
                 f'+{delta:.3f}', ha='center', va='bottom', fontsize=8, color='red')

    # 子图 2：n-gram + 投票融合
    ax2 = axes[2]
    ngram_vals = [bar_data['ngram扩展'][k] for k in ['unigram', 'bigram', 'trigram']]
    voting_vals = [bar_data['投票融合']['单模型最佳'], bar_data['投票融合']['软投票融合']]
    x2_labels = ['uni', 'bi', 'tri', '单模型', '软投票']
    x2_vals   = ngram_vals + voting_vals
    colors2 = ['#3182bd', '#6baed6', '#9ecae1', '#fdae6b', '#e6550d']
    bars = ax2.bar(x2_labels, x2_vals, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Macro-F1'); ax2.set_ylim(0.82, 1.0)
    ax2.set_title('3. n-gram & Voting Ensemble')
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                 f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    _plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_before_after_comparison.png')
    _plt.savefig(out, dpi=150, bbox_inches='tight')
    _plt.close()
    print(f"[保存] {out}")


# ============================================================
def save_results(baseline_results, tuned_results, predictions, output_dir='.', tfidf_params=None):
    """保存实验结果（同时输出 CSV、对齐 TXT、Markdown 表格）"""
    if tfidf_params is None:
        tfidf_params = {}

    # 保存预测结果
    predictions_df = pd.DataFrame(predictions)
    predictions_path = f'{output_dir}/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False, header=False)
    print(f"\n[保存] 预测结果已保存至: {predictions_path}")

    # 构建记录
    records = []
    for model_name in baseline_results:
        records.append({
            'model': model_name,
            'stage': 'baseline',
            'accuracy': baseline_results[model_name]['accuracy'],
            'time_seconds': baseline_results[model_name]['time'],
            'tfidf_max_features': baseline_results[model_name]['tfidf_params']['max_features'],
            'tfidf_ngram': str(baseline_results[model_name]['tfidf_params']['ngram_range']),
            'model_params': str(baseline_results[model_name]['model_params'])
        })

    for model_name in tuned_results:
        records.append({
            'model': model_name,
            'stage': 'tuned',
            'accuracy': tuned_results[model_name]['val_accuracy'],
            'time_seconds': tuned_results[model_name]['time'],
            'tfidf_max_features': tfidf_params.get('max_features', ''),
            'tfidf_ngram': str(tfidf_params.get('ngram_range', '')),
            'model_params': str(tuned_results[model_name]['best_params'])
        })

    results_df = pd.DataFrame(records)

    # 1) CSV（机器可读）
    csv_path = f'{output_dir}/experiments_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"[保存] 实验记录已保存至: {csv_path}")

    # 2) 对齐的 TXT 表格（等宽字体显示效果最好）
    txt_path = f'{output_dir}/experiments_results_table.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(format_aligned_table(results_df))
    print(f"[保存] 对齐表格已保存至: {txt_path}")

    # 3) Markdown 表格（报告里可直接用）
    md_path = f'{output_dir}/experiments_results.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(format_markdown_table(results_df))
    print(f"[保存] Markdown 表格已保存至: {md_path}")

    # 同时在控制台打印对齐表格
    print("\n" + format_aligned_table(results_df))

    return results_df


def format_aligned_table(df):
    """生成对齐的等宽文本表格"""
    cols = ['model', 'stage', 'accuracy', 'time_seconds',
            'tfidf_max_features', 'tfidf_ngram', 'model_params']
    headers = ['模型', '阶段', '准确率', '耗时(s)',
               'TF-IDF特征数', 'ngram', '模型参数']

    # 准备数据
    rows = []
    for _, row in df.iterrows():
        acc = f"{row['accuracy']:.4f}" if isinstance(row['accuracy'], (int, float)) else str(row['accuracy'])
        t = f"{row['time_seconds']:.2f}" if isinstance(row['time_seconds'], (int, float)) else str(row['time_seconds'])
        rows.append([
            str(row['model']),
            str(row['stage']),
            acc,
            t,
            str(row['tfidf_max_features']),
            str(row['tfidf_ngram']),
            str(row['model_params'])
        ])

    # 计算每列最大宽度（中文字符按 2 个宽度算）
    def display_width(s):
        w = 0
        for ch in s:
            w += 2 if ord(ch) > 127 else 1
        return w

    # 计算列宽
    col_widths = []
    for i, header in enumerate(headers):
        max_w = display_width(header)
        for row in rows:
            max_w = max(max_w, display_width(row[i]))
        col_widths.append(max_w)

    def pad(s, w):
        return s + ' ' * (w - display_width(s))

    # 表格分隔线
    sep = '+' + '+'.join('-' * (w + 2) for w in col_widths) + '+'

    lines = [sep]
    # 表头
    header_line = '|' + '|'.join(' ' + pad(headers[i], col_widths[i]) + ' ' for i in range(len(headers))) + '|'
    lines.append(header_line)
    lines.append(sep.replace('+', '+'))
    # 数据行
    for row in rows:
        data_line = '|' + '|'.join(' ' + pad(row[i], col_widths[i]) + ' ' for i in range(len(row))) + '|'
        lines.append(data_line)
    lines.append(sep)

    return '\n'.join(lines)


def format_markdown_table(df):
    """生成 Markdown 表格"""
    lines = []
    lines.append('| 模型 | 阶段 | 准确率 | 耗时(s) | TF-IDF特征数 | ngram | 模型参数 |')
    lines.append('|------|------|--------|---------|--------------|-------|----------|')
    for _, row in df.iterrows():
        acc = f"{row['accuracy']:.4f}" if isinstance(row['accuracy'], (int, float)) else str(row['accuracy'])
        t = f"{row['time_seconds']:.2f}" if isinstance(row['time_seconds'], (int, float)) else str(row['time_seconds'])
        lines.append(
            f"| {row['model']} | {row['stage']} | {acc} | {t} | "
            f"{row['tfidf_max_features']} | {row['tfidf_ngram']} | {row['model_params']} |"
        )
    return '\n'.join(lines)


# ============================================================
# 打印汇总表
# ============================================================
def print_summary(tuned_results):
    """打印最终汇总表"""
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"{'模型':<20} {'验证集准确率':<15} {'最佳参数'}")
    print("-" * 60)
    for model_name, result in sorted(tuned_results.items(), key=lambda x: -x[1]['val_accuracy']):
        print(f"{model_name:<20} {result['val_accuracy']:.4f}         {result['best_params']}")


# ============================================================
# MLP 训练/验证 loss 曲线（回答 question: MLP 是否过拟合？）
# ============================================================
def plot_mlp_loss_curve(X_train, X_val, y_train, y_val, tfidf_params):
    """
    用 sklearn MLPClassifier.partial_fit 手动逐 epoch 训练，
    每 epoch 在外部验证集 (X_val, y_val) 上算一次 acc，画 train_loss / val_acc 双轴曲线。

    关键改进（相对 sklearn.fit + early_stopping 的默认行为）：
      1. validation_scores_ 是 sklearn 从训练集里偷偷切 10% 当内部 val，
         跟 split_data 切出来的 X_val 不是一回事。我们关掉 early_stopping，
         改用 partial_fit 手动控制 epoch，每轮在真验证集上 predict。
      2. 关掉 early_stopping 后能跑满 n_epochs，曲线长度足够观察过拟合拐点。
    """
    print("\n" + "=" * 60)
    print("绘制 MLP 训练/验证 Loss 曲线 (partial_fit 手动训练)")
    print("=" * 60)

    # 1. TF-IDF 向量化
    vectorizer = create_tfidf_vectorizer(**tfidf_params)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf   = vectorizer.transform(X_val)
    print(f"[TF-IDF] 训练: {X_train_tfidf.shape}, 验证: {X_val_tfidf.shape}")

    # 2. 构造 MLP（关键：关 early_stopping，max_iter=1 准备 partial_fit 手动循环）
    n_epochs = 80
    mlp = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=1,                # partial_fit 一次就是 1 epoch
        early_stopping=False,      # 关掉，避免 sklearn 切内部 val
        random_state=RANDOM_STATE,
        warm_start=False,
        verbose=False,
    )

    # 3. partial_fit 必须知道所有类别
    classes = np.unique(y_train)

    # 4. 逐 epoch 训练 + 记录曲线
    train_losses, val_accs, val_f1s = [], [], []
    for epoch in range(1, n_epochs + 1):
        mlp.partial_fit(X_train_tfidf, y_train, classes=classes)
        # 当前 epoch 的 training loss（每个样本的负对数似然均值）
        train_losses.append(mlp.loss_)
        # 外部验证集上算 acc / f1
        y_pred = mlp.predict(X_val_tfidf)
        val_accs.append(accuracy_score(y_val, y_pred))
        val_f1s.append(f1_score(y_val, y_pred, average='macro'))
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            print(f"  epoch {epoch:3d}  train_loss={mlp.loss_:.4f}  "
                  f"val_acc={val_accs[-1]:.4f}  val_f1={val_f1s[-1]:.4f}")

    train_loss = np.array(train_losses)
    val_acc    = np.array(val_accs)
    val_f1     = np.array(val_f1s)
    final_acc  = val_accs[-1]
    final_f1   = val_f1s[-1]
    best_epoch = int(np.argmax(val_acc)) + 1
    best_acc   = val_acc.max()

    print(f"\n[MLP] 训练 {n_epochs} epochs 完成")
    print(f"[MLP] 最终 val_acc={final_acc:.4f}  macro-F1={final_f1:.4f}")
    print(f"[MLP] 最佳 val_acc={best_acc:.4f} 出现在 epoch {best_epoch}")

    # 5. 绘图：双 y 轴（左 train_loss，右 val_acc）
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    epochs_axis = np.arange(1, n_epochs + 1)

    l1, = ax1.plot(epochs_axis, train_loss, color='#d62728', lw=2, label='Train Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Train Loss', color='#d62728', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    l2, = ax2.plot(epochs_axis, val_acc, color='#1f77b4', lw=2, label='Val Accuracy')
    ax2.set_ylabel('Validation Accuracy (X_val)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0.80, 0.95)  # 拉近y轴，看清 val 拐点

    # 标记 val 最佳点
    ax2.axvline(best_epoch, ls=':', color='gray', alpha=0.7)
    ax2.scatter([best_epoch], [best_acc], s=60, color='#1f77b4',
                edgecolor='black', zorder=5, label=f'best (epoch {best_epoch})')

    plt.title(f'MLP Training Curve  (adam, hidden=(100,), '
              f'final val_acc={final_acc:.4f}, best={best_acc:.4f}@ep{best_epoch})',
              fontsize=12)
    ax1.legend(handles=[l1, l2], loc='center right', fontsize=10)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out_path = os.path.join(FIG_DIR, 'fig_mlp_loss_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {out_path}")

    # 6. 过拟合诊断
    print("\n-- 过拟合诊断 --")
    half = n_epochs // 2
    early_val = val_acc[:half].mean()
    late_val  = val_acc[half:].mean()
    print(f"  Val Acc 前 {half} epoch 均值: {early_val:.4f}")
    print(f"  Val Acc 后 {half} epoch 均值: {late_val:.4f}")
    print(f"  最佳 Val Acc: {best_acc:.4f} @ epoch {best_epoch}，"
          f"距末尾 {n_epochs - best_epoch} epoch")
    if late_val < early_val - 0.005:
        print("  → 验证 acc 在后期下降，发生过拟合。")
    elif best_epoch < n_epochs - 10:
        print("  → 验证 acc 在中前期达到峰值后回落，存在过拟合风险。")
    else:
        print("  → 验证 acc 全程未明显下降，模型未观察到明显过拟合。")


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='新闻文本十分类实验')
    parser.add_argument('--mode', type=str, default='full',
                        choices=['full', 'baseline', 'tuning', 'loss'],
                        help='运行模式: full(完整), baseline(仅基线), tuning(仅调参), loss(绘制 MLP 训练/验证 loss 曲线)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2)')
    parser.add_argument('--tfidf-features', type=int, default=10000,
                        help='TF-IDF 最大特征数 (默认: 10000)')
    parser.add_argument('--skip-tuning', action='store_true',
                        help='跳过网格搜索，使用默认参数')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='输出目录 (默认: 当前目录)')
    args = parser.parse_args()

    val_ratio = args.val_ratio

    print("=" * 60)
    print("新闻文本十分类实验")
    print("=" * 60)
    print(f"验证集比例: {val_ratio}")
    print(f"TF-IDF 最大特征数: {args.tfidf_features}")

    # 1. 加载数据
    X, y, X_test = load_data()

    # 2. 划分训练集和验证集
    X_train, X_val, y_train, y_val = split_data(X, y, val_ratio)

    # TF-IDF 参数
    tfidf_params = {
        'max_features': args.tfidf_features,
        'ngram_range': (1, 1),
        'min_df': 2
    }

    # 3. 基线对比
    if args.mode in ['full', 'baseline']:
        baseline_results, _ = run_baseline(X_train, X_val, y_train, y_val, tfidf_params)
    else:
        baseline_results = {}

    # 4. 网格搜索调参
    if args.mode in ['full', 'tuning'] and not args.skip_tuning:
        tuned_results, _ = tune_models(X_train, X_val, y_train, y_val, tfidf_params)
        print_summary(tuned_results)
    else:
        tuned_results = {}

    # 5. 用全部数据重训最佳模型并预测
    if args.mode == 'full' and not args.skip_tuning:
        predictions, best_model, best_score, best_params = retrain_best_on_full_data(
            tuned_results, X_train, X_val, y_train, y_val, X_test, tfidf_params
        )

        # 6. 保存结果
        save_results(baseline_results, tuned_results, predictions, args.output_dir, tfidf_params)

        # ===== 阶段四：优化实验（自动执行，无需开关）=====
        run_optimization_experiments(X_train, X_val, y_train, y_val,
                                     baseline_results, tuned_results, tfidf_params)

        print("\n" + "=" * 60)
        print("实验完成！")
        print(f"最终选择模型: {best_model}")
        print(f"验证集准确率: {best_score:.4f}")
        print("=" * 60)
    elif args.mode == 'baseline':
        print("\n[完成] 基线对比完成，未进行网格搜索和预测")
    elif args.mode == 'loss':
        plot_mlp_loss_curve(X_train, X_val, y_train, y_val, tfidf_params)


if __name__ == '__main__':
    main()
