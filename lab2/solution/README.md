# 实验二：搜索算法（A* + Alpha-Beta）

## 目录结构

```
solution/
├── astar.py               # 通用 A* 实现（三个题共用，支持 weight 加权 + f 并列 tie-breaking）
├── problem1_maze.py       # 题目一：迷宫寻路
├── problem2_weighted.py   # 题目二：加权地形
├── problem3_alphabeta.py  # 题目三：极大极小 + Alpha-Beta 剪枝
├── problem4_game.py       # 题目四：竞争环境下的宝藏分配（含距离矩阵缓存 + 后继排序）
├── run_all_tests.py       # 统一测试入口（7 条样例校验）
└── experiments.py         # 对照实验一键脚本（E1~E5，报告数据来源）
```

## 运行

```bash
cd solution
python run_all_tests.py     # 全部样例校验
python experiments.py       # 一键跑出报告所需的全部对照数据（E1~E5）
python problem1_maze.py     # 单独跑某一题
python problem4_game.py
```

> Windows 控制台默认 GBK，若中文输出乱码，先执行 `$env:PYTHONIOENCODING="utf-8"`。

## 每题要点

### 题目一 迷宫寻路
- 状态：格子坐标 `(row, col)`
- g：已走步数；h：曼哈顿距离 `|Δr| + |Δc|`；f = g + h
- 优先队列按 f 弹出，每次扩展 4 邻居；终点第一次弹出即最优
- 无解返回 `[]`；`use_heuristic=False` 即 h=0（退化为 Dijkstra），可作对照实验

### 题目二 加权地形
- 进入格子的代价 = 该格地形权值；`S`、`E` 代价为 0
- 启发函数 `manhattan * min_cost`（min_cost = 最小正权值 1）
  - 每步至少花 min_cost，所以缩放后的曼哈顿距离不高估 → 可采纳
  - 若乘上平均值或最大值就会高估，破坏最优性
- 无解返回 `-1`；`solve_weighted_with_path` 可打印逐步累计代价（报告用）

### 题目三 Alpha-Beta 剪枝
- MAX/MIN 交替，根为 MAX
- Minimax 与 Alpha-Beta 都实现并统计访问节点数（`visited`）与剪枝次数（`pruned`）
- 可改变 `children` 顺序对照剪枝效果
- 样例输出 5（Minimax 与 Alpha-Beta 一致）

### 题目四 宝藏分配（综合题）
- 用 A* 先算 `{A, B, T1..T4}` 两两最小代价，得 6×6 距离矩阵
- 博弈状态：`pos_a, pos_b, remaining, score_a, score_b, is_max`
- 每回合 Gain(p, Ti) = Vi − d(p, Ti)，U = Score_A − Score_B
- A 最大化 U，B 最小化 U；深度 H 或宝藏取完时用当前 U 评估
- 宝藏价值：T1=9, T2=6, T3=7, T4=7（PDF 中 T2/T3 的价值显示为页面底部的 "6 7"）
- 预期：H=2 → `T2, -1`；H=4 → `T3, 3`

## 距离矩阵（题目四）

|      | A  | B  | T1 | T2 | T3 | T4 |
|------|----|----|----|----|----|----|
| A    | 0  | 12 | 10 | 4  | 8  | 8  |
| B    | 12 | 0  | 8  | 8  | 4  | 10 |
| T1   | 10 | 8  | 0  | 6  | 6  | 12 |
| T2   | 4  | 8  | 6  | 0  | 6  | 6  |
| T3   | 8  | 4  | 6  | 6  | 0  | 6  |
| T4   | 8  | 10 | 12 | 6  | 6  | 0  |

## 剪枝统计（自动包含在返回值中）

| H | 算法       | 访问节点 | 剪枝次数 |
|---|-----------|---------|---------|
| 2 | Minimax    | 16      | 0       |
| 2 | Alpha-Beta | 16      | 0       |
| 4 | Minimax    | 64      | 0       |
| 4 | Alpha-Beta | 56      | 11      |
| 4 | Alpha-Beta + 后继排序 | 48 | 11 |

深度浅、分支少时剪枝收益不大；H=4 时 Alpha-Beta 明显减少节点访问，
后继排序（按即时 gain 降序）进一步把访问节点从 56 降到 48，最优值不变。

## P1 优化（创新分用，已实现）

- **加权 A\***：`astar(..., weight=w)`，f = g + w·h。样例迷宫上 w=1.5 仍最优（扩展 13），
  w=2 路径变长 8→10 但扩展降到 10（次优更快）。见 `experiments.py` 的 E5。
- **f 并列 tie-breaking**：f 相同时优先扩展 h 更小的节点（已默认启用，内置于堆序中）。
- **后继排序 Move Ordering**：`solve_game(..., sort_moves=True)`，MAX/MIN 层都优先试
  即时 gain 最大的宝藏，H=4 时 visited 56→48，最优值不变。见 E4。
- **距离矩阵缓存**：`build_distance_matrix(use_cache=True)`，solve_game 被调用 4 次时
  A\* 调用从 4×30=120 次降为 30 次（对角自环跳过）。见 E5。

## 对照实验（`python experiments.py` 一键跑出）

| 编号 | 内容 | 关键数据 |
|---|---|---|
| E1 | h=0 vs 曼哈顿 | 样例 16→13（-18.8%）；20×20 随机 261→98（-62.3%） |
| E2 | 可采纳性反例 | 自造地图：h×1 代价 5（最优），h×8 代价 24（贪心穿沼泽，次优） |
| E3 | 32 种访问顺序 | AB 的 visited 在 11/12/14/15 四档（各 8 种），Minimax 恒 15 |
| E4 | H=2/H=4 决策 + 剪枝 | H=2 选 T2，H=4 选 T3；H=4 时 64→56→48 |
| E5 | 加权 A\* + 缓存 | w=2 长度 8→10、扩展 13→10；缓存 120→30 次 A\* |

## 报告建议（数据来源标注）

- 题目一：对比 h=0 与曼哈顿距离的 `expanded` 节点数
- 题目二：展示代价累计过程，讨论可采纳性
- 题目三：改变子节点顺序对比 `visited`/`pruned`
- 题目四：H=2 vs H=4 决策差异，Minimax vs Alpha-Beta 开销对比
- 可自设随机地图/随机树做额外对照实验
