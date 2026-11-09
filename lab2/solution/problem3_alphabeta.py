"""题目三：极大极小博弈树 + Alpha-Beta 剪枝

数据结构（与题目一致）：
    class TreeNode:
        value    : 叶子得分（叶子节点）
        children : 子节点列表（非叶子节点）
    根节点为 MAX 玩家，层级交替：MAX -> MIN -> MAX ...

返回值：对 MAX 玩家而言的最佳结果。

同时实现了 Minimax 与 Alpha-Beta，并统计访问节点数，
便于报告中对照"剪枝效率"与"子节点访问顺序"的影响。
"""

from __future__ import annotations

from typing import List, Optional


class TreeNode:
    def __init__(self, value: Optional[int] = None, children: Optional[List["TreeNode"]] = None):
        self.value = value
        self.children = children if children is not None else []

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @staticmethod
    def from_dict(d: dict) -> "TreeNode":
        """把测试用例里的 JSON 结构转成树。
        {"is_maximizing_player": true, "children": [...]} 或 {"value": 3}
        """
        if "value" in d:
            return TreeNode(value=d["value"])
        return TreeNode(children=[TreeNode.from_dict(c) for c in d["children"]])


# ---------------------------------------------------------------- Minimax
def minimax(node: TreeNode, is_max: bool, stats: Optional[dict] = None) -> int:
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if node.is_leaf:
        return node.value

    values = [minimax(child, not is_max, stats) for child in node.children]
    return max(values) if is_max else min(values)


# ------------------------------------------------------------ Alpha-Beta
def alpha_beta(
    node: TreeNode,
    is_max: bool,
    alpha: float = float("-inf"),
    beta: float = float("inf"),
    stats: Optional[dict] = None,
) -> int:
    if stats is None:
        stats = {}
    stats["visited"] = stats.get("visited", 0) + 1

    if node.is_leaf:
        return node.value

    if is_max:
        value = float("-inf")
        for child in node.children:
            value = max(value, alpha_beta(child, False, alpha, beta, stats))
            alpha = max(alpha, value)
            if value >= beta:          # β 剪枝
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value
    else:
        value = float("inf")
        for child in node.children:
            value = min(value, alpha_beta(child, True, alpha, beta, stats))
            beta = min(beta, value)
            if value <= alpha:         # α 剪枝
                stats["pruned"] = stats.get("pruned", 0) + 1
                break
        return value


def solve_tree(root: TreeNode, use_alpha_beta: bool = True, stats: Optional[dict] = None) -> int:
    if use_alpha_beta:
        return alpha_beta(root, True, float("-inf"), float("inf"), stats)
    return minimax(root, True, stats)


def build_sample_tree() -> TreeNode:
    """题目给的样例树（注意：中间层的 min2 = [max2, max3] 复用了 max2）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])

    return TreeNode(children=[min1, min2])


def count_nodes(node: TreeNode) -> int:
    return 1 + sum(count_nodes(c) for c in node.children)


if __name__ == "__main__":
    root = build_sample_tree()
    print("Minimax        :", solve_tree(root, use_alpha_beta=False))
    print("Alpha-Beta     :", solve_tree(root, use_alpha_beta=True))
    # 期望均为 5
