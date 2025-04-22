import time

from matplotlib import pyplot as plt
from ortools.sat.python import cp_model

class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """
    遍历 CP-SAT 所有解，把解保存下来。
    """
    def __init__(self, x):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._solutions = []
        self._m = len(x)
        self._n = len(x[0]) if self._m > 0 else 0

    def on_solution_callback(self):
        self.OnSolutionCallback()

    def OnSolutionCallback(self):
        sol_matrix = []
        for i in range(self._m):
            row_vals = []
            for j in range(self._n):
                row_vals.append(self.Value(self._x[i][j]))
            sol_matrix.append(row_vals)
        self._solutions.append(sol_matrix)

    def solutions(self):
        return self._solutions

def build_multi_block_cp_model(m, n, row_constraints, col_constraints):

    model = cp_model.CpModel()

    # 定义网格变量 x[i][j]，布尔型
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f"x_{i}_{j}")
            row_vars.append(var)
        x.append(row_vars)

    # ---------------------
    # 1) 多段行约束
    # 对于第 i 行，row_constraints[i] 可能是 [2,1] 等多个段
    # 我们用 "行块布尔变量" rowBlockVars[(i, b, start)] 表示：
    # “第 i 行的第 b 段黑块，是否从列 start 开始”
    # 并确保这一行中每一段都正好出现一次，且相邻段要至少隔 1 列。
    # ---------------------
    rowBlockVars = {}
    for i in range(m):
        blocks = row_constraints[i]
        # 若这一行没有任何黑块(空约束)，说明整行都是 0
        if len(blocks) == 0:
            # 那就直接约束此行所有 x[i][j] = 0
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue

        # 否则，为本行每个“段”定义可能的起始位置
        for b, length_b in enumerate(blocks):
            for start_col in range(n - length_b + 1):
                rowBlockVars[(i, b, start_col)] = model.NewBoolVar(
                    f"rB_{i}_{b}_{start_col}"
                )

        # (1.1) 每段必须且只会选一种 start
        for b, length_b in enumerate(blocks):
            possible_starts = []
            for start_col in range(n - length_b + 1):
                possible_starts.append(rowBlockVars[(i, b, start_col)])
            model.Add(sum(possible_starts) == 1)

        # (1.2) 相邻段之间至少隔 1 个空白
        # 例如段 b 从 start_s 开始，则段 b+1 的 start_t >= start_s + length_b + 1
        # 等价于：若 start_t < start_s + length_b + 1，则二者不可能同时为真
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(n - length_b + 1):
                for start_t in range(n - blocks[b + 1] + 1):
                    # 如果 start_t < start_s + length_b + 1，则不能同时为1
                    if start_t < start_s + length_b + 1:
                        model.Add(
                            rowBlockVars[(i, b, start_s)]
                            + rowBlockVars[(i, b + 1, start_t)]
                            <= 1
                        )

        # (1.3) 对于每个格子 x[i][col]，它为 1 当且仅当“落在某段覆盖区间上”
        #       coverage = sum( rowBlockVars[(i,b,s)] for 所有 b,s 覆盖 col )
        #       x[i][col] == coverage
        # 为了实现“当且仅当”效果，这里用两条不等式：
        #   x[i][col] <= coverage
        #   x[i][col] >= coverage
        # 其中 coverage 是若干布尔之和，若不允许不同段重叠，则 coverage 最多为1。
        for col in range(n):
            cover_vars = []
            for b, length_b in enumerate(blocks):
                for start_s in range(n - length_b + 1):
                    if start_s <= col < start_s + length_b:
                        # 若这段从 start_s 开始且长度 length_b，则覆盖 [start_s, start_s+length_b-1]
                        cover_vars.append(rowBlockVars[(i, b, start_s)])
            if len(cover_vars) == 0:
                # 说明没有任何段会覆盖到 col
                model.Add(x[i][col] == 0)
            else:
                model.Add(x[i][col] <= sum(cover_vars))
                model.Add(x[i][col] >= sum(cover_vars))

    # ---------------------
    # 2) 多段列约束 (同理)
    # 对于第 j 列，col_constraints[j] 可能是 [3,2] 等多个段
    # 用 "列块布尔变量" colBlockVars[(j, b, start)] 表示：
    # “第 j 列的第 b 段黑块，是否从行 start 开始”
    # ---------------------
    colBlockVars = {}
    for j in range(n):
        blocks = col_constraints[j]
        if len(blocks) == 0:
            # 没有黑块，整列皆 0
            for i in range(m):
                model.Add(x[i][j] == 0)
            continue

        for b, length_b in enumerate(blocks):
            for start_row in range(m - length_b + 1):
                colBlockVars[(j, b, start_row)] = model.NewBoolVar(
                    f"cB_{j}_{b}_{start_row}"
                )

        # (2.1) 每段在该列只能有一种起始位置
        for b, length_b in enumerate(blocks):
            possible_starts = []
            for start_row in range(m - length_b + 1):
                possible_starts.append(colBlockVars[(j, b, start_row)])
            model.Add(sum(possible_starts) == 1)

        # (2.2) 相邻段要留 1 行空格
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            for start_s in range(m - length_b + 1):
                for start_t in range(m - blocks[b + 1] + 1):
                    if start_t < start_s + length_b + 1:
                        model.Add(
                            colBlockVars[(j, b, start_s)]
                            + colBlockVars[(j, b + 1, start_t)]
                            <= 1
                        )

        # (2.3) 每个格子 x[row][j] 由该列的段覆盖情况来确定
        for i in range(m):
            cover_vars = []
            for b, length_b in enumerate(blocks):
                for start_s in range(m - length_b + 1):
                    if start_s <= i < start_s + length_b:
                        cover_vars.append(colBlockVars[(j, b, start_s)])
            if len(cover_vars) == 0:
                model.Add(x[i][j] == 0)
            else:
                model.Add(x[i][j] <= sum(cover_vars))
                model.Add(x[i][j] >= sum(cover_vars))

    return model, x

def find_all_solutions_cp(m, n, row_constraints, col_constraints):
    """
    调用 build_multi_block_cp_model 构建多段 Nonogram 模型，并遍历所有可行解。
    row_constraints[i]: 第 i 行的段列表，如 [2,1]
    col_constraints[j]: 第 j 列的段列表，如 [3,2]
    """
    model, x = build_multi_block_cp_model(m, n, row_constraints, col_constraints)

    # 创建解收集器并启动搜索
    solver = cp_model.CpSolver()
    solution_collector = NonogramAllSolutionsCollector(x)
    status = solver.SearchForAllSolutions(model, solution_collector)

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        print("No solution found or the model is infeasible.")
        return []
    return solution_collector.solutions()

if __name__ == "__main__":
    row_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    col_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    m = len(row_constraints)
    n = len(col_constraints)
    start_time = time.time()
    solutions = find_all_solutions_cp(m, n, row_constraints, col_constraints)
    end_time = time.time()

    print(f"找到 {len(solutions)} 个可行解, 用时 {end_time - start_time:.8f} 秒，CP-SAT求解器计算.\n")
    for idx, sol in enumerate(solutions[:3], start=1):  # 如需展示更多改 [:3]
        plt.imshow(sol, cmap="gray_r", interpolation="none")
        plt.axis("off")
        plt.title(f"Solution #{idx}")
        plt.show()