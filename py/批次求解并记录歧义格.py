import os
import json
import pandas as pd
from ortools.sat.python import cp_model

class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """
    收集 Nonogram 的所有可行解（或上限 max_solutions）。
    """
    def __init__(self, x, max_solutions=1000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._solutions = []
        self._m = len(x)
        self._n = len(x[0]) if self._m > 0 else 0
        self._max_solutions = max_solutions

    def OnSolutionCallback(self):
        sol_matrix = []
        for i in range(self._m):
            row_vals = []
            for j in range(self._n):
                row_vals.append(self.Value(self._x[i][j]))
            sol_matrix.append(row_vals)
        self._solutions.append(sol_matrix)
        if len(self._solutions) >= self._max_solutions:
            self.StopSearch()

    def solutions(self):
        return self._solutions


def build_multi_block_cp_model(m, n, row_constraints, col_constraints):
    """构建多段 Nonogram 的 CP-SAT 模型。"""
    model = cp_model.CpModel()

    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f"x_{i}_{j}")
            row_vars.append(var)
        x.append(row_vars)

    # ---------------------------
    # 行约束
    # ---------------------------
    for i in range(m):
        blocks = row_constraints[i]
        if len(blocks) == 0:
            # 没有黑块 => 此行全白
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue

        # 每行的段起点
        rowBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_col in range(n - length_b + 1):
                starts.append(model.NewBoolVar(f"rB_{i}_{b}_{start_col}"))
            model.Add(sum(starts) == 1)
            rowBlockVars.append((length_b, starts))

        # 相邻段之间至少隔一列空
        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = rowBlockVars[b][1]
            length_next = blocks[b+1]
            starts_next = rowBlockVars[b+1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        # 每个格子由行block的覆盖情况决定
        for col in range(n):
            cover_exps = []
            for (length_b, starts) in rowBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= col < s + length_b:
                        cover_exps.append(var_s)
            if cover_exps:
                model.Add(x[i][col] == sum(cover_exps))
            else:
                model.Add(x[i][col] == 0)

    # ---------------------------
    # 列约束
    # ---------------------------
    for j in range(n):
        blocks = col_constraints[j]
        if len(blocks) == 0:
            # 没有黑块 => 此列全白
            for i in range(m):
                model.Add(x[i][j] == 0)
            continue

        colBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_row in range(m - length_b + 1):
                starts.append(model.NewBoolVar(f"cB_{j}_{b}_{start_row}"))
            model.Add(sum(starts) == 1)
            colBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = colBlockVars[b][1]
            length_next = blocks[b+1]
            starts_next = colBlockVars[b+1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        for i in range(m):
            cover_exps = []
            for (length_b, starts) in colBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= i < s + length_b:
                        cover_exps.append(var_s)
            if cover_exps:
                model.Add(x[i][j] == sum(cover_exps))
            else:
                model.Add(x[i][j] == 0)

    return model, x


def enumerate_solutions_and_compare(json_file_path, max_solutions=50, time_limit=10.0):
    """
    枚举给定 puzzle 的所有(或最多 max_solutions)可行解，
    并与 puzzle["grid"] 对比，记录差异。
    返回一个列表 solutions_info：[{solution_idx, solution_matrix, difference_count, ...}, ...]
    """
    if not os.path.isfile(json_file_path):
        print(f"找不到 JSON 文件: {json_file_path}")
        return []

    with open(json_file_path, 'r') as f:
        puzzle_data = json.load(f)

    row_hints = puzzle_data.get("row_hints", puzzle_data.get("row_constraints"))
    col_hints = puzzle_data.get("col_hints", puzzle_data.get("col_constraints"))
    original_grid = puzzle_data.get("grid", None)  # 用于和解对比

    m = len(row_hints)
    n = len(col_hints)
    if m == 0 or n == 0:
        print(f"警告：网格尺寸异常 (m=0 或 n=0): {json_file_path}")
        return []

    # 构建模型并搜索解
    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    collector = NonogramAllSolutionsCollector(x, max_solutions=max_solutions)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        print(f"Puzzle 无解或不可行: {json_file_path}")
        return []

    all_solutions = collector.solutions()
    solutions_info = []

    for idx, sol_matrix in enumerate(all_solutions):
        # 计算 difference_count
        diff_count = None
        if (original_grid is not None
                and len(original_grid) == m
                and len(original_grid[0]) == n):
            diff = 0
            for i in range(m):
                for j in range(n):
                    if sol_matrix[i][j] != original_grid[i][j]:
                        diff += 1
            diff_count = diff

        # 保存解信息
        info = {
            "solution_idx": idx,
            # 关键：把 0/1 矩阵序列化存储
            "solution_matrix": json.dumps(sol_matrix),
            "difference_count": diff_count
        }
        solutions_info.append(info)

    return solutions_info


def batch_collect_all_solutions(
    multi_csv="multiple_solutions_details.csv",
    out_csv="all_solutions_record.csv",
    max_solutions=50,
    time_limit=10.0
):
    """
    对 multiple_solutions_details.csv 里的拼图进行解枚举，将解和差异信息
    (含 solution_matrix) 保存到 out_csv.
    """
    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return

    df_multi = pd.read_csv(multi_csv)
    if 'json_file_path' not in df_multi.columns:
        print("错误：CSV 中没有 'json_file_path' 列，无法定位 JSON 文件。")
        return

    records = []

    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row["json_file_path"]

        sol_list = enumerate_solutions_and_compare(
            json_file_path=json_path,
            max_solutions=max_solutions,
            time_limit=time_limit
        )

        if not sol_list:
            print(f"Puzzle {puzzle_id} -> 无解或无可行解.")
            continue

        print(f"Puzzle #{puzzle_id}: 找到 {len(sol_list)} 个解 (最多={max_solutions}).")

        # 将每个解记录到 CSV
        for sol_info in sol_list:
            rec = {
                "puzzle_id": puzzle_id,
                "json_file_path": json_path,
                "solution_idx": sol_info["solution_idx"],
                # 这里包含 solution_matrix
                "solution_matrix": sol_info["solution_matrix"],
                "difference_count": sol_info["difference_count"]
            }
            records.append(rec)

    if not records:
        print("\n未生成任何解记录. 可能没有多解或 puzzle 无解.")
        return

    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"\n成功保存到 {out_csv}，其中包含 solution_matrix 列。")


if __name__=="__main__":
    batch_collect_all_solutions(
        multi_csv="multiple_solutions_details.csv",
        out_csv="all_solutions_record.csv",
        max_solutions=1000,
        time_limit=10.0
    )
