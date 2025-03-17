#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import random
from copy import deepcopy
from ortools.sat.python import cp_model

# =========== 1) 枚举解数达 1000 的 CP-SAT 求解函数 ===========

def build_multi_block_cp_model(m, n, row_constraints, col_constraints):
    """
    建立多段 Nonogram 的 CP-SAT 模型。
    row_constraints[i] = [r1, r2, ...]，col_constraints[j] = [c1, c2, ...]
    返回 (model, x)，其中 x[i][j] 是该格是否为黑格的布尔变量。
    """
    model = cp_model.CpModel()
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f'x_{i}_{j}')
            row_vars.append(var)
        x.append(row_vars)

    # 行约束
    for i in range(m):
        blocks = row_constraints[i]
        if not blocks:
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue
        rowBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_col in range(n - length_b + 1):
                starts.append(model.NewBoolVar(f'rB_{i}_{b}_{start_col}'))
            model.Add(sum(starts) == 1)
            rowBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = rowBlockVars[b][1]
            length_next = blocks[b+1]
            starts_next = rowBlockVars[b+1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

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

    # 列约束
    for j in range(n):
        blocks = col_constraints[j]
        if not blocks:
            for i in range(m):
                model.Add(x[i][j] == 0)
            continue
        colBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_row in range(m - length_b + 1):
                starts.append(model.NewBoolVar(f'cB_{j}_{b}_{start_row}'))
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


class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """
    遍历 CP-SAT 所有解，存储解到 self._solutions。
    当解数量 >= max_solutions 时，StopSearch() 提前结束。
    """
    def __init__(self, x, max_solutions=1000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._m = len(x)
        self._n = len(x[0]) if self._m>0 else 0
        self._solutions = []
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


def count_solutions_up_to_1000(puzzle_data, time_limit=10.0):
    """
    返回该 puzzle 实际找到的解数(最多 1000)。若实际解数 > 1000，则只返回 1000。
    """
    row_hints = puzzle_data.get("row_hints", puzzle_data.get("row_constraints"))
    col_hints = puzzle_data.get("col_hints", puzzle_data.get("col_constraints"))
    if not row_hints or not col_hints:
        return 0  # 数据不完整，当无解处理

    m = len(row_hints)
    n = len(col_hints)
    if m==0 or n==0:
        return 0

    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit

    collector = NonogramAllSolutionsCollector(x, max_solutions=1000)
    status = solver.SearchForAllSolutions(model, collector)
    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return 0  # 无解

    found = len(collector.solutions())
    return found


# =========== 2) 在微调中，每次都枚举(最多1000解)以看数量变化 ===========

def local_search_refinement_full_count(puzzle_data, max_iterations=10, time_limit=10.0):
    """
    用随机翻转+贪心的微调，每次都调用 count_solutions_up_to_1000 来获取完整解数(最多1000)。
    这样能看到解数的变化，但会非常耗时。
    """
    # 初始解数
    cur_solutions = count_solutions_up_to_1000(puzzle_data, time_limit=time_limit)
    print(f"微调前：解数 = {cur_solutions} (若==1000则可能实际更多)")

    if cur_solutions <= 1:
        # 若是 0 或 1 就不必微调
        return puzzle_data

    best_grid = deepcopy(puzzle_data["grid"])
    best_count = cur_solutions
    rows = len(best_grid)
    cols = len(best_grid[0]) if rows>0 else 0

    for iteration in range(max_iterations):
        r = random.randint(0, rows-1)
        c = random.randint(0, cols-1)
        old_val = best_grid[r][c]
        # 翻转
        best_grid[r][c] = 1 - old_val

        # 用新 grid 枚举解数(上限1000)
        puzzle_data["grid"] = best_grid
        new_count = count_solutions_up_to_1000(puzzle_data, time_limit=time_limit)

        if new_count < best_count:
            print(f"第{iteration+1}次翻转 ({r},{c}): 解数 {best_count} -> {new_count}")
            best_count = new_count
            # 若 <= 1 已达目的
            if best_count <= 1:
                break
        else:
            # 撤销翻转
            best_grid[r][c] = old_val
            puzzle_data["grid"] = best_grid

    # 最终结果
    print(f"微调结束：最终解数 = {best_count}")
    return puzzle_data


# =========== 3) 批量读取 multiple_solutions_details.csv 并微调 ===========

def batch_refine_count_full(
    multi_csv="multiple_solutions_details.csv",
    output_dir="./test_puzzles_refined",
    max_iterations=10,
    time_limit=10.0
):
    """
    读取 multiple_solutions_details.csv，逐个 puzzle 执行 local_search_refinement_full_count，
    以显示微调前后解数（最多 1000）。
    结果另存到 output_dir 下，不覆盖原文件。
    """

    if not os.path.isfile(multi_csv):
        print(f"错误：找不到 {multi_csv}")
        return

    df = pd.read_csv(multi_csv)
    if 'json_file_path' not in df.columns:
        print("CSV 中缺少 'json_file_path' 列。无法定位 JSON 文件。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    records = []
    for idx, row in df.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row["json_file_path"]
        # 只处理 num_solutions>1 的 puzzle
        nsol = row.get("num_solutions", 2)
        if nsol <= 1:
            # 已是唯一解/无解
            continue

        if not os.path.isfile(json_path):
            print(f"文件缺失: {json_path}")
            continue

        print(f"\n===== 微调 puzzle_id={puzzle_id} 文件: {json_path} =====")
        with open(json_path, 'r') as f:
            puzzle_data = json.load(f)

        # 先算原 puzzle 解数(上限 1000)
        before_count = count_solutions_up_to_1000(puzzle_data, time_limit=time_limit)
        print(f"微调前解数 = {before_count}")

        refined_data = local_search_refinement_full_count(
            puzzle_data,
            max_iterations=max_iterations,
            time_limit=time_limit
        )

        after_count = count_solutions_up_to_1000(refined_data, time_limit=time_limit)
        print(f"微调后解数 = {after_count}")

        # 保存微调后结果到新文件
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        out_json = os.path.join(output_dir, f"{base_name}_refined.json")
        with open(out_json, 'w') as f:
            json.dump(refined_data, f)

        rec = {
            "puzzle_id": puzzle_id,
            "json_file_path": json_path,
            "before_count": before_count,
            "after_count": after_count,
            "refined_json": out_json
        }
        records.append(rec)

    if records:
        df_out = pd.DataFrame(records)
        out_csv = os.path.join(output_dir, "refine_count_full_results.csv")
        df_out.to_csv(out_csv, index=False)
        print(f"\n处理完成：结果记录在 {out_csv}")
    else:
        print("\n没有需要微调的多解拼图，或处理结果为空。")


def main():
    # 示例：从 multiple_solutions_details.csv 读取 puzzle
    batch_refine_count_full(
        multi_csv="multiple_solutions_details.csv",
        output_dir="./test_puzzles_refined",
        max_iterations=10,
        time_limit=10.0
    )

if __name__=="__main__":
    main()

