#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import pandas as pd
import random
from copy import deepcopy
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model


########################################
# 1) 辅助函数：计算行/列提示
########################################
def compute_row_col_hints(grid):
    m = len(grid)
    n = len(grid[0]) if m else 0

    row_hints = []
    for i in range(m):
        hint = []
        count = 0
        for val in grid[i]:
            if val == 1:
                count += 1
            else:
                if count:
                    hint.append(count)
                    count = 0
        if count:
            hint.append(count)
        row_hints.append(hint)

    col_hints = []
    for j in range(n):
        hint = []
        count = 0
        for i in range(m):
            if grid[i][j] == 1:
                count += 1
            else:
                if count:
                    hint.append(count)
                    count = 0
        if count:
            hint.append(count)
        col_hints.append(hint)

    return row_hints, col_hints


########################################
# 2) CP-SAT 求解器相关：判断谜题解数
########################################
class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):

    def __init__(self, x, max_solutions=1000):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._x = x
        self._solutions = []
        self._m = len(x)
        self._n = len(x[0]) if self._m > 0 else 0
        self._max_solutions = max_solutions

    def OnSolutionCallback(self):
        sol = []
        for i in range(self._m):
            row = []
            for j in range(self._n):
                row.append(self.Value(self._x[i][j]))
            sol.append(row)
        self._solutions.append(sol)
        if len(self._solutions) >= self._max_solutions:
            self.StopSearch()

    def solutions(self):
        return self._solutions


def build_multi_block_cp_model(m, n, row_constraints, col_constraints):

    model = cp_model.CpModel()
    x = []
    for i in range(m):
        row_vars = []
        for j in range(n):
            var = model.NewBoolVar(f"x_{i}_{j}")
            row_vars.append(var)
        x.append(row_vars)

    # 行约束
    for i in range(m):
        blocks = row_constraints[i]
        if not blocks:
            # 无黑格 -> 全部置0
            for j in range(n):
                model.Add(x[i][j] == 0)
            continue

        rowBlockVars = []
        for b, length_b in enumerate(blocks):
            starts = []
            for start_col in range(n - length_b + 1):
                starts.append(model.NewBoolVar(f"rB_{i}_{b}_{start_col}"))
            model.Add(sum(starts) == 1)
            rowBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = rowBlockVars[b][1]
            starts_next = rowBlockVars[b + 1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        for j in range(n):
            cover_vars = []
            for (length_b, starts) in rowBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= j < s + length_b:
                        cover_vars.append(var_s)
            if cover_vars:
                model.Add(x[i][j] == sum(cover_vars))
            else:
                model.Add(x[i][j] == 0)

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
                starts.append(model.NewBoolVar(f"cB_{j}_{b}_{start_row}"))
            model.Add(sum(starts) == 1)
            colBlockVars.append((length_b, starts))

        for b in range(len(blocks) - 1):
            length_b = blocks[b]
            starts_b = colBlockVars[b][1]
            starts_next = colBlockVars[b + 1][1]
            for s1, var1 in enumerate(starts_b):
                for s2, var2 in enumerate(starts_next):
                    if s2 < s1 + length_b + 1:
                        model.Add(var1 + var2 <= 1)

        for i in range(m):
            cover_vars = []
            for (length_b, starts) in colBlockVars:
                for s, var_s in enumerate(starts):
                    if s <= i < s + length_b:
                        cover_vars.append(var_s)
            if cover_vars:
                model.Add(x[i][j] == sum(cover_vars))
            else:
                model.Add(x[i][j] == 0)

    return model, x


def count_solutions_up_to_1000(puzzle_data, time_limit=100.0):

    row_hints = puzzle_data.get("row_hints")
    col_hints = puzzle_data.get("col_hints")
    if not row_hints or not col_hints:
        return 0
    m = len(row_hints)
    n = len(col_hints)
    if m == 0 or n == 0:
        return 0

    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    collector = NonogramAllSolutionsCollector(x, max_solutions=1000)
    status = solver.SearchForAllSolutions(model, collector)

    if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
        return 0
    return len(collector.solutions())


########################################
# 3) 连通性检测等函数
########################################
def count_black_components(grid):

    m = len(grid)
    if m == 0:
        return 0
    n = len(grid[0])
    visited = [[False] * n for _ in range(m)]

    def dfs(i, j):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= m or cj < 0 or cj >= n:
                continue
            if visited[ci][cj] or grid[ci][cj] == 0:
                continue
            visited[ci][cj] = True
            for di, dj in [(-1, -1), (-1, 0), (-1, 1),
                           (0, -1), (0, 1),
                           (1, -1), (1, 0), (1, 1)]:
                stack.append((ci + di, cj + dj))

    comp_count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                comp_count += 1
                dfs(i, j)
    return comp_count


def count_white_components(grid):
    inverted = [[1 - cell for cell in row] for row in grid]
    return count_black_components(inverted)


def allowed_to_flip(original_grid, i, j):
    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)

    candidate = deepcopy(original_grid)
    candidate[i][j] = 1 - candidate[i][j]

    new_black = count_black_components(candidate)
    new_white = count_white_components(candidate)
    return not (new_black > orig_black or new_white > orig_white)


########################################
# 4) 生成原图与微调后图的对比图片
########################################
def save_comparison_image(original_grid, refined_grid, out_img_path):
    """
    将原图 / 微调后图 / 差分图并排显示并保存。
    差分图表示两个grid对应位置不同则显示1，否则0。
    """
    import matplotlib.pyplot as plt
    import numpy as np

    original_arr = np.array(original_grid)
    refined_arr = np.array(refined_grid)
    diff_arr = np.abs(original_arr - refined_arr)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(original_arr, cmap='binary', interpolation='nearest')
    ax[0].set_title("Original Grid")
    ax[0].axis('off')

    ax[1].imshow(refined_arr, cmap='binary', interpolation='nearest')
    ax[1].set_title("Refined Grid")
    ax[1].axis('off')

    ax[2].imshow(diff_arr, cmap='gray', interpolation='nearest')
    ax[2].set_title("Difference (Flips)")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_img_path)
    plt.close(fig)


########################################
# 5) MIP 优化方法 (mip_refinement)
#    并输出 iteration_used, time_spent, final_hamming_distance, success
########################################
def compute_modified_grid(original_grid, delta):
    """
    根据 MIP解出的 delta，将 original_grid 对应位置翻转 (XOR)。
    delta[i][j]=1 表示该格子需要翻转。
    """
    m = len(original_grid)
    n = len(original_grid[0]) if m else 0
    new_grid = []
    for i in range(m):
        new_row = []
        for j in range(n):
            new_row.append(original_grid[i][j] ^ delta[i][j])
        new_grid.append(new_row)
    return new_grid


def compute_ambiguity_weight(original_grid):
    """
    利用 CP-SAT 求解器枚举部分解(上限50)，统计与原图不同的格子频率，
    并返回一个 weight 矩阵： weight[i][j] = 1 / (freq + 1)
    歧义格频率越高 -> 权重越小 -> 修改成本更低
    """
    m = len(original_grid)
    n = len(original_grid[0])
    # 先算 hints
    row_hints, col_hints = compute_row_col_hints(original_grid)
    puzzle_temp = {"grid": original_grid, "row_hints": row_hints, "col_hints": col_hints}

    model, x = build_multi_block_cp_model(m, n, row_hints, col_hints)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    collector = NonogramAllSolutionsCollector(x, max_solutions=50)
    solver.SearchForAllSolutions(model, collector)
    solutions = collector.solutions()

    ambiguous_freq = {}
    for sol in solutions:
        for i in range(m):
            for j in range(n):
                if sol[i][j] != original_grid[i][j]:
                    ambiguous_freq[(i, j)] = ambiguous_freq.get((i, j), 0) + 1

    weight = [[1.0 / (ambiguous_freq.get((i, j), 0) + 1) for j in range(n)] for i in range(m)]
    return weight


def mip_refinement(puzzle_data, mip_time_limit=60, max_mip_iterations=10):
    """
    基于 MIP 求解器对谜题进行最小修改，使得修改后的谜题唯一解，
    且保证黑白连通分量不增(allowed_to_flip)。
    同时利用歧义格信息对目标函数加权(歧义越多->权重越低->修改成本越低)。

    返回一个 result dict:
    {
        "final_grid": ...,
        "iteration_used": ...,
        "time_spent": ...,
        "final_hamming_distance": ...,
        "success": ...
    }
    """
    start_time = time.time()

    original_grid = deepcopy(puzzle_data["grid"])
    m = len(original_grid)
    n = len(original_grid[0])

    # 计算行列提示
    orig_row_hints, orig_col_hints = compute_row_col_hints(original_grid)
    puzzle_data["row_hints"] = orig_row_hints
    puzzle_data["col_hints"] = orig_col_hints

    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)

    # 构造 allowed_mask: 若单翻转(i,j)会增加连通分量,就不允许
    allowed_mask = [[1 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if not allowed_to_flip(original_grid, i, j):
                allowed_mask[i][j] = 0

    # 计算歧义格加权
    weight_matrix = compute_ambiguity_weight(original_grid)

    # 创建MIP模型
    prob = LpProblem("Nonogram_Modification", LpMinimize)

    # 定义二元变量 delta[i][j]
    delta = [[None] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if allowed_mask[i][j] == 1:
                delta[i][j] = LpVariable(f"delta_{i}_{j}", cat=LpBinary)
            else:
                # 不允许翻转 -> 强制0
                delta[i][j] = 0

    # 目标函数: 加权最小化修改数
    prob += lpSum(
        weight_matrix[i][j] * delta[i][j] if allowed_mask[i][j] == 1 else 0
        for i in range(m)
        for j in range(n)
    ), "Total_Weighted_Modifications"

    iteration_used = 0
    success = False
    final_grid = deepcopy(original_grid)

    # 用来存 "排除已经找到的 candidate" 的 cut 约束
    cuts = []

    for iteration in range(max_mip_iterations):
        iteration_used += 1
        print(f"MIP 迭代 {iteration + 1} 开始求解...")

        # 求解
        status = prob.solve(PULP_CBC_CMD(timeLimit=mip_time_limit, msg=0))
        if status != 1:  # 1=optimal
            print("MIP 求解失败或超时.")
            break

        # 提取 MIP 解
        candidate_delta = [[int(delta[i][j].varValue) if allowed_mask[i][j] == 1 else 0
                            for j in range(n)] for i in range(m)]
        candidate_grid = compute_modified_grid(original_grid, candidate_delta)

        # 统计: 解数, black/white连通分量, 修改格数
        new_row_hints, new_col_hints = compute_row_col_hints(candidate_grid)
        candidate_puzzle = {
            "grid": candidate_grid,
            "row_hints": new_row_hints,
            "col_hints": new_col_hints
        }
        sol_count = count_solutions_up_to_1000(candidate_puzzle, time_limit=30)
        cand_black = count_black_components(candidate_grid)
        cand_white = count_white_components(candidate_grid)
        mod_count = sum(sum(r) for r in candidate_delta)

        print(f"[Iteration {iteration + 1}] 修改 {mod_count} 格子 -> 解数={sol_count}, "
              f"blackComp={cand_black}, whiteComp={cand_white}")

        # 判断是否满足条件
        if sol_count == 1 and cand_black <= orig_black and cand_white <= orig_white:
            print("找到唯一解且黑白连通分量未增.")
            final_grid = candidate_grid
            success = True
            break
        else:
            # 排除该 candidate
            # 对candidate_delta里=1的格子 -> delta[i][j],
            #  =0的格子 -> (1-delta[i][j])，将这些相加 <= total_allowed - 1
            cut_expr = lpSum(
                delta[i][j] if candidate_delta[i][j] == 1 else (1 - delta[i][j])
                for i in range(m) for j in range(n) if allowed_mask[i][j] == 1
            )
            total_allowed = sum(allowed_mask[i][j] for i in range(m) for j in range(n))
            prob += cut_expr <= total_allowed - 1, f"Cut_{iteration}"
            cuts.append(cut_expr)
            print(f"添加 cut 约束，当前cut总数={len(cuts)}")

    end_time = time.time()
    time_spent = end_time - start_time

    # 计算最终与原图的汉明距离
    diff_count = 0
    for r1, r2 in zip(original_grid, final_grid):
        for a, b in zip(r1, r2):
            if a != b:
                diff_count += 1

    # 返回结果
    result = {
        "final_grid": final_grid,
        "iteration_used": iteration_used,
        "time_spent": time_spent,
        "final_hamming_distance": diff_count,
        "success": success
    }
    return result


########################################
# 6) 批量处理函数
#    从CSV读多解记录 -> 调用 mip_refinement -> 保存对比图/JSON -> 记录结果
########################################
def batch_mip_refine_and_save_all(multi_csv="multiple_solutions_details.csv",
                                  out_json_dir="./mip_refined_jsons",
                                  out_img_dir="./mip_refined_images",
                                  mip_time_limit=60,
                                  max_mip_iterations=10):
    """
    批量对多解puzzle执行MIP微调, 并将结果存到 'mip_refinement_results.csv'
    """
    if not os.path.isfile(multi_csv):
        print(f"错误: 找不到 {multi_csv}")
        return

    df = pd.read_csv(multi_csv)
    df_multi = df[df["num_solutions"] > 1]  # 仅处理多解
    if df_multi.empty:
        print("没有多解谜题记录, 退出.")
        return

    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    records = []

    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"row_{idx}")
        json_path = row.get("json_file_path", None)
        if not json_path or not os.path.isfile(json_path):
            print(f"[警告] puzzle_id={puzzle_id}: 缺少有效 json_file_path 或文件不存在.")
            continue

        print(f"\n===== MIP 微调 Puzzle {puzzle_id} ({json_path}) =====")
        with open(json_path, 'r', encoding='utf-8') as f:
            puzzle_data = json.load(f)

        original_grid = deepcopy(puzzle_data["grid"])

        # 调用 MIP 微调
        result = mip_refinement(
            puzzle_data,
            mip_time_limit=mip_time_limit,
            max_mip_iterations=max_mip_iterations
        )
        refined_grid = result["final_grid"]

        # 保存对比图
        out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        save_comparison_image(original_grid, refined_grid, out_img_path)

        # 保存 refined JSON
        refined_json_path = os.path.join(out_json_dir, f"puzzle_{puzzle_id}_mip_refined.json")
        puzzle_refined_data = {
            "grid": refined_grid,
            "row_hints": compute_row_col_hints(refined_grid)[0],
            "col_hints": compute_row_col_hints(refined_grid)[1]
        }
        with open(refined_json_path, 'w', encoding='utf-8') as jf:
            json.dump(puzzle_refined_data, jf, ensure_ascii=False, indent=2)

        record = {
            "puzzle_id": puzzle_id,
            "json_file_path": json_path,
            "num_solutions": row["num_solutions"],

            "iteration_used": result["iteration_used"],
            "time_spent": result["time_spent"],
            "final_hamming_distance": result["final_hamming_distance"],
            "success": result["success"],

            "refined_img_path": out_img_path,
            "refined_json_path": refined_json_path
        }
        records.append(record)

        print(f"Puzzle {puzzle_id} 完毕: success={record['success']}, "
              f"iter={record['iteration_used']}, time={record['time_spent']:.2f}s, "
              f"hamming_distance={record['final_hamming_distance']}")

    # 汇总写入CSV
    out_csv = "mip_refinement_results.csv"
    df_out = pd.DataFrame(records)
    df_out.to_csv(out_csv, index=False)
    print(f"\n=== 批量处理结束，共处理 {len(records)} 个拼图 ===")
    print(f"结果已保存到 {out_csv}")


########################################
# 7) main
########################################
def main():
    batch_mip_refine_and_save_all(
        multi_csv="multiple_solutions_details.csv",
        out_json_dir="./mip_refined_jsons",
        out_img_dir="./mip_refined_images",
        mip_time_limit=60,
        max_mip_iterations=200
    )


if __name__ == "__main__":
    main()
