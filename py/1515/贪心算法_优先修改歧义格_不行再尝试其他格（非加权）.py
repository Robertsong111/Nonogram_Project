import os
import json
import time

import numpy as np
import pandas as pd
from copy import deepcopy
import random

from matplotlib import pyplot as plt
from ortools.sat.python import cp_model
from PIL import Image


############################################################
# 1) 这里是你新的非加权贪心微调函数，带计时、迭代次数、汉明距离、success 等
############################################################

def compute_row_col_hints(grid):
    m = len(grid)
    n = len(grid[0]) if m else 0
    row_hints = []
    for i in range(m):
        count = 0
        hints = []
        for val in grid[i]:
            if val == 1:
                count += 1
            else:
                if count > 0:
                    hints.append(count)
                    count = 0
        if count > 0:
            hints.append(count)
        row_hints.append(hints)

    col_hints = []
    for j in range(n):
        count = 0
        hints = []
        for i in range(m):
            if grid[i][j] == 1:
                count += 1
            else:
                if count > 0:
                    hints.append(count)
                    count = 0
        if count > 0:
            hints.append(count)
        col_hints.append(hints)

    return row_hints, col_hints


class NonogramAllSolutionsCollector(cp_model.CpSolverSolutionCallback):
    """搜集最多 max_solutions 个可行解，若达到上限则中止搜索。"""

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


def count_solutions_up_to_1000(puzzle_data, time_limit=10.0):
    """调用 CP-SAT 求解器，统计 up to 1000 个可行解数。"""
    row_hints = puzzle_data.get("row_hints", puzzle_data.get("row_constraints"))
    col_hints = puzzle_data.get("col_hints", puzzle_data.get("col_constraints"))
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


def count_black_components(grid):
    """统计网格中黑色连通分量数。可用于判断翻转是否破坏连通性。"""
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
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                stack.append((ci + di, cj + dj))

    count = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1 and not visited[i][j]:
                count += 1
                dfs(i, j)
    return count


def count_white_components(grid):
    """统计白色连通分量，方法是将网格反转后调用上面函数。"""
    inverted = [[1 - cell for cell in row] for row in grid]
    return count_black_components(inverted)


def allowed_to_flip(original_grid, i, j):
    """判断翻转 (i,j) 是否会增加黑/白连通分量。若增加则不允许。"""
    orig_black = count_black_components(original_grid)
    orig_white = count_white_components(original_grid)

    candidate = deepcopy(original_grid)
    candidate[i][j] = 1 - candidate[i][j]

    new_black = count_black_components(candidate)
    new_white = count_white_components(candidate)

    # 若翻转后黑连通分量或白连通分量增加，则不允许翻转
    if new_black > orig_black or new_white > orig_white:
        return False
    return True


def greedy_refinement(puzzle_data, max_iterations=50, time_limit=10.0):
    """
    非加权贪心微调：
      1) 枚举部分解(最多 50 个)，找出所有“歧义格”(与当前状态不一致的格子)；
      2) 随机选取一个歧义格翻转，若解数降低则接受；
      3) 重复直到解数=1 或达到 max_iterations。

    返回结果字典，含：
      - "final_grid": 最终微调后棋盘
      - "iteration_used": 实际迭代次数
      - "time_spent": 求解总耗时(秒)
      - "final_hamming_distance": 最终与原图的汉明距离
      - "success": bool，是否达到唯一解
    """
    original_grid = deepcopy(puzzle_data["grid"])
    current_state = deepcopy(original_grid)

    # 预先补齐提示
    row_h, col_h = compute_row_col_hints(current_state)
    puzzle_data["row_hints"] = row_h
    puzzle_data["col_hints"] = col_h

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # 计算当前解数
        candidate_puzzle = {
            "grid": current_state,
            "row_hints": compute_row_col_hints(current_state)[0],
            "col_hints": compute_row_col_hints(current_state)[1]
        }
        sol_count = count_solutions_up_to_1000(candidate_puzzle, time_limit=time_limit)
        if sol_count == 1:
            # 已经到唯一解了
            break

        # 使用 CP-SAT 枚举最多 50 个解, 找出所有歧义格
        m_val = len(current_state)
        n_val = len(current_state[0])
        model, x = build_multi_block_cp_model(m_val, n_val,
                                              puzzle_data["row_hints"],
                                              puzzle_data["col_hints"])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        collector = NonogramAllSolutionsCollector(x, max_solutions=50)
        solver.SearchForAllSolutions(model, collector)
        solutions = collector.solutions()

        # 歧义格搜集
        ambiguous_cells = []
        for sol in solutions:
            for i in range(m_val):
                for j in range(n_val):
                    if sol[i][j] != current_state[i][j]:
                        ambiguous_cells.append((i, j))
        if not ambiguous_cells:
            # 当前无歧义格可改，退出
            break

        # 从歧义格中随机选一个
        i_sel, j_sel = random.choice(ambiguous_cells)
        # 判断是否允许翻转
        if not allowed_to_flip(current_state, i_sel, j_sel):
            # 不允许翻转，直接跳过此迭代
            continue

        # 构造候选
        candidate_state = deepcopy(current_state)
        candidate_state[i_sel][j_sel] = 1 - candidate_state[i_sel][j_sel]

        # 翻转后谜题解数
        candidate_puzzle_after = {
            "grid": candidate_state,
            "row_hints": compute_row_col_hints(candidate_state)[0],
            "col_hints": compute_row_col_hints(candidate_state)[1]
        }
        new_sol_count = count_solutions_up_to_1000(candidate_puzzle_after, time_limit=time_limit)

        # 如果解数有降低，则接受
        if new_sol_count < sol_count:
            current_state = candidate_state

    end_time = time.time()

    final_sol_count = count_solutions_up_to_1000({
        "grid": current_state,
        "row_hints": compute_row_col_hints(current_state)[0],
        "col_hints": compute_row_col_hints(current_state)[1]
    }, time_limit=time_limit)

    # 计算最终汉明距离
    diff_count = 0
    for row_orig, row_final in zip(original_grid, current_state):
        for a, b in zip(row_orig, row_final):
            if a != b:
                diff_count += 1

    # 是否成功（final_sol_count == 1）
    success = (final_sol_count == 1)

    result = {
        "final_grid": current_state,
        "iteration_used": iteration,
        "time_spent": end_time - start_time,
        "final_hamming_distance": diff_count,
        "success": success
    }
    return result


############################################################
# 2) 额外的工具函数：把 0/1 grid 保存为图像
############################################################
def save_comparison_image(original_grid, refined_grid, out_img_path):
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


############################################################
# 3) 批量处理函数：读取 CSV，每个 Puzzle 做微调并保存结果
############################################################

def batch_refine_and_save_all(multi_csv="multiple_solutions_details.csv",
                              out_img_dir="./Greedy_refined_images",
                              out_json_dir="./Greedy_refined_jsons",
                              max_iterations=50,
                              time_limit=10.0):
    # 确保输出文件夹存在
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    if not os.path.exists(out_json_dir):
        os.makedirs(out_json_dir)

    # 读取多解 CSV
    df_multi = pd.read_csv(multi_csv)
    # 在输出的结果里，我们会新增一些列，比如 iteration_used, time_spent, final_hamming_distance, success
    new_columns = ["puzzle_id", "num_solutions", "json_file_path",
                   "iteration_used", "time_spent", "final_hamming_distance", "success",
                   "refined_json_path", "refined_img_path"]
    results_list = []

    for idx, row in df_multi.iterrows():
        puzzle_id = row.get("puzzle_id", f"puzzle_{idx}")
        json_path = row.get("json_file_path", None)
        num_sol = row.get("num_solutions", None)

        if not json_path or not os.path.isfile(json_path):
            print(f"[警告] puzzle_id={puzzle_id} 缺少有效 json_file_path, 跳过.")
            continue

        # 读取 JSON puzzle
        with open(json_path, 'r') as f:
            puzzle_data = json.load(f)
        original_grid = deepcopy(puzzle_data["grid"])
        # 调用贪心微调
        result = greedy_refinement(puzzle_data, max_iterations=max_iterations, time_limit=time_limit)

        # 微调后 grid
        refined_grid = result["final_grid"]
        # 保存图片out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        out_img_path = os.path.join(out_img_dir, f"puzzle_{puzzle_id}_comparison.png")
        save_comparison_image(original_grid, refined_grid, out_img_path)

        # 保存新的 JSON
        puzzle_data_refined = {
            "grid": refined_grid,
            "row_hints": compute_row_col_hints(refined_grid)[0],
            "col_hints": compute_row_col_hints(refined_grid)[1]
        }
        refined_json_path = os.path.join(out_json_dir, f"{puzzle_id}_refined.json")
        with open(refined_json_path, "w") as f:
            json.dump(puzzle_data_refined, f)

        # 将关键指标写入
        record = {
            "puzzle_id": puzzle_id,
            "num_solutions": num_sol,
            "json_file_path": json_path,

            "iteration_used": result["iteration_used"],
            "time_spent": result["time_spent"],
            "final_hamming_distance": result["final_hamming_distance"],
            "success": result["success"],

            "refined_json_path": refined_json_path,
            "refined_img_path": out_img_path,
        }
        results_list.append(record)

        print(f"[INFO] puzzle_id={puzzle_id} 完成, success={result['success']}, time={result['time_spent']:.2f}s")

    # 整合保存结果 CSV
    out_csv = "greedy_refinement_results.csv"
    df_out = pd.DataFrame(results_list, columns=new_columns)
    df_out.to_csv(out_csv, index=False)
    print(f"\n所有拼图处理完毕，结果已保存到 {out_csv}")


############################################################
# 4) main 函数
############################################################

def main():
    batch_refine_and_save_all(
        multi_csv="multiple_solutions_details.csv",
        out_img_dir="./Greedy_refined_images",
        out_json_dir="./Greedy_refined_jsons",
        max_iterations=50,  # 可根据需要调整
        time_limit=10.0  # 微调时的最大解算时间
    )


if __name__ == "__main__":
    main()
