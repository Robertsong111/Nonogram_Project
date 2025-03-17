#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob

# 假设你已将之前的“随机翻转微调”逻辑放到 puzzle_refinement.py 里
# 并且里面提供了 local_search_refinement, solve_puzzle_num_solutions 函数。
# 注意根据你的实际模块位置做导入
from 普通贪心微调 import local_search_refinement, solve_puzzle_num_solutions


def batch_local_search_demo(input_dir, output_dir, pattern="*.json"):
    """
    批量读取 input_dir 下符合 pattern 的 JSON 拼图文件，
    对每个拼图做“随机翻转+贪心”的微调，并将结果输出到 output_dir 下，
    文件名改为 原名 + '_refined.json'。
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 找到所有要处理的 JSON 文件
    puzzle_files = glob.glob(os.path.join(input_dir, pattern))
    if not puzzle_files:
        print(f"在 {input_dir} 下未找到任何 {pattern} 文件。")
        return

    for puzzle_file in puzzle_files:
        with open(puzzle_file, 'r') as f:
            puzzle_data = json.load(f)

        puzzle_name = os.path.splitext(os.path.basename(puzzle_file))[0]  # 不带后缀的文件名
        print(f"\n=== 处理 {puzzle_name} ===")

        # 1) 微调前先测一下解数（这里我们 max_solutions=2，只要判断是否多解即可）
        sol_before = solve_puzzle_num_solutions(puzzle_data, time_limit=5.0, max_solutions=2)
        print(f"微调前解数(0=无解,1=唯一解,>=2=多解): {sol_before}")

        # 2) 执行微调（最多 50 次翻转）
        refined_data = local_search_refinement(
            puzzle_data,
            max_iterations=50,
            time_limit=5.0
        )

        # 3) 微调后再测一下解数
        sol_after = solve_puzzle_num_solutions(refined_data, time_limit=5.0, max_solutions=2)
        print(f"微调后解数: {sol_after}")

        # 4) 将 refined_data 另存为新文件
        out_path = os.path.join(output_dir, puzzle_name + "_refined.json")
        with open(out_path, 'w') as f:
            json.dump(refined_data, f)
        print(f"已输出: {out_path}")


def main():
    # 示例：我们假设有一个文件夹 "./test_puzzles" 存放要处理的 JSON
    # 输出到 "./test_puzzles_refined"
    input_dir = "./mnist_nonograms"
    output_dir = "./test_puzzles"

    batch_local_search_demo(input_dir, output_dir, pattern="*.json")


if __name__ == "__main__":
    main()
