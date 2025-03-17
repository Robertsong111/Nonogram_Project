import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

def visualize_blackwhite(matrix_2d, ax, title=""):
    """
    用 matplotlib 在 ax 上显示一个 0/1 矩阵的黑白图。
    0 -> 白, 1 -> 黑 (使用 'binary' colormap).
    """
    ax.imshow(matrix_2d, cmap='binary', interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')  # 隐藏坐标轴

def analyze_solution_differences(csv_file):
    """
    1) 读取 all_solutions_record.csv 并对 difference_count 做整体统计和直方图
    2) 分 puzzle_id 统计 min/max/mean/count 并输出 puzzle_diff_stats.csv
    3) 导出每个 puzzle 的最贴近原图解(closest_solutions_per_puzzle.csv)
    4) 随机挑选 5 个 puzzle:
       - 忽略 difference_count=0 的解(只看>0)
       - 最接近原图/最远离原图
       - 如果 puzzle JSON 中有 "grid", 就展示三幅图(原图+最接近+最远), 否则只展示两幅图
    """

    df = pd.read_csv(csv_file)

    # 检查必要的列
    if "difference_count" not in df.columns:
        print("Error: CSV 中没有 'difference_count' 列，无法统计差异。")
        return
    if "solution_matrix" not in df.columns:
        print("Error: CSV 中没有 'solution_matrix' 列，无法可视化解的黑白格。")
        return
    if "puzzle_id" not in df.columns:
        print("Error: CSV 中没有 'puzzle_id' 列，无法分 puzzle 统计。")
        return

    # ---------------------------
    # A. 整体分布统计
    # ---------------------------
    print("=== difference_count 基本统计 ===")
    print(df["difference_count"].describe())

    plt.figure()
    df["difference_count"].hist(bins=30)
    plt.title("Distribution of difference_count (All Solutions)")
    plt.xlabel("difference_count")
    plt.ylabel("Frequency")
    plt.show()

    threshold = 10
    count_leq_thr = (df["difference_count"] <= threshold).sum()
    total_solutions = len(df)
    print(f"\n差异 <= {threshold} 的解数: {count_leq_thr}/{total_solutions} ({count_leq_thr/total_solutions:.2%})")

    # ---------------------------
    # B. 分 puzzle_id 统计 min / max / mean / count
    # ---------------------------
    grouped = df.groupby("puzzle_id")["difference_count"]
    stats_df = grouped.agg(["min", "max", "mean", "count"]).rename(
        columns={"min": "min_diff", "max": "max_diff", "mean": "avg_diff", "count": "num_solutions"}
    )
    stats_df.to_csv("puzzle_diff_stats.csv")
    print("\n=== 每个 puzzle 的差异统计 已输出到 puzzle_diff_stats.csv ===")
    print(stats_df.head(10))

    plt.figure()
    stats_df["min_diff"].hist(bins=30)
    plt.title("Distribution of Min Difference per Puzzle")
    plt.xlabel("min_diff")
    plt.ylabel("Frequency")
    plt.show()

    # ---------------------------
    # C. 每个 puzzle 最贴近原图的解 (difference_count 最小)
    # ---------------------------
    idxmin_series = df.groupby("puzzle_id")["difference_count"].idxmin()
    closest_solutions = df.loc[idxmin_series].copy()
    closest_solutions.to_csv("closest_solutions_per_puzzle.csv", index=False)
    print("\n=== 每个 puzzle 与原图最接近的解 已输出到 closest_solutions_per_puzzle.csv ===")

    # ---------------------------
    # D. 随机挑选 5 个 puzzle, 展示 "原图 + 最接近(>0) + 最远(>0)"
    # ---------------------------
    puzzle_ids = df["puzzle_id"].unique()
    if len(puzzle_ids) == 0:
        print("\n没有可用 puzzle，无法演示对比。")
        return

    # 若 puzzle 不到5个，就只展示 puzzle_ids.size个
    num_to_display = min(5, len(puzzle_ids))
    sampled_puzzle_ids = pd.Series(puzzle_ids).sample(n=num_to_display, random_state=42).values

    for pid in sampled_puzzle_ids:
        subset = df[df["puzzle_id"] == pid]
        if subset.empty:
            continue

        # 在该 puzzle 的解中，忽略 difference_count=0
        subset_nonzero = subset[subset["difference_count"] > 0].copy()
        if subset_nonzero.empty:
            print(f"\nPuzzle {pid}: 全部解都与原图完全相同(或无解)，跳过展示.")
            continue

        # 找 difference_count 的最小/最大
        min_idx = subset_nonzero["difference_count"].idxmin()
        max_idx = subset_nonzero["difference_count"].idxmax()

        row_min = subset_nonzero.loc[min_idx]
        row_max = subset_nonzero.loc[max_idx]

        print(f"\nPuzzle {pid}: (仅考虑 difference_count>0 的解)")
        print(f"  最接近原图 (diff={row_min['difference_count']}, solution_idx={row_min.get('solution_idx', None)})")
        print(f"  最远离原图 (diff={row_max['difference_count']}, solution_idx={row_max.get('solution_idx', None)})")

        # 反序列化
        try:
            min_mat = json.loads(row_min["solution_matrix"])
            max_mat = json.loads(row_max["solution_matrix"])
        except Exception as e:
            print("反序列化矩阵失败:", e)
            continue

        # 尝试加载 puzzle JSON 并查看 puzzle_data["grid"]
        puzzle_json_path = row_min["json_file_path"]
        original_grid = None
        if os.path.isfile(puzzle_json_path):
            with open(puzzle_json_path, 'r') as f:
                puzzle_data = json.load(f)
            original_grid = puzzle_data.get("grid", None)

        # 若有原图则三图显示，否则两图
        if (original_grid is not None
            and len(original_grid) > 0
            and len(original_grid[0]) > 0):
            fig, ax = plt.subplots(1, 3, figsize=(12,4))
            # (1) 原图
            visualize_blackwhite(
                original_grid,
                ax[0],
                title=f"Puzzle {pid}\nOriginal"
            )
            # (2) 最接近
            visualize_blackwhite(
                min_mat,
                ax[1],
                title=f"Min diff={row_min['difference_count']}"
            )
            # (3) 最远
            visualize_blackwhite(
                max_mat,
                ax[2],
                title=f"Max diff={row_max['difference_count']}"
            )
        else:
            # 不存在原图 => 只展示两幅
            fig, ax = plt.subplots(1, 2, figsize=(8,4))
            visualize_blackwhite(
                min_mat,
                ax[0],
                title=f"Puzzle {pid}\nMin diff={row_min['difference_count']}"
            )
            visualize_blackwhite(
                max_mat,
                ax[1],
                title=f"Puzzle {pid}\nMax diff={row_max['difference_count']}"
            )

        plt.tight_layout()
        plt.show()

def main():
    csv_file = "all_solutions_record.csv"
    analyze_solution_differences(csv_file)

if __name__ == "__main__":
    main()
