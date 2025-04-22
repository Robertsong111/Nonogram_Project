import pandas as pd
import matplotlib.pyplot as plt


def analyze_greedy_results(csv_file="greedy_refinement_results.csv"):
    # 1. 读取数据
    df = pd.read_csv(csv_file)

    # 确保列名都在
    needed_columns = ["puzzle_id", "iteration_used", "time_spent",
                      "final_hamming_distance", "success"]
    for col in needed_columns:
        if col not in df.columns:
            print(f"警告：'{col}' 列不存在，后续统计会有问题。")

    # 2. 统计拼图总数、成功数、成功率
    total_count = len(df)
    if total_count == 0:
        print("无数据可分析.")
        return
    success_count = df["success"].sum()  # True 当成 1
    success_rate = success_count / total_count

    print("=== 基本统计 ===")
    print(f"总拼图数: {total_count}")
    print(f"成功(唯一解)拼图数: {success_count}")
    print(f"成功率: {success_rate:.2%}")

    # 3. 统计迭代次数
    print("\n=== 迭代次数 iteration_used ===")
    avg_iter = df["iteration_used"].mean()
    max_iter = df["iteration_used"].max()
    min_iter = df["iteration_used"].min()
    print(f"平均迭代次数: {avg_iter:.2f}，最大: {max_iter}，最小: {min_iter}")

    # 4. 统计耗时
    print("\n=== 求解耗时 time_spent ===")
    avg_time = df["time_spent"].mean()
    max_time = df["time_spent"].max()
    min_time = df["time_spent"].min()
    print(f"平均耗时: {avg_time:.2f} 秒, 最大: {max_time:.2f} 秒, 最小: {min_time:.2f} 秒")

    # 5. 统计汉明距离
    print("\n=== 最终汉明距离 final_hamming_distance ===")
    avg_hamming = df["final_hamming_distance"].mean()
    max_hamming = df["final_hamming_distance"].max()
    min_hamming = df["final_hamming_distance"].min()
    print(f"平均汉明距离: {avg_hamming:.2f}，最大: {max_hamming}，最小: {min_hamming}")

    # 6. 可视化示例
    # 6.1 迭代次数直方图
    plt.figure()
    df["iteration_used"].hist(bins=20)
    plt.xlabel("Iteration Used")
    plt.ylabel("Frequency")
    plt.title("Greedy Refinement - Iteration Distribution")
    plt.show()

    # 6.2 耗时直方图
    plt.figure()
    df["time_spent"].hist(bins=20)
    plt.xlabel("Time Spent (sec)")
    plt.ylabel("Frequency")
    plt.title("Greedy Refinement - Time Distribution")
    plt.show()

    # 6.3 汉明距离直方图
    plt.figure()
    df["final_hamming_distance"].hist(bins=20)
    plt.xlabel("Final Hamming Distance")
    plt.ylabel("Frequency")
    plt.title("Greedy Refinement - Hamming Distance Distribution")
    plt.show()

    # 6.4 (可选) success 的柱状图
    #    通常 success 是布尔，统计 True/False 各多少
    success_counts = df["success"].value_counts()
    # 可能 success_counts.index = [True,False], success_counts.values = [xxx,yyy]
    plt.figure()
    success_counts.plot(kind="bar")
    plt.xticks(rotation=0)
    plt.xlabel("Success")
    plt.ylabel("Count")
    plt.title("Greedy Refinement - Success/Fail Count")
    plt.show()

    print("\n=== 分析结束 ===")


if __name__ == "__main__":
    analyze_greedy_results("mip_refinement_results.csv")
