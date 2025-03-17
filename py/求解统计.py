import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_nonogram_results(
    csv_file,
    output_multi_csv="multiple_solutions_details.csv",
    puzzle_dir="./mnist_nonograms"   # 假设你把 JSON 文件都放在这个文件夹
):
    """
    读取 CP-SAT 求解结果的 csv_file（如 cp_sat_solver_results_seq.csv），
    分析无解/单解/多解的分布情况，
    绘制可视化图表，并将多解的拼图信息导出到 output_multi_csv。
    同时，若 'json_file_path' 列不存在，则自动通过 puzzle_id 在 puzzle_dir 中推断出 JSON 文件路径。
    """

    # 1) 读取 CSV
    df = pd.read_csv(csv_file)

    # ---------------------------
    # 1.1 统计无解、单解、多解的数量
    # ---------------------------
    count_no_solution = (df['num_solutions'] == 0).sum()
    count_one_solution = (df['num_solutions'] == 1).sum()
    count_multiple_solutions = (df['num_solutions'] > 1).sum()

    total = len(df)
    print("=== 基本解数统计 ===")
    print(f"总谜题数: {total}")
    print(f"无解 (num_solutions=0): {count_no_solution}")
    print(f"唯一解 (num_solutions=1): {count_one_solution}")
    print(f"多解 (num_solutions>1): {count_multiple_solutions}")

    # 也可查看比例:
    print("\n=== 解数分布(比例) ===")
    print(f"无解占比: {count_no_solution / total:.2%}")
    print(f"唯一解占比: {count_one_solution / total:.2%}")
    print(f"多解占比: {count_multiple_solutions / total:.2%}")

    # ---------------------------
    # 1.2 用柱状图可视化无解/单解/多解分布
    # ---------------------------
    plt.figure()
    labels = ['No Solution', '1 Solution', 'Multiple Solutions']
    counts = [count_no_solution, count_one_solution, count_multiple_solutions]
    plt.bar(labels, counts)
    plt.title("Distribution of Solutions Count")
    plt.xlabel("Solution Category")
    plt.ylabel("Number of Puzzles")
    plt.show()

    # ---------------------------
    # 2. 分析求解时长
    # ---------------------------
    avg_time = df['solving_time_sec'].mean()
    max_time = df['solving_time_sec'].max()
    min_time = df['solving_time_sec'].min()

    print("\n=== 求解时间统计 ===")
    print(f"平均求解时间: {avg_time:.4f} 秒")
    print(f"最大求解时间: {max_time:.4f} 秒")
    print(f"最小求解时间: {min_time:.4f} 秒")

    # 绘制求解时间直方图
    plt.figure()
    df['solving_time_sec'].hist(bins=30)  # 你可以调整 bins 数目
    plt.title("Histogram of Solving Time")
    plt.xlabel("Solving Time (seconds)")
    plt.ylabel("Frequency")
    plt.show()

    # ---------------------------
    # 3. (可选) 分析不同 label 的多解情况
    # ---------------------------
    if 'label' in df.columns and df['label'].notna().any():
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df_label = df.dropna(subset=['label'])

        label_stats = []
        for digit in range(10):
            subset = df_label[df_label['label'] == digit]
            if len(subset) == 0:
                continue
            count_total_digit = len(subset)
            count_multi_digit = (subset['num_solutions'] > 1).sum()
            multi_rate = count_multi_digit / count_total_digit
            label_stats.append((digit, count_total_digit, count_multi_digit, multi_rate))

        print("\n=== 不同数字标签的多解统计 ===")
        for digit, total_cnt, multi_cnt, multi_rate in label_stats:
            print(f"数字 {digit}: 总数={total_cnt}, 多解数={multi_cnt}, 多解率={multi_rate:.2%}")

        # 柱状图可视化
        digits = [s[0] for s in label_stats]
        multi_rates = [s[3] for s in label_stats]

        plt.figure()
        plt.bar(digits, multi_rates)
        plt.title("Multi-Solution Rate per MNIST Digit")
        plt.xlabel("Digit Label")
        plt.ylabel("Multi-Solution Rate")
        plt.show()

    # ---------------------------
    # 4. 导出多解的详细信息 (含 json_file_path)
    # ---------------------------
    df_multi = df[df['num_solutions'] > 1].copy()
    if not df_multi.empty:
        # 如果原始 df 中不含 'json_file_path' 列，我们就基于 puzzle_id 推断
        if 'json_file_path' not in df_multi.columns:
            if 'puzzle_id' in df_multi.columns:
                # 假设 puzzle_id 用数字表示，如 123 -> "./mnist_nonograms/mnist_puzzle_123.json"
                def build_json_path(pid):
                    return os.path.join(puzzle_dir, f"mnist_puzzle_{pid}.json")
                df_multi['json_file_path'] = df_multi['puzzle_id'].apply(build_json_path)
            else:
                print("警告: 无法生成 'json_file_path' (缺少 puzzle_id)，请确保原始数据有 puzzle_id 字段.")

        # 在最终导出的列中，加入 json_file_path
        columns_to_save = ['puzzle_id', 'label', 'row_hints', 'col_hints', 'num_solutions', 'json_file_path']
        # 如果某些列不存在，就先过滤
        columns_exist = [col for col in columns_to_save if col in df_multi.columns]
        df_export = df_multi[columns_exist]

        df_export.to_csv(output_multi_csv, index=False)
        print(f"\n已将多解拼图信息输出至: {output_multi_csv}")
        print("该文件中包含 json_file_path，用于后续歧义格定位、微调等操作。")
    else:
        print("\n没有多解的记录，因此无需导出多解信息.")


if __name__ == "__main__":
    # 示例： 指定输入CSV, 输出多解详情, 并告知存放Nonogram JSON的文件夹 puzzle_dir
    analyze_nonogram_results(
        csv_file="cp_sat_solver_results_seq.csv",
        output_multi_csv="multiple_solutions_details.csv",
        puzzle_dir="./mnist_nonograms"
    )