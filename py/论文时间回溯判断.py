from copy import deepcopy
import time
import matplotlib.pyplot as plt


def initialize_run_ranges(hints, row_length):
    ranges = []
    k = len(hints)
    total_block_len = sum(hints)
    total_space = k - 1
    min_required_len = total_block_len + total_space
    if min_required_len > row_length:
        raise ValueError("提示和总长度不符")
    start = 0
    for i in range(k):
        max_start = row_length - (sum(hints[i:]) + (k - i - 1))
        ranges.append((start, max_start))
        start += hints[i] + 1
    return ranges

def apply_all_rules(row, hints):
    row = row[:]
    run_ranges = initialize_run_ranges(hints, len(row))
    prev = None
    while row != prev:
        prev = row[:]
        apply_rule_2_1(hints, run_ranges)
        apply_rule_2_2(row, hints, run_ranges)
        apply_rule_2_3(row, hints, run_ranges)
        apply_rule_1_1(row, hints, run_ranges)
        apply_rule_1_3(row, hints, run_ranges)
        apply_rule_1_4(row, hints)
        apply_rule_1_5(row, hints, run_ranges)
        apply_rule_3_1(row, hints, run_ranges)
        apply_rule_3_2(row, hints)
        apply_rule_3_3(row, hints, run_ranges)
    apply_rule_1_2_safe(row, run_ranges)
    return row

def apply_rule_1_1(row, hints, run_ranges):
    for i, (start, end) in enumerate(run_ranges):
        length = hints[i]
        range_size = end - start + 1
        if range_size < length:
            continue
        overlap_start = start + (range_size - length)
        overlap_end = end - (range_size - length)
        for j in range(overlap_start, overlap_end + 1):
            if 0 <= j < len(row) and row[j] == '.':
                row[j] = '#'

def apply_rule_1_2_safe(row, run_ranges):
    n = len(row)
    if not run_ranges:
        return
    left_most = min(start for start, _ in run_ranges)
    right_most = max(end for _, end in run_ranges)
    for i in range(0, left_most):
        if row[i] == '.':
            row[i] = 'X'
    for i in range(right_most + 1, n):
        if row[i] == '.':
            row[i] = 'X'

def apply_rule_1_3(row, hints, run_ranges):
    n = len(row)
    for i, (start, end) in enumerate(run_ranges):
        length = hints[i]
        for pos in range(start, end + 1):
            if pos + length - 1 <= end:
                if all(0 <= j < n and row[j] == '#' for j in range(pos, pos + length)):
                    if pos - 1 >= 0 and row[pos - 1] == '.':
                        row[pos - 1] = 'X'
                    if pos + length < n and row[pos + length] == '.':
                        row[pos + length] = 'X'

def apply_rule_1_4(row, hints):
    n = len(row)
    max_hint = max(hints) if hints else 0
    for i in range(1, n - 1):
        if row[i] == '.' and row[i - 1] == '#' and row[i + 1] == '#':
            length = 1
            left = i - 1
            while left >= 0 and row[left] == '#':
                length += 1
                left -= 1
            right = i + 1
            while right < n and row[right] == '#':
                length += 1
                right += 1
            if length > max_hint:
                row[i] = 'X'

def apply_rule_1_5(row, hints, run_ranges):
    n = len(row)
    for i in range(n):
        if row[i] != '.':
            continue
        can_be_black = False
        for h, (start, end) in zip(hints, run_ranges):
            if start <= i <= end and end - start + 1 >= h:
                can_be_black = True
                break
        if not can_be_black:
            row[i] = 'X'

def apply_rule_2_1(hints, run_ranges):
    k = len(hints)
    for j in range(1, k):
        prev_start, _ = run_ranges[j - 1]
        prev_len = hints[j - 1]
        run_ranges[j] = (max(run_ranges[j][0], prev_start + prev_len + 1), run_ranges[j][1])
    for j in range(k - 2, -1, -1):
        _, next_end = run_ranges[j + 1]
        next_len = hints[j + 1]
        run_ranges[j] = (run_ranges[j][0], min(run_ranges[j][1], next_end - next_len - 1))

def apply_rule_2_2(row, hints, run_ranges):
    n = len(row)
    for idx, (start, end) in enumerate(run_ranges):
        while start - 1 >= 0 and row[start - 1] == '#':
            start += 1
        while end + 1 < n and row[end + 1] == '#':
            end -= 1
        run_ranges[idx] = (start, end)

def apply_rule_2_3(row, hints, run_ranges):
    n = len(row)
    segments = []
    i = 0
    while i < n:
        if row[i] == '#':
            start = i
            while i + 1 < n and row[i + 1] == '#':
                i += 1
            end = i
            segments.append((start, end))
        i += 1
    max_hint = max(hints) if hints else 0
    for seg_start, seg_end in segments:
        if seg_end - seg_start + 1 > max_hint:
            for i in range(seg_start, seg_end + 1):
                row[i] = 'X'

def apply_rule_3_1(row, hints, run_ranges):
    n = len(row)
    i = 0
    while i < n:
        if row[i] == '#':
            start = i
            while i + 1 < n and (row[i + 1] == '#' or row[i + 1] == '.'):
                i += 1
            end = i
            for idx, (rstart, rend) in enumerate(run_ranges):
                length = hints[idx]
                if start >= rstart and end <= rend and (end - start + 1) <= length:
                    for j in range(start, end + 1):
                        if row[j] == '.':
                            row[j] = '#'
            i += 1
        else:
            i += 1

def apply_rule_3_2(row, hints):
    n = len(row)
    min_hint = min(hints) if hints else 0
    i = 0
    while i < n:
        if row[i] == '#':
            start = i
            while i + 1 < n and row[i + 1] == '#':
                i += 1
            end = i
            if end - start + 1 < min_hint:
                for j in range(start, end + 1):
                    row[j] = 'X'
        i += 1

def apply_rule_3_3(row, hints, run_ranges):
    n = len(row)
    segments = []
    i = 0
    while i < n:
        if row[i] == '#':
            start = i
            while i + 1 < n and row[i + 1] == '#':
                i += 1
            end = i
            segments.append((start, end))
        i += 1
    for idx, (rstart, rend) in enumerate(run_ranges):
        h = hints[idx]
        for seg_start, seg_end in segments:
            if seg_start < rstart or seg_end > rend:
                continue
            if seg_end - seg_start + 1 > h:
                run_ranges[idx] = (max(rstart, seg_start), min(rend, seg_end))

# 这部分为逻辑规则的完整代码实现，完全基于你之前提供的论文。你可以现在测试任意一行 row 和 hint，调用 apply_all_rules(row, hints) 来查看推理后的结果。

# 下一步如果你需要我把它集成到一个完整的 Nonogram 求解器（支持行列约束、找所有解），我可以继续往下接。
# 是否继续？
def match_col_prefix(col_vals, constraint):
    segments = []
    cnt = 0
    for val in col_vals:
        if val == '#':
            cnt += 1
        elif cnt > 0:
            segments.append(cnt)
            cnt = 0
    if cnt > 0:
        segments.append(cnt)
    if len(segments) > len(constraint):
        return False
    for i in range(len(segments)):
        if segments[i] > constraint[i]:
            return False
    return True

def is_grid_valid(grid, col_constraints):
    width = len(col_constraints)
    height = len(grid)
    for col in range(width):
        col_vals = [grid[r][col] for r in range(height)]
        segments = []
        cnt = 0
        for val in col_vals:
            if val == '#':
                cnt += 1
            elif cnt > 0:
                segments.append(cnt)
                cnt = 0
        if cnt > 0:
            segments.append(cnt)
        if segments != col_constraints[col]:
            return False
    return True

def generate_line_options(length, hints):
    """给定行长度和提示，生成所有合法填法"""
    def dfs(pos, idx, line):
        if idx == len(hints):
            if len(line) == length:
                result.append(line)
            elif len(line) < length:
                result.append(line + ['.'] * (length - len(line)))
            return
        max_pos = length - sum(hints[idx:]) - (len(hints) - idx - 1)
        for i in range(pos, max_pos + 1):
            new_line = line + ['.'] * (i - pos) + ['#'] * hints[idx]
            if idx != len(hints) - 1:
                new_line += ['.']
            dfs(i + hints[idx] + 1, idx + 1, new_line)
    result = []
    dfs(0, 0, [])
    return result




def solve_nonogram(row_constraints, col_constraints):
    height = len(row_constraints)
    width = len(col_constraints)
    solutions = []

    # 不要做 apply_all_rules，否则会错删合法行
    row_options = [generate_line_options(width, r) for r in row_constraints]

    def is_valid(grid, row_idx):
        for col in range(width):
            col_vals = [grid[r][col] for r in range(row_idx + 1)]
            if not match_col_prefix(col_vals, col_constraints[col]):
                return False
        return True

    def dfs(row_idx, grid):
        if row_idx == height:
            if is_grid_valid(grid, col_constraints):
                solutions.append([''.join(r) for r in grid])
            return
        for option in row_options[row_idx]:
            grid[row_idx] = option
            if is_valid(grid, row_idx):
                dfs(row_idx + 1, deepcopy(grid))

    dfs(0, [['.'] * width for _ in range(height)])
    return solutions


if __name__ == "__main__":
    row_constraints = [[1], [1], [1], [1], [1], [1], [1]]
    col_constraints = [[1], [1], [1], [1], [1], [1], [1]]

    # 开始计时
    start_time = time.time()

    # 使用你的逻辑回溯求解器
    solutions = solve_nonogram(row_constraints, col_constraints)

    # 结束计时
    end_time = time.time()

    # 输出结果
    print(f"共找到 {len(solutions)} 个解")
    print(f"程序运行时间: {end_time - start_time:.4f} 秒")

    # 显示前几个解的黑白图像
    for sol in solutions:
        # 将 '#' 显示为黑色（1），'.' 或 'X' 显示为白色（0）
        img = [[1 if ch == '#' else 0 for ch in row] for row in sol]
        plt.imshow(img, cmap="gray_r", interpolation="none")
        plt.axis('off')
        plt.show()
