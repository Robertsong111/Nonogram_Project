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
#若某段提示的可行区间比提示长度只大一点，那么它们在区间内必定会有一段重叠区，这部分重叠区域必定是黑格。
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
#若某些列在所有提示段区间之外，则这些列不可能是黑格，直接标记为白格 X。
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
#当在某段区间内已经出现了足够长度的连续黑格并与该段长度相符，那么该段两端相邻的未知位置必然是白格。
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
#若行内出现「# . #」的结构，且这几个连续 # 的总长度已经超过所有提示的最大值，则夹在中间的那个 . 不可能是黑格，必为 X。
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
#对于每个尚未确定的格子，如果它在所有提示段的区间中都无合适位置（即该 . 不可能被任何提示段覆盖），则该格必为白格 X。
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
#对相邻两个提示段的区间进行「相互挤压」。因为每段之间至少留出 1 格空白，若前面一段占用的区间往右推，就会推动后面一段的起点也向右移动，反之亦然。
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
#若在行中已经有确定的 # 出现，那么对应的段区间不能与这些 # 冲突，需要往内收缩。
def apply_rule_2_2(row, hints, run_ranges):
    n = len(row)
    for idx, (start, end) in enumerate(run_ranges):
        while start - 1 >= 0 and row[start - 1] == '#':
            start += 1
        while end + 1 < n and row[end + 1] == '#':
            end -= 1
        run_ranges[idx] = (start, end)
#如果实际出现了一段比所有提示都大的连续 # 块，这显然与提示冲突，这块不可能是黑格，应视为无效，直接填 X。
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
#如果某段已经出现了部分连续 #，并且这些 # 完全处在某个提示段区间中，而且不超过该段长度，那么那段之间的 .（即连接这些 # 的空位）也必是 #。
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
#如果行中实际出现一段连续 # 的长度比任意提示段的最小值还要小，那它不可能是真正的黑块，应把它们改成白格。
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
#若某行出现了一段较长的 # 并且它位于某一提示段的 (rstart, rend) 内，但却比该提示长度大，那么说明这段提示的范围实际上要缩小。
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

    start_time = time.time()

    solutions = solve_nonogram(row_constraints, col_constraints)

    end_time = time.time()

    print(f"共找到 {len(solutions)} 个解")
    print(f"程序运行时间: {end_time - start_time:.6f} 秒")

    for sol in solutions:
        img = [[1 if ch == '#' else 0 for ch in row] for row in sol]
        plt.imshow(img, cmap="gray_r", interpolation="none")
        plt.axis('off')
        plt.show()
