import os
import json
from PIL import Image
import torch
from torchvision import datasets, transforms


# ---------------------------
# 1) MNIST 加载
# ---------------------------
def load_mnist_dataset(download=True, root='./mnist_data'):
    """
    加载 MNIST 训练集 (60000 张图).
    """
    mnist_train = datasets.MNIST(
        root=root,
        train=True,
        download=download,
        transform=None  # 先不用任何 transforms
    )
    return mnist_train


# ---------------------------
# 2) 将 MNIST 的图像处理成 Nonogram
# ---------------------------
def mnist_image_to_nonogram(img_pil, final_size=(15,15), threshold=128):
    """
    传入一张 MNIST 的 PIL Image (28x28 灰度)，转成 final_size (15x15) 的 0/1 矩阵 + 行列提示
    """
    # (a) 缩放
    img_resized = img_pil.resize(final_size, Image.Resampling.NEAREST)

    # (b) 转灰度（确保是灰度图）
    gray = img_resized.convert('L')

    # (c) 二值化：这里反转，使得背景为白（255），数字为黑（0）
    bw = gray.point(lambda p: 255 if p < threshold else 0)

    # (d) 转成 0/1 矩阵
    width, height = final_size
    px = bw.load()
    grid = []
    for y in range(height):
        row = []
        for x in range(width):
            val = px[x,y]  # 0 或 255
            # 这里我们希望数字为黑（像素0），映射为 1；白色为0
            cell = 1 if val == 0 else 0
            row.append(cell)
        grid.append(row)

    # (e) 提取行提示
    row_constraints = []
    for r in range(height):
        hint = []
        count = 0
        for c in range(width):
            if grid[r][c] == 1:
                count += 1
            else:
                if count > 0:
                    hint.append(count)
                    count = 0
        if count > 0:
            hint.append(count)
        row_constraints.append(hint)

    # (f) 提取列提示
    col_constraints = []
    for c in range(width):
        hint = []
        count = 0
        for r in range(height):
            if grid[r][c] == 1:
                count += 1
            else:
                if count > 0:
                    hint.append(count)
                    count = 0
        if count > 0:
            hint.append(count)
        col_constraints.append(hint)

    return grid, row_constraints, col_constraints


# ---------------------------
# 新增：将 0/1 矩阵保存为黑白图像
# ---------------------------
def save_grid_as_image(grid, out_path, cell_size=10):
    """
    根据 0/1 矩阵生成黑白图像，并保存。
    grid: 0/1 二维列表，1 表示黑，0 表示白。
    cell_size: 每个谜题单元放大倍数，例如 10，即每个格子 10x10 像素。
    """
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    img = Image.new("L", (width * cell_size, height * cell_size), 255)  # 白底
    pixels = img.load()
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 1:  # 1 表示黑
                for y in range(i*cell_size, (i+1)*cell_size):
                    for x in range(j*cell_size, (j+1)*cell_size):
                        pixels[x, y] = 0  # 黑
    img.save(out_path)


# ---------------------------
# 3) 主函数：批量处理 MNIST 并保存谜题数据和对应的黑白图
# ---------------------------
def bulk_mnist_to_nonogram(
        mnist_dataset,  # 上面的 mnist_train
        num_samples=2000,
        final_size=(15, 15),
        out_dir="./mnist_nonograms",
        out_img_dir="./mnist_nonograms_images",
        threshold=128
):
    """
    从 mnist_dataset 里取前 num_samples 张图像，
    转成 Nonogram puzzle，并分别存储 JSON 文件和对应的黑白图像。
    JSON 文件中保存行列提示，黑白图像直观显示谜题格子。
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    n_total = len(mnist_dataset)
    num_samples = min(num_samples, n_total)
    print(f"Total MNIST samples: {n_total}, we will process: {num_samples}")

    for i in range(num_samples):
        # 获取 MNIST 图像和标签
        img_pil, label = mnist_dataset[i]
        # 转成 Nonogram
        grid, row_cons, col_cons = mnist_image_to_nonogram(img_pil, final_size, threshold)
        puzzle_data = {
            "id": i,
            "label": int(label),
            "grid": grid,
            "row_constraints": row_cons,
            "col_constraints": col_cons
        }
        # 存成 JSON
        out_json = os.path.join(out_dir, f"mnist_puzzle_{i}.json")
        with open(out_json, 'w') as f:
            json.dump(puzzle_data, f)
        # 同时保存黑白图像
        out_img = os.path.join(out_img_dir, f"mnist_puzzle_{i}.png")
        save_grid_as_image(grid, out_img, cell_size=10)

        if i % 100 == 0:
            print(f"Processed {i}/{num_samples}")

    print(f"Done. Puzzles saved in '{out_dir}' and images saved in '{out_img_dir}'.")


# ---------------------------
# 4) 示例调用
# ---------------------------
if __name__=="__main__":
    # a) 加载 MNIST 训练集
    mnist_train = load_mnist_dataset(download=True, root="./mnist_data")
    # b) 只取前 1000 张示例（你可以调整 num_samples 至 60000）
    bulk_mnist_to_nonogram(
        mnist_dataset=mnist_train,
        num_samples=3000,
        final_size=(20,20),   # 可以根据需要调整输出谜题的格子尺寸
        out_dir="./mnist_nonograms",
        out_img_dir="./mnist_nonograms_images",
        threshold=128
    )