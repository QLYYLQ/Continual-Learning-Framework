import random
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

# --- 测试配置 ---
N_LOAD_REPEATS = 10000  # 对单个图像调用 load() 的次数，用于平均其加载速度
N_LIST_PROCESS_REPEATS = 3000  # 处理整个混合图像列表的次数
N_IMAGES_IN_MIXED_LIST = 500  # 混合列表中图像的数量

IMAGE_SIZES_CONFIG = {
    "small": (128, 128),
    "medium": (512, 512),
    "large": (1024, 1024),  # 可根据需要调整，较大的图像会显著增加测试时间
}
IMAGE_FORMATS_CONFIG = ["JPEG", "PNG"]  # 您可以添加更多格式，如 "GIF", "BMP"

# --- 用户 IO 类的占位符 ---
# ⚠️ 重要: 请替换下面的 DummyIO 实现或实例化为您实际的 dataset.io.IO 类。
# 例如:
# from dataset.io import IO # 假设您的类名为 IO
# io_to_test = IO() # 实例化您的类
#
# 或者，如果您的 IO 对象是预先实例化的，并且可以导入：
# from your_project_module import pre_instantiated_io_object
# io_to_test = pre_instantiated_io_object


from dataset.io import IO

# 实例化待测试的 IO 类
# ⚠️ 请务必修改这一行，使用您自己的 IO 类或实例！
io_to_test = IO


# --- 辅助函数 ---
def generate_image_file(
    tmp_path: Path, file_name_prefix: str, size: tuple, img_format: str
) -> Path:
    """生成一个随机图像文件并保存。"""
    img = Image.new(
        "RGB",
        size,
        color=(random.randint(0, 127), random.randint(0, 127), random.randint(0, 127)),
    )
    draw = ImageDraw.Draw(img)
    # 绘制一些随机线条，使图像不那么容易被压缩
    for _ in range(min(size) // 20 + 1):  # 根据图像大小绘制不同数量的线条
        draw.line(
            (
                random.randint(0, size[0] - 1),
                random.randint(0, size[1] - 1),
                random.randint(0, size[0] - 1),
                random.randint(0, size[1] - 1),
            ),
            fill=(
                random.randint(128, 255),
                random.randint(128, 255),
                random.randint(128, 255),
            ),
            width=random.randint(1, 3),
        )

    # 确保父目录存在
    tmp_path.mkdir(parents=True, exist_ok=True)

    file_path = (
        tmp_path / f"{file_name_prefix}_{size[0]}x{size[1]}.{img_format.lower()}"
    )
    try:
        img.save(file_path, img_format)
    except Exception as e:
        pytest.fail(f"无法保存图像 {file_path} (格式: {img_format}): {e}")
    return file_path


def pil_loader(image_path: str):
    """使用 PIL.Image.open 加载图像并确保数据已读入内存。"""
    try:
        img = Image.open(str(image_path))
        return img
    except Exception as e:
        print(f"错误发生在 pil_loader({image_path}): {e}")
        return None


# --- Pytest 测试用例 ---


@pytest.mark.parametrize("size_name, dimensions", IMAGE_SIZES_CONFIG.items())
@pytest.mark.parametrize("img_format", IMAGE_FORMATS_CONFIG)
def test_individual_image_loading_speed(
    tmp_path: Path, size_name: str, dimensions: tuple, img_format: str
):
    """
    测试加载具有特定尺寸和格式的单个图像多次。
    比较 io_to_test.load() 与 PIL.Image.open() 的平均加载时间。
    """
    image_file = generate_image_file(
        tmp_path, f"img_{size_name}", dimensions, img_format
    )
    print(
        f"\n[测试单张图片] 尺寸: {size_name} ({dimensions[0]}x{dimensions[1]}), 格式: {img_format}, 路径: {image_file}"
    )

    # 测试 io_to_test.load 的速度
    io_load_times = []
    for i in range(N_LOAD_REPEATS):
        start_time = time.perf_counter()
        img1 = io_to_test.load(str(image_file))
        end_time = time.perf_counter()
        assert img1 is not None, f"io_to_test.load({image_file}) 第 {i+1} 次加载失败"
        io_load_times.append(end_time - start_time)
    avg_io_load_time = sum(io_load_times) / N_LOAD_REPEATS

    # 测试 PIL.Image.open 的速度
    pil_load_times = []
    for i in range(N_LOAD_REPEATS):
        start_time = time.perf_counter()
        img2 = pil_loader(str(image_file))
        end_time = time.perf_counter()
        assert img2 is not None, f"pil_loader({image_file}) 第 {i+1} 次加载失败"
        pil_load_times.append(end_time - start_time)
    avg_pil_load_time = sum(pil_load_times) / N_LOAD_REPEATS

    print(f"  平均 io_to_test.load 时间: {avg_io_load_time:.6f} 秒")
    print(f"  平均 PIL.Image.open 时间: {avg_pil_load_time:.6f} 秒")
    difference = avg_io_load_time - avg_pil_load_time
    print(
        f"  时间差 (io_to_test - PIL): {difference:.6f} 秒 ({'+' if difference > 0 else ''}{difference/avg_pil_load_time*100 if avg_pil_load_time else float('inf'):.2f}%)"
    )
    # 可选：如果对其性能有预期，可以添加断言
    # assert avg_io_load_time < avg_pil_load_time * 1.1, "io_to_test.load 比 PIL 慢了10%以上"


def test_mixed_size_image_loading_speed(tmp_path: Path):
    """
    测试加载包含混合尺寸图像的列表。
    整个列表会加载多次，以平均总处理时间。
    """
    print(
        f"\n[测试混合尺寸图片] 列表包含 {N_IMAGES_IN_MIXED_LIST} 张图片, 处理 {N_LIST_PROCESS_REPEATS} 次"
    )
    image_files = []
    size_names = list(IMAGE_SIZES_CONFIG.keys())
    for i in range(N_IMAGES_IN_MIXED_LIST):
        size_name = size_names[i % len(size_names)]  # 循环使用定义的尺寸
        dimensions = IMAGE_SIZES_CONFIG[size_name]
        # 在此混合尺寸测试中，统一使用 JPEG 格式以减少变量
        img_format = "JPEG"
        image_files.append(
            generate_image_file(tmp_path, f"mixed_size_img_{i}", dimensions, img_format)
        )
    print(f"  生成图片: {[str(f.name) for f in image_files]}")

    # 测试 io_to_test.load 处理列表的速度
    total_io_processing_times = []
    for i in range(N_LIST_PROCESS_REPEATS):
        run_start_time = time.perf_counter()
        for image_file in image_files:
            img = io_to_test.load(str(image_file))
            assert (
                img is not None
            ), f"io_to_test.load({image_file}) 在混合尺寸测试的第 {i+1} 轮中失败"
        run_end_time = time.perf_counter()
        total_io_processing_times.append(run_end_time - run_start_time)
    avg_total_io_time = sum(total_io_processing_times) / N_LIST_PROCESS_REPEATS

    # 测试 PIL.Image.open 处理列表的速度
    total_pil_processing_times = []
    for i in range(N_LIST_PROCESS_REPEATS):
        run_start_time = time.perf_counter()
        for image_file in image_files:
            img = pil_loader(str(image_file))
            assert img is not None, f"pil_loader({image_file}) 在混合尺寸测试的第 {i+1} 轮中失败"
        run_end_time = time.perf_counter()
        total_pil_processing_times.append(run_end_time - run_start_time)
    avg_total_pil_time = sum(total_pil_processing_times) / N_LIST_PROCESS_REPEATS

    print(f"  混合尺寸列表 - 平均总时间 (io_to_test.load): {avg_total_io_time:.6f} 秒")
    print(f"  混合尺寸列表 - 平均总时间 (PIL.Image.open): {avg_total_pil_time:.6f} 秒")
    difference = avg_total_io_time - avg_total_pil_time
    print(
        f"  时间差 (io_to_test - PIL): {difference:.6f} 秒 ({'+' if difference > 0 else ''}{difference/avg_total_pil_time*100 if avg_total_pil_time else float('inf'):.2f}%)"
    )


def test_mixed_format_image_loading_speed(tmp_path: Path):
    """
    测试加载包含混合格式图像的列表。
    整个列表会加载多次，以平均总处理时间。
    """
    print(
        f"\n[测试混合格式图片] 列表包含 {N_IMAGES_IN_MIXED_LIST} 张图片, 处理 {N_LIST_PROCESS_REPEATS} 次"
    )
    image_files = []
    # 在此测试中，所有图像统一使用中等尺寸，仅改变格式
    dimensions = IMAGE_SIZES_CONFIG["medium"]
    for i in range(N_IMAGES_IN_MIXED_LIST):
        img_format = IMAGE_FORMATS_CONFIG[i % len(IMAGE_FORMATS_CONFIG)]  # 循环使用定义的格式
        image_files.append(
            generate_image_file(
                tmp_path, f"mixed_format_img_{i}", dimensions, img_format
            )
        )
    print(f"  生成图片: {[str(f.name) for f in image_files]}")

    # 测试 io_to_test.load 处理列表的速度
    total_io_processing_times = []
    for i in range(N_LIST_PROCESS_REPEATS):
        run_start_time = time.perf_counter()
        for image_file in image_files:
            img = io_to_test.load(str(image_file))
            assert (
                img is not None
            ), f"io_to_test.load({image_file}) 在混合格式测试的第 {i+1} 轮中失败"
        run_end_time = time.perf_counter()
        total_io_processing_times.append(run_end_time - run_start_time)
    avg_total_io_time = sum(total_io_processing_times) / N_LIST_PROCESS_REPEATS

    # 测试 PIL.Image.open 处理列表的速度
    total_pil_processing_times = []
    for i in range(N_LIST_PROCESS_REPEATS):
        run_start_time = time.perf_counter()
        for image_file in image_files:
            img = pil_loader(str(image_file))
            assert img is not None, f"pil_loader({image_file}) 在混合格式测试的第 {i+1} 轮中失败"
        run_end_time = time.perf_counter()
        total_pil_processing_times.append(run_end_time - run_start_time)
    avg_total_pil_time = sum(total_pil_processing_times) / N_LIST_PROCESS_REPEATS

    print(f"  混合格式列表 - 平均总时间 (io_to_test.load): {avg_total_io_time:.6f} 秒")
    print(f"  混合格式列表 - 平均总时间 (PIL.Image.open): {avg_total_pil_time:.6f} 秒")
    difference = avg_total_io_time - avg_total_pil_time
    print(
        f"  时间差 (io_to_test - PIL): {difference:.6f} 秒 ({'+' if difference > 0 else ''}{difference/avg_total_pil_time*100 if avg_total_pil_time else float('inf'):.2f}%)"
    )
