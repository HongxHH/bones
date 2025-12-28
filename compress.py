from PIL import Image
import os
from io import BytesIO


def compress_image_to_target_size(
        input_img_path: str,
        output_img_path: str,
        target_size_kb: float,
        min_quality: int = 10,
        quality_step: int = 5,
        size_tolerance: float = 0.1  # 目标大小的±10%视为符合要求
) -> tuple[bool, str, float]:
    """
    将指定图像压缩到目标大小附近（优先调整质量，其次缩放分辨率）

    参数:
        input_img_path: 输入图像的路径（支持JPG/PNG等常见格式）
        output_img_path: 压缩后图像的保存路径（建议后缀为.jpg，质量调整对JPG效果更显著）
        target_size_kb: 目标文件大小（单位：KB）
        min_quality: 最低图像质量（1-100，越低压缩率越高，画质越差）
        quality_step: 质量调整的步长（每次调整减少的质量值）
        size_tolerance: 大小容差范围，默认±10%

    返回:
        (是否成功, 提示信息, 最终文件大小KB)
    """
    # 合法性校验
    if not os.path.exists(input_img_path):
        return False, "输入图像文件不存在", 0.0
    if target_size_kb <= 0:
        return False, "目标大小必须大于0", 0.0
    if not (1 <= min_quality <= 100):
        return False, "最低质量必须在1-100之间", 0.0
    target_size_bytes = target_size_kb * 1024  # 转换为字节（1KB=1024字节）
    lower_bound = target_size_bytes * (1 - size_tolerance)
    upper_bound = target_size_bytes * (1 + size_tolerance)

    try:
        # 1. 打开原始图像，保留原始格式和宽高
        with Image.open(input_img_path) as img:
            # 处理PNG透明通道（如果有），转换为JPG兼容格式（白色背景）
            if img.mode in ("RGBA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background.convert("RGB")

            original_width, original_height = img.size
            current_quality = 95  # 初始质量（从高质量开始递减）
            scale_factor = 1.0  # 分辨率缩放因子（1.0为原始分辨率）

            while True:
                # 2. 内存中保存图像，避免频繁写入磁盘
                img_buffer = BytesIO()
                img_resized = img.resize(
                    (int(original_width * scale_factor), int(original_height * scale_factor)),
                    Image.Resampling.LANCZOS  # 高质量缩放算法
                )
                img_resized.save(img_buffer, format="JPEG", quality=current_quality, optimize=True)
                current_size_bytes = img_buffer.tell()
                current_size_kb = current_size_bytes / 1024

                # 3. 判断当前大小是否在目标范围内
                if lower_bound <= current_size_bytes <= upper_bound:
                    # 符合要求，写入最终文件
                    with open(output_img_path, "wb") as f:
                        f.write(img_buffer.getvalue())
                    return True, f"压缩成功，分辨率缩放{scale_factor:.2f}倍，最终质量{current_quality}", current_size_kb

                # 4. 大小超出范围，调整参数
                # 4.1 先尝试降低质量（优先保留分辨率）
                if current_size_bytes > upper_bound and current_quality > min_quality:
                    current_quality = max(min_quality, current_quality - quality_step)
                    continue

                # 4.2 质量已到下限，仍过大，缩小分辨率
                if current_size_bytes > upper_bound and current_quality <= min_quality:
                    # 计算需要缩放的比例（按文件大小近似比例，预留一定余量）
                    scale_factor *= 0.9  # 每次缩小10%分辨率
                    # 防止分辨率无限缩小
                    if scale_factor < 0.1:
                        return False, "分辨率已缩小至10%以下，仍无法达到目标大小", current_size_kb
                    continue

                # 4.3 大小过小，尝试提高质量（若未到上限）
                if current_size_bytes < lower_bound and current_quality < 100:
                    current_quality = min(100, current_quality + quality_step)
                    continue

                # 4.4 质量已到100，仍过小，放大分辨率（不超过原始分辨率）
                if current_size_bytes < lower_bound and current_quality >= 100 and scale_factor < 1.0:
                    scale_factor *= 1.1  # 每次放大10%分辨率
                    continue

                # 5. 无法进一步调整，返回当前最优结果
                with open(output_img_path, "wb") as f:
                    f.write(img_buffer.getvalue())
                return True, f"已调整至最优状态（质量{current_quality}，缩放{scale_factor:.2f}倍），无法完全匹配目标大小", current_size_kb

    except Exception as e:
        return False, f"压缩过程出错：{str(e)}", 0.0


# ---------------------- 使用示例 ----------------------
if __name__ == "__main__":
    # 配置参数
    INPUT_IMAGE = r"E:\dev\Code\Python\bones\证件照_白底.jpg"  # 你的输入图像路径
    OUTPUT_IMAGE = "output_compressed.jpg"  # 压缩后保存路径
    TARGET_SIZE_KB = 200  # 目标大小200KB

    # 执行压缩
    success, msg, final_size = compress_image_to_target_size(
        input_img_path=INPUT_IMAGE,
        output_img_path=OUTPUT_IMAGE,
        target_size_kb=TARGET_SIZE_KB
    )

    # 打印结果
    print(f"结果：{'成功' if success else '失败'}")
    print(f"提示：{msg}")
    print(f"最终文件大小：{final_size:.2f} KB")