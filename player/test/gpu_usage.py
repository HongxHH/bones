import time
import pynvml
import torch

def init_gpu_monitor(gpu_id=0):
    """
    初始化GPU监测，返回GPU总内存（字节）
    :param gpu_id: GPU编号（多GPU时指定，默认0号GPU）
    :return: GPU总内存（bytes）
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = gpu_info.total  # 总内存（字节）
    print(f"GPU {gpu_id} 总内存：{total_memory / 1024**3:.2f} GB")
    return handle, total_memory

def allocate_fixed_gpu_memory(handle, total_memory, target_mem_ratio=0.7, gpu_id=0):
    """
    分配固定大小的GPU内存，稳定内存占用率
    :param handle: GPU句柄
    :param total_memory: GPU总内存（bytes）
    :param target_mem_ratio: 目标内存占用率（0-1之间，比如0.7表示70%）
    :param gpu_id: GPU编号
    :return: 分配的GPU张量（需保持引用，避免被释放）
    """
    # 计算目标占用内存大小（预留少量内存给系统，避免占满导致报错）
    target_memory = int(total_memory * target_mem_ratio)
    # PyTorch中，张量大小按字节计算：float32=4字节/元素，float64=8字节/元素
    element_size = 4  # 使用float32，节省计算资源
    num_elements = target_memory // element_size

    # 在GPU上分配张量（设为requires_grad=False，不参与梯度计算，避免被自动释放）
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise Exception("未检测到NVIDIA GPU，无法分配GPU内存")
    
    # 分配一维张量（形状不影响内存占用，仅需总元素数达标）
    fixed_tensor = torch.randn(num_elements, device=device, requires_grad=False)
    # 验证实际内存占用
    current_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    current_mem_ratio = current_mem_info.used / total_memory
    print(f"实际GPU内存占用率：{current_mem_ratio:.2%}（目标：{target_mem_ratio:.2%}）")
    print(f"实际占用内存：{current_mem_info.used / 1024**3:.2f} GB")
    return fixed_tensor

def gpu_workload_task(device, batch_size=1024, matrix_size=512):
    """
    生成可循环执行的GPU运算任务（矩阵乘法，轻量且可控）
    :param device: GPU设备
    :param batch_size: 批次大小（影响单次运算量）
    :param matrix_size: 矩阵维度（影响单次运算量）
    :return: 运算函数
    """
    def task():
        # 生成随机矩阵并执行乘法（每次运算都生成新矩阵，保证GPU有工作可做）
        a = torch.randn(batch_size, matrix_size, matrix_size, device=device, requires_grad=False)
        b = torch.randn(batch_size, matrix_size, matrix_size, device=device, requires_grad=False)
        c = torch.matmul(a, b)
        # 同步GPU，确保运算执行完毕（避免异步执行导致利用率监测不准）
        torch.cuda.synchronize(device)
    return task

def maintain_fixed_gpu_util(handle, gpu_id=0, target_util_ratio=0.8, 
                            mem_target_ratio=0.7, adjust_interval=0.5):
    """
    维持GPU利用率在目标区间，同时稳定内存占用率
    :param handle: GPU句柄
    :param gpu_id: GPU编号
    :param target_util_ratio: 目标GPU利用率（0-1，比如0.8=80%）
    :param mem_target_ratio: 目标内存占用率
    :param adjust_interval: 利用率监测与调节间隔（秒，越小调节越灵敏）
    """
    # 1. 初始化GPU内存（固定占用）
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    fixed_tensor = allocate_fixed_gpu_memory(handle, total_memory, mem_target_ratio, gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # 2. 初始化GPU运算任务
    work_task = gpu_workload_task(device)
    # 初始运算间隔（可根据实际情况调整）
    work_interval = 0.01

    # 3. 闭环调控主循环
    print(f"\n开始维持GPU利用率在 {target_util_ratio:.2%} 附近，按 Ctrl+C 停止...")
    try:
        while True:
            # 获取当前GPU利用率（单位：%，取最近1秒的平均值）
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            current_util = util_info.gpu / 100.0  # 转为0-1的比例

            # 4. 动态调节运算间隔
            if current_util < target_util_ratio - 0.05:
                # 利用率偏低，减少间隔（让GPU更繁忙）
                work_interval = max(0.001, work_interval * 0.9)
            elif current_util > target_util_ratio + 0.05:
                # 利用率偏高，增加间隔（让GPU休息）
                work_interval = min(0.5, work_interval * 1.1)

            # 5. 执行GPU运算任务
            work_task()
            # 按调节后的间隔休息
            time.sleep(work_interval)

            # 6. 打印状态（可选，便于观察）
            current_mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            current_mem_ratio = current_mem_info.used / total_memory
            print(f"\r当前GPU利用率：{current_util:.2%} | 内存占用率：{current_mem_ratio:.2%} "
                  f"| 运算间隔：{work_interval:.4f}s", end="", flush=True)

            # 按固定间隔监测（避免打印过于频繁）
            time.sleep(adjust_interval - work_interval)

    except KeyboardInterrupt:
        print("\n\n停止GPU任务，释放资源...")
    finally:
        # 释放pynvml资源
        pynvml.nvmlShutdown()
        # PyTorch张量会在程序结束后自动释放，无需手动操作
        del fixed_tensor
        torch.cuda.empty_cache()
        print("资源释放完成")

if __name__ == "__main__":
    # 配置参数（可根据你的需求修改）
    GPU_ID = 0  # 使用0号GPU
    TARGET_UTIL_RATIO = 0.8  # 目标GPU利用率80%
    TARGET_MEM_RATIO = 0.7   # 目标GPU内存占用率70%

    # 初始化GPU监测并启动维持程序
    gpu_handle, _ = init_gpu_monitor(GPU_ID)
    maintain_fixed_gpu_util(
        handle=gpu_handle,
        gpu_id=GPU_ID,
        target_util_ratio=TARGET_UTIL_RATIO,
        mem_target_ratio=TARGET_MEM_RATIO
    )