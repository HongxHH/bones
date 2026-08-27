import time
import pynvml
import torch
import threading
from collections import deque


class PIDController:
    """PID控制器，用于动态调节GPU利用率"""
    def __init__(self, kp=1.0, ki=0.1, kd=0.05, target=0.5):
        self.kp = kp  # 比例系数
        self.ki = ki  # 积分系数
        self.kd = kd  # 微分系数
        self.target = target  # 目标值
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()
    
    def update(self, current_value):
        """更新PID控制器，返回控制输出"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            dt = 0.001
        
        error = self.target - current_value
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项（防止积分饱和）
        self.integral += error * dt
        self.integral = max(-10.0, min(10.0, self.integral))  # 限制积分项
        i_term = self.ki * self.integral
        
        # 微分项
        d_term = self.kd * (error - self.last_error) / dt
        
        output = p_term + i_term + d_term
        
        self.last_error = error
        self.last_time = current_time
        
        return output


class GPUMemoryController:
    """GPU内存占用控制器"""
    def __init__(self, device, handle, total_memory):
        self.device = device
        self.handle = handle
        self.total_memory = total_memory
        self.memory_tensors = []  # 存储所有内存张量
        self.element_size = 4  # float32 = 4字节
    
    def get_current_memory_ratio(self):
        """获取当前内存占用率"""
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        return mem_info.used / self.total_memory
    
    def allocate_memory(self, target_ratio):
        """分配内存以达到目标占用率"""
        # 先释放现有内存
        self.release_all()
        
        # 计算目标内存（预留5%给系统）
        safe_target_ratio = min(target_ratio, 0.95)
        target_memory = int(self.total_memory * safe_target_ratio)
        
        # 获取当前已占用内存（可能包括其他进程）
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        current_used = mem_info.used
        
        # 计算需要分配的内存
        need_allocate = max(0, target_memory - current_used)
        
        if need_allocate > 0:
            num_elements = need_allocate // self.element_size
            if num_elements > 0:
                # 分批分配，避免一次性分配失败
                batch_size = min(num_elements, 100000000)  # 每次最多分配约400MB
                allocated = 0
                while allocated < num_elements:
                    current_batch = min(batch_size, num_elements - allocated)
                    tensor = torch.randn(current_batch, device=self.device, requires_grad=False)
                    self.memory_tensors.append(tensor)
                    allocated += current_batch
                    # 检查是否达到目标
                    current_ratio = self.get_current_memory_ratio()
                    if current_ratio >= safe_target_ratio:
                        break
        
        return self.get_current_memory_ratio()
    
    def adjust_memory(self, target_ratio, tolerance=0.02):
        """动态调整内存占用"""
        current_ratio = self.get_current_memory_ratio()
        
        if abs(current_ratio - target_ratio) > tolerance:
            return self.allocate_memory(target_ratio)
        
        return current_ratio
    
    def release_all(self):
        """释放所有分配的内存"""
        for tensor in self.memory_tensors:
            del tensor
        self.memory_tensors.clear()
        torch.cuda.empty_cache()


class GPUWorkloadController:
    """GPU工作负载控制器"""
    def __init__(self, device):
        self.device = device
        self.batch_size = 64  # 初始批次大小
        self.matrix_size = 256  # 初始矩阵大小
        self.work_interval = 0.0  # 工作间隔（秒）
        self.min_batch_size = 1
        self.max_batch_size = 2048
        self.min_matrix_size = 64
        self.max_matrix_size = 2048
    
    def execute_workload(self):
        """执行GPU工作负载"""
        a = torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                       device=self.device, requires_grad=False)
        b = torch.randn(self.batch_size, self.matrix_size, self.matrix_size, 
                       device=self.device, requires_grad=False)
        c = torch.matmul(a, b)
        torch.cuda.synchronize(self.device)
    
    def adjust_workload(self, pid_output):
        """根据PID输出调整工作负载"""
        # PID输出为正表示需要增加负载，为负表示需要减少负载
        # 通过调整批次大小和矩阵大小来控制
        
        if pid_output > 0:
            # 需要增加负载
            if self.work_interval > 0:
                self.work_interval = max(0.0, self.work_interval - 0.001)
            else:
                # 增加批次大小或矩阵大小
                if self.batch_size < self.max_batch_size:
                    self.batch_size = min(self.max_batch_size, 
                                        int(self.batch_size * (1 + abs(pid_output) * 0.1)))
                elif self.matrix_size < self.max_matrix_size:
                    self.matrix_size = min(self.max_matrix_size, 
                                         int(self.matrix_size * (1 + abs(pid_output) * 0.05)))
        else:
            # 需要减少负载
            if self.batch_size > self.min_batch_size:
                self.batch_size = max(self.min_batch_size, 
                                    int(self.batch_size * (1 + pid_output * 0.1)))
            elif self.matrix_size > self.min_matrix_size:
                self.matrix_size = max(self.min_matrix_size, 
                                     int(self.matrix_size * (1 + pid_output * 0.05)))
            else:
                # 增加工作间隔
                self.work_interval = min(0.1, self.work_interval + 0.001)
    
    def get_workload_info(self):
        """获取工作负载信息"""
        return {
            'batch_size': self.batch_size,
            'matrix_size': self.matrix_size,
            'work_interval': self.work_interval
        }


def init_gpu_monitor(gpu_id=0):
    """
    初始化GPU监测，返回GPU信息和句柄
    :param gpu_id: GPU编号（多GPU时指定，默认0号GPU）
    :return: GPU句柄、总内存、GPU名称等信息
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    # 获取GPU基本信息
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = mem_info.total
    
    # 获取GPU计算能力
    try:
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap_str = f"{compute_cap[0]}.{compute_cap[1]}"
    except:
        compute_cap_str = "未知"
    
    print(f"GPU {gpu_id} 信息：")
    print(f"  名称：{gpu_name}")
    print(f"  总内存：{total_memory / 1024**3:.2f} GB")
    print(f"  计算能力：{compute_cap_str}")
    
    return handle, total_memory, gpu_name


def maintain_gpu_utilization(handle, gpu_id=0, target_util_ratio=0.8, 
                             target_mem_ratio=0.7, adjust_interval=0.2):
    """
    维持GPU利用率和内存占用率在目标值附近
    :param handle: GPU句柄
    :param gpu_id: GPU编号
    :param target_util_ratio: 目标GPU利用率（0-1，比如0.8=80%）
    :param target_mem_ratio: 目标内存占用率（0-1，比如0.7=70%）
    :param adjust_interval: 调节间隔（秒）
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    total_memory = pynvml.nvmlDeviceGetMemoryInfo(handle).total
    
    # 初始化控制器
    pid_controller = PIDController(kp=2.0, ki=0.2, kd=0.1, target=target_util_ratio)
    memory_controller = GPUMemoryController(device, handle, total_memory)
    workload_controller = GPUWorkloadController(device)
    
    # 初始化内存占用
    print(f"\n初始化GPU内存占用率到 {target_mem_ratio:.2%}...")
    memory_controller.allocate_memory(target_mem_ratio)
    initial_mem_ratio = memory_controller.get_current_memory_ratio()
    print(f"实际GPU内存占用率：{initial_mem_ratio:.2%}（目标：{target_mem_ratio:.2%}）")
    
    # 预热GPU
    print("预热GPU...")
    for _ in range(10):
        workload_controller.execute_workload()
    time.sleep(0.5)
    
    print(f"\n开始维持GPU利用率在 {target_util_ratio:.2%} 附近，按 Ctrl+C 停止...")
    print("-" * 80)
    
    # 用于平滑GPU利用率读数
    util_history = deque(maxlen=5)
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            # 获取当前GPU利用率
            util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            current_util = util_info.gpu / 100.0
            util_history.append(current_util)
            
            # 使用平均值来平滑读数
            smoothed_util = sum(util_history) / len(util_history)
            
            # PID控制更新
            pid_output = pid_controller.update(smoothed_util)
            
            # 调整工作负载
            workload_controller.adjust_workload(pid_output)
            
            # 执行工作负载
            workload_controller.execute_workload()
            
            # 如果设置了工作间隔，则等待
            if workload_controller.work_interval > 0:
                time.sleep(workload_controller.work_interval)
            
            # 定期调整内存占用（每10次迭代调整一次，避免频繁调整）
            if iteration % 10 == 0:
                memory_controller.adjust_memory(target_mem_ratio, tolerance=0.03)
            
            # 获取当前内存占用率
            current_mem_ratio = memory_controller.get_current_memory_ratio()
            
            # 打印状态（每5次迭代打印一次，避免输出过于频繁）
            if iteration % 5 == 0:
                workload_info = workload_controller.get_workload_info()
                print(f"\rGPU利用率：{smoothed_util:.2%} (目标: {target_util_ratio:.2%}) | "
                      f"内存占用率：{current_mem_ratio:.2%} (目标: {target_mem_ratio:.2%}) | "
                      f"批次: {workload_info['batch_size']}, "
                      f"矩阵: {workload_info['matrix_size']}, "
                      f"间隔: {workload_info['work_interval']:.4f}s", 
                      end="", flush=True)
            
            # 控制循环频率
            time.sleep(adjust_interval)
    
    except KeyboardInterrupt:
        print("\n\n停止GPU任务，释放资源...")
    finally:
        # 释放资源
        memory_controller.release_all()
        pynvml.nvmlShutdown()
        print("资源释放完成")


if __name__ == "__main__":
    # 配置参数（可根据你的需求修改）
    GPU_ID = 0  # 使用0号GPU
    TARGET_UTIL_RATIO = 0.5   # 目标GPU利用率50%
    TARGET_MEM_RATIO = 0.6    # 目标GPU内存占用率60%
    
    # 初始化GPU监测并启动维持程序
    gpu_handle, total_mem, gpu_name = init_gpu_monitor(GPU_ID)
    maintain_gpu_utilization(
        handle=gpu_handle,
        gpu_id=GPU_ID,
        target_util_ratio=TARGET_UTIL_RATIO,
        target_mem_ratio=TARGET_MEM_RATIO,
        adjust_interval=0.2  # 200ms调节间隔，响应更快
    )
