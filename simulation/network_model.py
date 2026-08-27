import copy  # 导入深拷贝模块，用于创建对象的完全独立副本，避免修改原始数据


class NetworkModel():
    def __init__(self, logger, net_time, bandwidth, latency):
        """
        基础网络模型构造函数：基于离线轨迹的分段常值带宽/时延仿真
        模拟网络环境，支持跨周期的下载和时延计算
        
        :param logger: 信息日志器，用于输出调试和状态信息
        :param net_time: 网络周期时长数组 (num_period,)，每个元素是一个周期的持续时间（毫秒）
        :param bandwidth: 网络周期带宽数组 (num_period,)，每个元素是一个周期的带宽（kbps）
        :param latency: 网络周期时延数组 (num_period,)，每个元素是一个周期的时延（毫秒）
        """
        # 基础配置参数
        self.logger = logger  # 日志器实例，用于输出调试信息
        self.net_time = net_time  # 网络周期时长数组，定义每个周期的持续时间
        self.bandwidth = bandwidth  # 网络周期带宽数组，定义每个周期的传输速率
        self.latency = latency  # 网络周期时延数组，定义每个周期的网络延迟

        # 网络状态变量
        self.num_period = net_time.shape[0]  # 网络周期总数，用于循环遍历和边界检查
        self.idx_period = -1  # 当前网络周期索引，初始为-1表示未开始任何周期
        self.time_to_next = 0  # 当前周期剩余时间（毫秒），用于跟踪周期内的时间进度

    def download(self, size):
        """
        模拟下载一个分段的完整过程：计算首包时延、下载时间、吞吐量
        这是网络模型的核心方法，处理下载的完整生命周期
        
        :param size: 要下载的分段大小（比特），必须非负
        :return: (下载时间, 吞吐量, 首包时延) 三元组
        """
        assert size >= 0  # 断言检查，确保分段大小非负，防止无效输入
        
        # 计算首包时延：模拟网络往返时延
        delay_lat = self.delay_latency(1)  # 调用时延计算方法，参数1表示1个时延单位
        # 首包时延通常是一次网络往返的时间
        
        # 计算纯下载时间：基于带宽和分段大小
        delay_down = self.delay_download(size)  # 调用下载时间计算方法
        # 下载时间 = 分段大小 / 带宽，可能跨越多个月期
        
        # 计算平均吞吐量
        tput = size / delay_down  # 吞吐量 = 分段大小 / 下载时间（比特/毫秒）
        # 注意：这里计算的是实际吞吐量，可能低于网络带宽

        # 详细日志输出：记录下载过程的详细信息
        if self.logger.is_verbose():  # 检查是否开启详细日志模式
            self.logger.write(
                "(download) size {:.2f} total_time {:.2f} latency {:.2f} download_time {:.2f} throughput {:.2f}"
                .format(size, delay_lat + delay_down, delay_lat, delay_down, tput
            ))
            # 输出格式：分段大小、总时间、时延、下载时间、吞吐量，保留2位小数
        
        return delay_down, tput, delay_lat  # 返回下载时间、吞吐量、首包时延

    def delay_download(self, size):
        """
        计算下载任务引起的时延：在多个网络周期上积分下载时间
        处理跨周期的下载场景，确保时间计算的准确性
        
        :param size: 要下载的分段大小（比特），必须非负
        :return: 下载引起的总时延（毫秒）
        """
        total_time = 0  # 累计下载时间，用于跨周期计算
        
        while size > 0:  # 循环直到所有数据下载完成
            current_bandwidth = self.bandwidth[self.idx_period]  # 获取当前周期的带宽
            # 注意：这里假设idx_period已经正确设置
            
            if size <= self.time_to_next * current_bandwidth:  # 检查当前周期是否能完成下载
                # 当前周期内能完成下载
                time = size / current_bandwidth  # 计算在当前周期内下载所需时间
                # 公式：时间 = 数据量 / 带宽
                total_time += time  # 累加到总下载时间
                self.time_to_next -= time  # 减少当前周期剩余时间
                size = 0  # 数据下载完成，退出循环
            else:  # 当前周期无法完成下载，需要跨周期
                # 使用当前周期的全部剩余时间
                total_time += self.time_to_next  # 累加当前周期剩余时间
                size -= self.time_to_next * current_bandwidth  # 减少已下载的数据量
                # 公式：剩余数据 = 原数据 - (剩余时间 × 带宽)
                self.next_period()  # 切换到下一个网络周期
        
        return total_time  # 返回总下载时间

    def delay_latency(self, delay_units):
        """
        计算网络时延引起的延迟：模拟网络往返时延
        处理跨周期的时延计算，确保时延的连续性
        
        :param delay_units: 时延单位数量，通常为1表示一次往返
        :return: 时延引起的总延迟（毫秒）
        """
        total_time = 0  # 累计时延时间，用于跨周期计算
        
        while delay_units > 0:  # 循环直到所有时延单位处理完成
            current_latency = self.latency[self.idx_period]  # 获取当前周期的时延
            time = delay_units * current_latency  # 计算时延时间
            # 公式：时延时间 = 时延单位数 × 单次时延
            
            if time <= self.time_to_next:  # 检查当前周期是否能完成时延
                # 当前周期内能完成时延
                total_time += time  # 累加到总时延时间
                self.time_to_next -= time  # 减少当前周期剩余时间
                delay_units = 0  # 时延完成，退出循环
            else:  # 当前周期无法完成时延，需要跨周期
                # 使用当前周期的全部剩余时间
                total_time += self.time_to_next  # 累加当前周期剩余时间
                delay_units -= self.time_to_next / current_latency  # 减少已处理的时延单位
                # 公式：剩余时延单位 = 原时延单位 - (剩余时间 / 单次时延)
                self.next_period()  # 切换到下一个网络周期
        
        return total_time  # 返回总时延时间

    def go_by(self, time):
        """
        时间流逝：推进网络时间线，可能跨越多个月期
        用于模拟网络环境的自然时间流逝，不进行任何下载操作
        
        :param time: 要推进的时间量（毫秒），必须非负
        :return: 无返回值，通过修改实例属性更新状态
        """
        while time > self.time_to_next:  # 循环直到推进时间用完
            # 要推进的时间超过当前周期剩余时间
            time -= self.time_to_next  # 减去当前周期剩余时间
            self.next_period()  # 切换到下一个周期
        
        # 处理剩余时间
        self.time_to_next -= time  # 减少当前周期剩余时间
        return  # 方法结束

    def next_period(self):
        """
        切换到下一个网络周期：更新周期索引和剩余时间
        处理周期循环，确保网络轨迹的连续性
        
        :return: 无返回值，通过修改实例属性更新状态
        """
        self.idx_period += 1  # 周期索引自增
        
        if self.idx_period == self.num_period:  # 检查是否到达最后一个周期
            self.idx_period = 0  # 循环回到第一个周期，实现无限循环
        
        self.time_to_next = self.net_time[self.idx_period]  # 设置新周期的剩余时间
        # 新周期开始时，剩余时间等于周期总时长

        # 详细日志输出：记录网络周期切换信息
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(network) period {} time {} bandwidth {} latency {}"
                              .format(self.idx_period, self.net_time[self.idx_period],
                                      self.bandwidth[self.idx_period], self.latency[self.idx_period]))
            # 输出格式：周期索引、周期时长、带宽、时延
        
        return  # 方法结束


class MonitorNetworkModel(NetworkModel):
    def __init__(self, logger, net_time, bandwidth, latency, monitor_hook, buffer_enhance, buffer_download):
        """
        监控网络模型构造函数：允许在下载过程中监控并可能切换下载选项
        继承自NetworkModel，添加了分片监控和动态切换功能
        
        :param logger: 信息日志器，用于输出调试和状态信息
        :param net_time: 网络周期时长数组 (num_period,)
        :param bandwidth: 网络周期带宽数组 (num_period,)
        :param latency: 网络周期时延数组 (num_period,)
        :param monitor_hook: 监控回调函数，用于决定是否切换下载选项
        :param buffer_enhance: 增强缓冲引用，用于监控增强缓冲状态
        :param buffer_download: 下载缓冲引用，用于监控下载缓冲状态
        """
        # 调用父类构造函数
        super(MonitorNetworkModel, self).__init__(logger, net_time, bandwidth, latency)
        
        # 监控参数配置
        self.min_monitor_size = 12000  # 每个监控间隔的最小下载大小（比特）
        # 确保监控间隔不会太小，避免频繁切换
        self.min_monitor_time = 50  # 每个监控间隔的最小时长（毫秒）
        # 确保监控间隔有足够的时间进行决策
        
        # 监控相关引用
        self.monitor_hook = monitor_hook  # 监控回调函数，由外部策略提供
        self.buffer_enhance = buffer_enhance  # 增强缓冲引用，用于状态监控
        self.buffer_download = buffer_download  # 下载缓冲引用，用于状态监控

    def download(self, size):
        """
        分片监控下载：将下载过程分解为小片段，每片段后调用监控回调决定是否切换
        这是监控网络模型的核心方法，实现动态下载策略调整
        
        :param size: 要下载的分段大小（比特），必须非负
        :return: (下载时间, 吞吐量, 首包时延) 三元组
        """
        assert size >= 0  # 断言检查，确保分段大小非负
        
        # 在虚拟时间轴上模拟下载过程，深拷贝缓冲状态避免修改原始数据
        buffer_enh = copy.deepcopy(self.buffer_enhance.buffer_level)  # 增强缓冲水位副本
        buffer_down = copy.deepcopy(self.buffer_download.buffer_level)  # 下载缓冲水位副本
        # 使用深拷贝确保监控过程不会影响真实的缓冲状态
        
        # 下载统计变量
        total_download_time = 0  # 累计下载时间
        total_download_size = 0  # 累计下载大小
        remain_size = copy.deepcopy(size)  # 剩余要下载的大小

        # 计算首包时延
        latency = self.delay_latency(1)  # 1个时延单位，模拟网络往返
        total_download_time += latency  # 累加到总时间

        # 分片下载循环：将大下载分解为小片段
        while remain_size > 0:  # 循环直到所有数据下载完成
            # 下载一个监控片段
            bits, time = self.delay_download_monitor(remain_size)  # 调用监控下载方法
            total_download_time += time  # 累加时间
            total_download_size += bits  # 累加下载大小
            remain_size -= bits  # 减少剩余大小

            # 模拟下载过程：更新虚拟缓冲水位
            buffer_enh = max(0, buffer_enh - time)  # 增强缓冲水位随时间减少
            buffer_down = max(0, buffer_down - time)  # 下载缓冲水位随时间减少
            # 使用max(0, ...)确保水位不会变为负数

            if remain_size > 0:  # 如果还有剩余数据需要下载
                # 监控下载过程：调用回调函数决定是否切换
                tput = total_download_size / (total_download_time - latency)  # 计算当前吞吐量
                # 公式：吞吐量 = 总下载大小 / (总时间 - 首包时延)
                
                # 调用监控回调函数
                new_size = self.monitor_hook(buffer_down, buffer_enh, tput, latency, remain_size, total_download_time)
                # 参数：下载缓冲水位、增强缓冲水位、吞吐量、时延、剩余大小、总时间
                
                if new_size is not None:  # 如果回调返回新的目标大小
                    # 中止当前下载，开始新的下载
                    remain_size = new_size  # 更新剩余大小为新的目标
                    total_download_time += latency  # 新下载会产生额外的首包时延
                    
                    if self.logger.is_verbose():  # 详细日志输出切换信息
                        self.logger.write("(download) new size {:.2f}".format(new_size))

        # 将整个过程汇总为一个下载事件
        delay_down = total_download_time - latency  # 纯下载时间（减去首包时延）
        tput = total_download_size / delay_down  # 平均吞吐量
        delay_lat = latency  # 首包时延

        # 详细日志输出：记录最终下载信息
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write(
                "(download) size {:.2f} total_time {:.2f} latency {:.2f} download_time {:.2f} throughput {:.2f}"
                .format(size, delay_lat + delay_down, delay_lat, delay_down, tput
            ))
            # 输出格式：原始大小、总时间、时延、下载时间、吞吐量
        
        return delay_down, tput, delay_lat  # 返回下载时间、吞吐量、首包时延

    def delay_download_monitor(self, size):
        """
        在监控间隔内计算下载时延：确保每个间隔满足最小大小和最小时间约束
        这是监控网络模型的核心辅助方法，处理分片下载的复杂逻辑
        
        :param size: 要下载的分段大小（比特），必须非负
        :return: (实际下载大小, 实际下载时间) 元组
        """
        # 复制监控约束参数，避免修改原始值
        min_size = copy.deepcopy(self.min_monitor_size)  # 最小下载大小
        min_time = copy.deepcopy(self.min_monitor_time)  # 最小时长
        
        # 下载统计变量
        total_size = 0  # 累计下载大小
        total_time = 0  # 累计下载时间
        
        # 监控间隔下载循环
        while size > 0 and (min_size > 0 or min_time > 0):  # 还有数据要下载且未满足最小约束
            current_bandwidth = self.bandwidth[self.idx_period]  # 获取当前周期带宽
            
            if current_bandwidth > 0:  # 带宽大于0，可以进行下载
                # 计算满足最小约束需要的最小比特数
                min_bits = max(min_size, min_time * current_bandwidth)
                # 公式：最小比特数 = max(最小大小, 最小时长 × 带宽)
                
                # 计算当前周期能下载的最大比特数
                bits_to_next = self.time_to_next * current_bandwidth
                # 公式：最大比特数 = 剩余时间 × 带宽
                
                # 总是选择最短的时间间隔，优化下载效率
                if size <= min_bits and size <= bits_to_next:  # 当前数据量满足所有约束
                    # 下载全部剩余数据
                    bits = size  # 下载全部剩余数据
                    time = bits / current_bandwidth  # 计算下载时间
                    self.time_to_next -= time  # 减少当前周期剩余时间
                elif min_bits <= bits_to_next:  # 最小约束优先
                    # 下载满足最小约束的数据量
                    bits = min_bits  # 下载满足最小约束的数据量
                    time = bits / current_bandwidth  # 计算下载时间
                    min_size = 0  # 重置最小大小约束
                    min_time = 0  # 重置最小时长约束
                    self.time_to_next -= time  # 减少当前周期剩余时间
                else:  # 当前周期无法满足最小约束，使用当前周期全部时间
                    # 下载当前周期能下载的最大量
                    bits = bits_to_next  # 下载当前周期能下载的最大量
                    time = self.time_to_next  # 使用当前周期全部剩余时间
                    self.next_period()  # 切换到下一个周期
            else:  # 当前带宽为0，无法下载
                bits = 0  # 无法下载任何数据
                if min_size > 0 or min_time > self.time_to_next:  # 需要等待到下一个周期
                    # 等待当前周期结束
                    time = self.time_to_next  # 等待当前周期结束
                    self.next_period()  # 切换到下一个周期
                else:  # 满足最小时间约束
                    # 等待最小时长
                    time = min_time  # 等待最小时长
                    self.time_to_next -= time  # 减少当前周期剩余时间
            
            # 更新统计变量
            total_size += bits  # 累加下载大小
            total_time += time  # 累加下载时间
            size -= bits  # 减少剩余大小
            min_size -= bits  # 减少最小大小约束
            min_time -= time  # 减少最小时长约束
        
        return total_size, total_time  # 返回实际下载大小和时间

    def delay_latency_monitor(self, delay_units):
        """
        在监控间隔内计算网络时延：确保满足最小时间约束
        这是监控网络模型的辅助方法，处理分片时延计算
        
        :param delay_units: 时延单位数量，必须非负
        :return: (实际时延单位, 实际时延时间) 元组
        """
        # 复制监控约束参数
        min_size = copy.deepcopy(self.min_monitor_size)  # 最小大小约束（未使用）
        min_time = copy.deepcopy(self.min_monitor_time)  # 最小时长约束
        
        # 时延统计变量
        total_delay_units = 0  # 累计时延单位
        total_delay_time = 0  # 累计时延时间
        
        # 监控间隔时延循环
        while delay_units > 0 and min_time > 0:  # 还有时延单位要处理且未满足最小时长约束
            current_latency = self.latency[self.idx_period]  # 获取当前周期时延
            time = delay_units * current_latency  # 计算时延时间
            # 公式：时延时间 = 时延单位数 × 单次时延
            
            if time <= min_time and time <= self.time_to_next:  # 满足所有约束
                # 处理全部时延单位
                units = delay_units  # 处理全部时延单位
                self.time_to_next -= time  # 减少当前周期剩余时间
            elif min_time <= self.time_to_next:  # 最小时长约束优先
                # 使用最小时长
                time = min_time  # 使用最小时长
                units = time / current_latency  # 计算能处理的时延单位数
                # 公式：时延单位数 = 时间 / 单次时延
                self.time_to_next -= time  # 减少当前周期剩余时间
            else:  # 当前周期无法满足最小时长约束
                # 使用当前周期全部剩余时间
                time = self.time_to_next  # 使用当前周期全部剩余时间
                units = time / current_latency  # 计算能处理的时延单位数
                self.next_period()  # 切换到下一个周期
            
            # 更新统计变量
            total_delay_units += units  # 累加时延单位
            total_delay_time += time  # 累加时延时间
            delay_units -= units  # 减少剩余时延单位
            min_time -= time  # 减少最小时长约束
        
        return total_delay_units, total_delay_time  # 返回实际时延单位和时间


def test_network_model():
    """
    测试网络模型：加载轨迹数据并测试下载和时延计算
    这是一个简单的单元测试函数，演示网络模型的使用
    """
    import json  # 导入JSON文件处理模块
    import numpy as np  # 导入数值计算库
    from logger import Logger  # 导入日志器类
    
    # 解析网络轨迹
    bandwidth_multiplier = 1  # 带宽倍率，用于调整网络条件
    trace = json.load(open("../data/trace/dummy.json"))  # 加载轨迹文件
    # 假设轨迹文件包含网络条件的历史数据
    
    # 初始化轨迹数据列表
    net_time = []  # 网络周期时长列表
    bandwidth = []  # 网络周期带宽列表
    latency = []  # 网络周期时延列表
    
    # 遍历轨迹数据，提取网络参数
    for i in range(len(trace)):  # 遍历轨迹数据
        net_time.append(trace[i]['duration_ms'])  # 添加周期时长
        bandwidth.append(trace[i]['bandwidth_kbps'])  # 添加周期带宽
        latency.append(trace[i]['latency_ms'])  # 添加周期时延
    
    # 转换为numpy数组并应用倍率
    net_time = np.array(net_time)  # 转换为numpy数组
    bandwidth = np.array(bandwidth) * bandwidth_multiplier  # 应用带宽倍率
    latency = np.array(latency)  # 转换为numpy数组
    
    # 测试网络模型
    network = NetworkModel(Logger(), net_time, bandwidth, latency)  # 创建网络模型实例
    
    # 进行100次随机测试
    for i in range(100):  # 进行100次测试
        size = np.random.uniform(500000, 5000000)  # 随机分段大小（比特）
        time = np.random.uniform(0, 1000)  # 随机时间推进（毫秒）
        network.download(size)  # 测试下载功能
        network.go_by(time)  # 测试时间推进功能
        print("test", i, "size", size, "time", time, "period", network.idx_period, "next", network.time_to_next)  # 输出测试结果


if __name__ == '__main__':
    test_network_model()  # 当文件作为脚本直接运行时执行测试