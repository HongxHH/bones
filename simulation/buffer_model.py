import numpy as np  # 导入numpy数值计算库，用于数组操作和数据类型定义


class BufferModel():
    def __init__(self, logger, max_buffer_level, num_seg, name):
        """
        缓冲区模型构造函数：初始化缓冲区的基本属性和状态
        :param logger: 信息日志器，用于输出调试和状态信息
        :param max_buffer_level: 最大缓冲水位，单位毫秒，防止缓冲区溢出
        :param num_seg: 视频分段总数，用于初始化决策历史数组大小
        :param name: 缓冲区名称，用于区分下载缓冲("download")和增强缓冲("enhance")
        """
        # 基础配置参数
        self.logger = logger  # 日志器实例，用于输出调试信息
        self.max_level = max_buffer_level  # 最大缓冲水位（毫秒），超过此值会触发溢出检查
        self.num_seg = num_seg  # 视频分段总数，用于确定决策历史数组的维度
        self.name = name  # 缓冲区名称字符串，用于日志输出和调试识别

        # 缓冲区内容管理
        self.content = []  # 分段内容队列，存储格式为[(segment_index, decision, segment_time), ...]
        # 每个元素是一个三元组：
        # - segment_index: 分段索引（整数）
        # - decision: 该分段的决策（码率索引或增强方法索引）
        # - segment_time: 分段时长（毫秒）
        
        self.history = np.zeros((num_seg, ), dtype=int)  # 决策历史数组，记录每个分段的最终决策
        # 数组索引对应分段索引，值为该分段的决策
        # dtype=int确保存储的是整数索引而不是浮点数
        
        # 缓冲区状态变量
        self.buffer_level = 0  # 当前缓冲水位（毫秒），表示缓冲区中所有分段的总时长
        self.time_spent = 0  # 当前分段已消费的时间（毫秒），用于部分消费场景

        # 性能统计记录
        self.idle_time = 0  # 累计空转时间（毫秒），当缓冲区为空但需要消费时累加；QoE 边界必须 /1000 换算为秒（与播放器 analyzer 同口径）
        self.idle_event = 0  # 空转事件计数，记录缓冲区耗尽的次数

    def consume(self, time):
        """
        从缓冲区消费内容：模拟播放或处理过程，更新缓冲水位和空转统计
        这是缓冲区的核心方法，处理分段消费的复杂逻辑
        
        :param time: 要消费的时间量（毫秒），必须非负
        :return: 无返回值，通过修改实例属性更新状态
        """
        # 详细日志输出：记录消费操作的开始
        if self.logger.is_verbose():  # 检查是否开启详细日志模式
            self.logger.write("(consume) name {} time {:.2f}".format(self.name, time))
            # 输出格式：缓冲区名称和消费时间，保留2位小数
        
        # 边界情况处理：缓冲区为空
        if self.buffer_level == 0:  # 检查缓冲区是否为空
            # 没有内容可消费，记录为空转时间
            self.idle_time += time  # 累加到总空转时间
            return  # 直接返回，不进行后续处理

        # 当前分段部分消费处理
        if self.time_spent > 0:  # 检查是否有分段正在部分消费中
            # 完成当前分段的处理
            _, _, segment_time = self.content[0]  # 解包队首分段信息，获取分段时长
            # 注意：这里只关心segment_time，所以用_忽略前两个值
            
            if time + self.time_spent < segment_time:  # 检查是否无法用完当前分段
                # 无法用完当前分段，只消费部分时间
                self.time_spent += time  # 增加已消费时间
                self.buffer_level -= time  # 减少缓冲水位（模拟消费）
                return  # 处理完成，直接返回
            
            # 用完当前分段的情况
            time -= segment_time - self.time_spent  # 计算剩余消费时间
            # 公式：剩余时间 = 总消费时间 - (分段时长 - 已消费时间)
            self.buffer_level -= segment_time - self.time_spent  # 减少缓冲水位
            self.content.pop(0)  # 从队列中移除已用完的分段
            self.time_spent = 0  # 重置已消费时间，准备处理下一个分段

        # 完整分段消费循环
        # 此时self.time_spent == 0，可以开始处理新的分段
        while time > 0 and len(self.content) > 0:  # 还有时间要消费且缓冲区有内容
            _, decision, segment_time = self.content[0]  # 获取队首分段信息
            # decision: 分段决策（码率或增强方法索引）
            # segment_time: 分段时长

            if time >= segment_time:  # 消费时间足够用完当前分段
                # 完全消费当前分段
                self.content.pop(0)  # 移除当前分段
                self.buffer_level -= segment_time  # 减少缓冲水位
                time -= segment_time  # 减少剩余消费时间
            else:  # 消费时间不足用完当前分段
                # 部分消费当前分段
                self.time_spent = time  # 记录已消费时间
                self.buffer_level -= time  # 减少缓冲水位
                time = 0  # 消费时间用完，退出循环

        # 空转处理：缓冲区耗尽但还有消费时间
        if time > 0:  # 检查是否还有剩余消费时间
            # 缓冲区耗尽，记录为空转
            self.idle_time += time  # 累加空转时间
            self.idle_event += 1  # 增加空转事件计数
            if self.logger.is_verbose():  # 详细日志输出空转信息
                self.logger.write("(consume) name {} idle time {}".format(self.name, time))

        return  # 方法结束

    def push(self, segment_index, decision, segment_time):
        """
        向缓冲区推入新分段：更新内容队列、决策历史和缓冲水位
        这是缓冲区的输入方法，负责接收新的分段
        
        :param segment_index: 分段索引（整数），用于标识分段和记录历史
        :param decision: 分段决策（整数），码率索引或增强方法索引
        :param segment_time: 分段时长（毫秒），用于计算缓冲水位
        :return: 无返回值，通过修改实例属性更新状态
        """
        # 更新内容队列
        self.content.append((segment_index, decision, segment_time))  # 添加分段到队列末尾
        # 三元组格式：(分段索引, 决策, 时长)
        
        # 更新决策历史
        self.history[segment_index] = decision  # 记录该分段的决策到历史数组
        # 数组索引对应分段索引，值为决策
        
        # 更新缓冲水位
        self.buffer_level += segment_time  # 增加缓冲水位

        # 详细日志输出：记录推入操作
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(push) name {} decision {} segment_time {:.2f}".format(
                self.name, decision, segment_time
            ))
            # 输出格式：缓冲区名称、决策、分段时长（保留2位小数）
        
        return  # 方法结束

    def get_rest(self):
        """
        获取当前分段的剩余处理时间：用于会话调度器确定增强时间
        这是一个查询方法，不修改缓冲区状态
        
        :return: 剩余时间（毫秒），如果没有分段则返回None
        """
        if len(self.content) > 0:  # 检查缓冲区是否有内容
            _, _, segment_time = self.content[0]  # 获取队首分段的时长
            return segment_time - self.time_spent  # 返回剩余时间
            # 公式：剩余时间 = 分段时长 - 已消费时间
        else:  # 缓冲区为空
            return None  # 返回None表示没有分段可处理


def test_buffer_model():
    """
    测试缓冲区模型：验证推入和消费功能的基本正确性
    这是一个简单的单元测试函数，演示缓冲区的使用
    """
    from logger import Logger  # 导入日志器类
    
    # 创建测试实例
    buffer = BufferModel(Logger(), 10, 5)  # 参数：日志器、最大水位10ms、5个分段
    
    def report():  # 定义内部报告函数
        print("level", buffer.buffer_level)  # 输出当前缓冲水位

    # 测试序列1：推入分段0
    buffer.push(0, 1, 1)  # 推入分段0，决策1，时长1ms
    report()  # 报告状态：level 1

    # 测试序列2：消费3ms（超过分段时长）
    buffer.consume(3)  # 消费3ms，会消费完分段0并产生2ms空转
    report()  # 报告状态：level 0

    # 测试序列3：推入分段1
    buffer.push(1, 2, 5)  # 推入分段1，决策2，时长5ms
    report()  # 报告状态：level 5

    # 测试序列4：消费3ms（部分消费）
    buffer.consume(3)  # 消费3ms，分段1剩余2ms
    report()  # 报告状态：level 2

    # 测试序列5：消费3ms（消费完剩余并产生空转）
    buffer.consume(3)  # 消费3ms，会消费完分段1并产生1ms空转
    report()  # 报告状态：level 0
    
    return  # 测试结束


if __name__ == '__main__':
    test_buffer_model()  # 当文件作为脚本直接运行时执行测试