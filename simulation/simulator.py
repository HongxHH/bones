import sys  # 导入系统模块，用于路径操作
sys.path.append("..")  # 将上级目录添加到Python路径，用于导入其他模块
import json  # 导入JSON模块，用于解析配置文件
import numpy as np  # 导入numpy数值计算库
import argparse  # 导入命令行参数解析模块
import math  # 导入数学函数模块
# 导入simulation子模块
from .tput_history import SlidingWindow, ExponentialWindow, HarmonicMean  # 导入吞吐量估计器
from .network_model import NetworkModel  # 导入基础网络模型
from .buffer_model import BufferModel  # 导入缓冲区模型
from .session_model import SessionModel  # 导入会话模型
from .logger import Logger  # 导入日志器


class Simulator():
    def __init__(self, args):
        """
        神经增强流媒体仿真器父类构造函数：负责解析输入、搭建网络/缓冲/会话/吞吐估计器，
        并提供控制钩子（下载开始/结束、增强开始/结束）供子类策略实现
        
        :param args: 参数设置对象，包含所有仿真配置参数
        """
        # 初始化成员变量（视频、网络、增强、质量、时间与决策索引等）
        self.args = args  # 保存参数设置对象，供后续使用
        
        # 视频相关参数
        self.seg_time = None  # 分段时长（毫秒），单个视频分段的持续时间
        self.bitrate = None  # 码率梯级数组 (num_bitrate,)，可用的码率选项
        self.seg_size = None  # 分段大小矩阵 (num_segment, num_bitrate)，每个分段在各码率下的大小（比特）
        
        # 网络相关参数
        self.net_time = None  # 网络周期时长数组 (num_period,)，每个网络周期的持续时间
        self.bandwidth = None  # 各周期带宽数组 (num_period,)，每个周期的传输速率（kbps）
        self.latency = None  # 各周期时延数组 (num_period,)，每个周期的网络延迟（ms）
        
        # 增强相关参数
        self.enhance_time = None  # 增强耗时矩阵 (num_bitrate, num_method + 1)，各码率-方法组合的增强时间
        
        # 质量相关参数（VMAF指标）
        self.vmaf_down_seg = None  # 下载质量（VMAF）分段级 (num_segment, num_bitrate)，每个分段的下载质量
        self.vmaf_enh_seg = None  # 增强后质量（VMAF）分段级 (num_segment, num_bitrate, num_method + 1)，每个分段的增强质量
        self.vmaf_down_avg = None  # 下载质量（VMAF）平均 (num_bitrate,)，各码率的平均下载质量
        self.vmaf_enh_avg = None  # 增强后质量（VMAF）平均 (num_bitrate, num_method + 1)，各码率-方法组合的平均增强质量
        
        # 质量相关参数（SSIM指标）
        self.ssim_down_seg = None  # 下载质量（SSIM）分段级，每个分段的下载质量
        self.ssim_enh_seg = None  # 增强后质量（SSIM）分段级，每个分段的增强质量
        
        # 全局状态变量
        self.global_time = 0  # 全局时间戳（毫秒），仿真开始后的累计时间
        self.idx_down = 0  # 等待下载决策的分段索引，指向下一个要下载的分段
        self.idx_enh = 0  # 等待增强决策的分段索引，指向下一个要增强的分段
        
        # 下载相关状态
        self.last_download = None  # 上一次下载的信息 (download_time, throughput, latency)
        self.action_down = math.inf  # 下一个下载动作（码率索引），默认无效值
        self.action_enh = 0  # 下一个增强动作（方法索引），默认0表示不增强
        self.extra_download = 0  # 附加下载大小（比特），用于中途切换/模型权重下载等

        # 解析输入文件：清单、网络轨迹与（可选）增强性能
        self._parse_manifest(args.manifest_path, args.video_length)  # 解析视频清单文件
        self._parse_trace(args.trace_path, args.bandwidth_multiplier)  # 解析网络轨迹文件
        
        if args.enhance_path == '':  # 检查是否提供了增强文件
            # 若未提供增强文件，则用码率对数近似质量，并设增强开销为0
            self._estimate_quality()
        else:
            # 解析增强性能文件
            self._parse_enhance(args.enhance_path, args.speed_multiplier)
        
        # 日志器（可带时间戳），支持verbose输出到控制台与文件
        self.logger = Logger(args.log_path, args.verbose, self.get_timestamp)

        # 吞吐估计器：支持ewma（默认）、滑动窗口sw、调和平均hm
        if args.throughput == "ewma":  # 指数加权移动平均
            self.tput_history = ExponentialWindow(self.logger, self.seg_time)
        elif args.throughput == "sw":  # 滑动窗口
            self.tput_history = SlidingWindow(self.logger)
        elif args.throughput == "hm":  # 调和平均
            self.tput_history = HarmonicMean(self.logger)
        else:
            raise NotImplementedError  # 不支持的估计器类型

        # 网络模型：基于离线轨迹的分段常值带宽/时延仿真
        self.network = NetworkModel(self.logger, self.net_time, self.bandwidth, self.latency)

        # 两个缓冲区：下载缓冲与增强缓冲，容量以毫秒计（由秒×1000传入）
        self.buffer_download = BufferModel(self.logger, args.download_buffer * 1000, self.seg_size.shape[0], "download")
        self.buffer_enhance = BufferModel(self.logger, args.enhance_buffer * 1000, self.seg_size.shape[0], "enhance")

        # 会话模型：管理下载/增强两条时间线的等待与处理会话切换
        self.session = SessionModel(self.logger)

    def run(self):
        """
        仿真主循环：在下载与增强两条时间线上交替推进；
        每次推进到下一个会话边界（下载开始/结束、增强开始/结束），由子类控制钩子决定动作
        
        这是仿真器的核心方法，实现双时间线的协调调度和状态管理
        """
        # 预热阶段：下载第0段（最低码率、无增强），更新吞吐历史并清零起播卡顿统计
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(warmup_stage)")  # 输出预热阶段开始标记
        self.warmup_stage()  # 执行预热阶段

        # 主阶段：直到所有分段的下载与增强都完成
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(main_stage)")  # 输出主阶段开始标记
        
        while self.is_running():  # 循环直到仿真结束
            # 取下一会话类型与当前会话剩余时间
            idx_session, time_session = self.session.next()

            # 推进时间：全局时间、两个缓冲的消费；若当前不在下载态，也推进网络时间线
            self.global_time += time_session  # 更新全局时间戳
            self.buffer_download.consume(time_session)  # 消费下载缓冲内容
            self.buffer_enhance.consume(time_session)  # 消费增强缓冲内容
            
            if not self.session.is_downloading:  # 检查当前是否在下载状态
                # 非下载态下，网络时间也在流逝
                self.network.go_by(time_session)

            # 会话决策与切换
            if idx_session == self.session.DOWNLOAD_START:  # 下载开始会话
                # 结束等待，开始下载：由子类决定下载/增强动作
                self.control_download_start()  # 调用子类实现的下载开始控制
                if not self.is_running():  # 检查仿真是否应该继续
                    break  # 如果仿真结束，退出循环
                
                # 启动新的下载任务（分段大小 + 可能的附加下载），并清空附加
                size = self.seg_size[self.idx_down, self.action_down] + self.extra_download  # 计算总下载大小
                self.extra_download = 0  # 清空附加下载大小
                self.last_download = self.network.download(size)  # 执行网络下载
                
                # 设置新的下载会话倒计时（下载时间 + 首包时延）
                cd_download = self.last_download[0] + self.last_download[2]  # 总下载时间
                self.session.set_download(True, cd_download)  # 设置下载状态和倒计时

            elif idx_session == self.session.DOWNLOAD_FINISH:  # 下载结束会话
                # 下载完成，转入等待：记录吞吐观测并把下载决策写入下载缓冲
                self.tput_history.push(*self.last_download)  # 将下载信息推入吞吐历史
                assert not math.isinf(self.action_down)  # 断言检查下载动作有效
                self.buffer_download.push(segment_index=self.idx_down, decision=self.action_down, segment_time=self.seg_time)  # 将分段推入下载缓冲

                self.idx_down += 1  # 更新下载索引
                # 避免缓冲溢出：默认立刻可下载下一段，具体等待由子类控制下载结束时决定
                self.session.set_download(False, 0)  # 设置下载为等待状态
                self.control_download_finish()  # 调用子类实现的下载结束控制
                if not self.is_running():  # 检查仿真是否应该继续
                    break  # 如果仿真结束，退出循环

            elif idx_session == self.session.ENHANCE_START:  # 增强开始会话
                # 结束等待，开始增强：由子类可选地调整增强动作
                self.control_enhance_start()  # 调用子类实现的增强开始控制
                if not self.is_running():  # 检查仿真是否应该继续
                    break  # 如果仿真结束，退出循环
                
                # 新的增强会话：若增强缓冲有段，则用其剩余时长作为倒计时，否则进入"无穷等待"
                cd_enhance = self.buffer_enhance.get_rest()  # 获取增强缓冲剩余时间
                if cd_enhance is not None:  # 检查是否有分段可增强
                    self.session.set_enhance(True, cd_enhance)  # 设置增强状态和倒计时
                else:
                    # 无增强任务则休眠，直至被下载完成唤醒
                    self.session.set_enhance(False, math.inf)  # 设置增强为无穷等待

            else:  # idx_session == self.session.ENHANCE_FINISH，增强结束会话
                # 增强完成，转入等待：立刻允许增强下一段
                self.session.set_enhance(False, 0)  # 设置增强为等待状态
                self.control_enhance_finish()  # 调用子类实现的增强结束控制
                if not self.is_running():  # 检查仿真是否应该继续
                    break  # 如果仿真结束，退出循环

            # 每轮推进后校验：缓冲不溢出、增强不领先下载
            self.validate()  # 执行状态校验

        # 最终阶段：把下载缓冲中剩余内容"播放完"，推进相同时间到增强缓冲
        self.final_stage()  # 执行最终阶段
        return  # 方法结束

    def report(self, title=""):
        """
        汇总性能指标：平均VMAF/SSIM、振荡、卡顿时间/事件数、平均带宽、QoE，以及决策历史
        
        :param title: 报告标题，用于标识不同的报告
        :return: 指标字典，包含所有性能指标
        """
        # 获取决策历史
        history_down = self.buffer_download.history  # 下载决策历史
        history_enh = self.buffer_enhance.history  # 增强决策历史

        # 逐段复盘决策并累计指标
        num_seg = self.seg_size.shape[0]  # 分段总数
        quality_vmaf = np.zeros((num_seg, ))  # VMAF质量数组
        oscillation_vmaf = 0  # VMAF振荡累计
        quality_ssim = np.zeros((num_seg, ))  # SSIM质量数组
        oscillation_ssim = 0  # SSIM振荡累计
        total_bandwidth = np.zeros((1, ), dtype=np.int64)  # 总带宽累计

        # 遍历所有分段，计算质量指标
        for i in range(num_seg):  # 遍历每个分段
            action_down = history_down[i]  # 获取分段i的下载决策
            action_enh = history_enh[i]  # 获取分段i的增强决策
            quality_vmaf[i] = self.vmaf_enh_seg[i][action_down][action_enh]  # 计算分段i的VMAF质量
            quality_ssim[i] = self.ssim_enh_seg[i][action_down][action_enh]  # 计算分段i的SSIM质量
            
            if i > 0:  # 从第二个分段开始计算振荡
                oscillation_vmaf += abs(quality_vmaf[i] - quality_vmaf[i - 1])  # 累加VMAF振荡
                oscillation_ssim += abs(quality_ssim[i] - quality_ssim[i - 1])  # 累加SSIM振荡
            
            total_bandwidth += self.seg_size[i][action_down]  # 累加带宽使用

        # 计算平均指标
        quality_vmaf = np.mean(quality_vmaf)  # 平均VMAF质量
        quality_ssim = np.mean(quality_ssim)  # 平均SSIM质量
        oscillation_vmaf = oscillation_vmaf / (num_seg - 1)  # 平均VMAF振荡
        oscillation_ssim = oscillation_ssim / (num_seg - 1)  # 平均SSIM振荡
        avg_freeze = self.buffer_download.idle_time / self.seg_size.shape[0]  # 平均卡顿时长
        avg_bandwidth = total_bandwidth / num_seg / self.seg_time  # 平均带宽

        # 计算QoE（Quality of Experience）
        qoe = quality_vmaf - oscillation_vmaf - avg_freeze / 10  # QoE公式

        # 构建报告字典
        report_simulator = {
            "playback time": self.global_time,  # 播放时间
            "average quality vmaf": quality_vmaf,  # 平均VMAF质量
            "average oscillation vmaf": oscillation_vmaf,  # 平均VMAF振荡
            "average quality ssim": quality_ssim,  # 平均SSIM质量
            "average oscillation ssim": oscillation_ssim,  # 平均SSIM振荡
            "freeze time": self.buffer_download.idle_time,  # 卡顿总时间
            "freeze event": self.buffer_download.idle_event,  # 卡顿事件数
            "average bandwidth": avg_bandwidth,  # 平均带宽
            "qoe": qoe,  # QoE指标
            "download history": history_down,  # 下载决策历史
            "enhance history": history_enh  # 增强决策历史
        }
        
        # 输出报告
        if self.logger.is_verbose():  # 检查是否开启详细日志
            title = "Simulator Report" if title == "" else title  # 设置默认标题
            self.logger.summarize(title, report_simulator)  # 输出汇总报告
        
        return report_simulator  # 返回报告字典

    def control_enhance_finish(self):
        """
        增强结束控制钩子：由子类实现，默认未实现
        子类可以在此方法中实现增强结束后的处理逻辑
        """
        raise NotImplementedError  # 强制子类实现此方法

    def control_enhance_start(self):
        """
        增强开始控制钩子：由子类实现，默认未实现
        子类可以在此方法中实现增强开始前的处理逻辑
        """
        raise NotImplementedError  # 强制子类实现此方法

    def control_download_finish(self):
        """
        下载结束控制钩子：由子类实现，默认未实现
        子类可以在此方法中实现下载结束后的处理逻辑
        """
        raise NotImplementedError  # 强制子类实现此方法

    def control_download_start(self):
        """
        下载开始控制钩子：由子类实现，默认未实现
        子类可以在此方法中实现下载开始前的处理逻辑
        """
        raise NotImplementedError  # 强制子类实现此方法

    def validate(self):
        """
        校验仿真状态：确保系统状态的一致性
        - 下载缓冲不得溢出
        - 增强缓冲不得溢出
        - 增强缓冲水位不得领先下载缓冲
        
        :return: 合法返回True，否则抛出异常
        """
        # 检查下载缓冲溢出
        if self.buffer_download.buffer_level > self.buffer_download.max_level + 1E-6:  # 允许小的浮点误差
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Download buffer overflow." + error_msg)  # 抛出下载缓冲溢出异常
        
        # 检查增强缓冲溢出
        if self.buffer_enhance.buffer_level > self.buffer_enhance.max_level + 1E-6:  # 允许小的浮点误差
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Enhance buffer overflow." + error_msg)  # 抛出增强缓冲溢出异常
        
        # 检查增强缓冲是否领先下载缓冲（逻辑错误）
        if self.buffer_enhance.buffer_level > self.buffer_download.buffer_level + 1E-6:  # 允许小的浮点误差
            error_msg = f" Enhancement: {self.args.enhance_path}, Trace: {self.args.trace_path}, Download Buffer: {self.buffer_download.buffer_level}, Enhancement Buffer: {self.buffer_enhance.buffer_level}"
            raise RuntimeError("Late computation." + error_msg)  # 抛出计算延迟异常
        
        return True  # 所有检查通过，返回True

    def final_stage(self):
        """
        结束阶段：播放/处理下载缓冲剩余的内容（假设增强缓冲不超过下载缓冲），
        推进等量时间并清空水位
        
        这个阶段确保所有已下载的内容都被"播放"完成
        """
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(final_stage)")  # 输出最终阶段开始标记

        # 清空两个缓冲（假设增强缓冲水位 <= 下载缓冲）
        final_time = self.buffer_download.buffer_level  # 获取下载缓冲剩余时间
        self.global_time += final_time  # 更新全局时间
        self.buffer_download.consume(final_time)  # 消费下载缓冲剩余内容
        self.buffer_enhance.consume(final_time)  # 消费增强缓冲剩余内容
        
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(clear) buffer {:.2f}".format(final_time))  # 输出清空缓冲信息
        
        return  # 方法结束

    def warmup_stage(self):
        """
        预热：下载第0段的最低码率且不增强，用于初始化吞吐估计，
        并将起播延迟不计入卡顿统计
        
        这个阶段确保系统有一个良好的初始状态
        """
        if self.logger.is_verbose():  # 检查是否开启详细日志
            self.logger.write("(control) download 0 enhance 0")  # 输出预热控制信息

        # 设置预热参数
        self.action_down = 0  # 下载第0段（最低码率）
        self.action_enh = 0  # 增强第0段（不增强）
        size = self.seg_size[0][0]  # 获取第0段最低码率的大小
        
        # 执行预热下载
        delay_down, tput, delay_lat = self.network.download(size)  # 下载第0段
        self.tput_history.push(delay_down, tput, delay_lat)  # 将下载信息推入吞吐历史
        total_time = delay_down + delay_lat  # 计算总时间
        
        # 更新系统状态
        self.global_time += total_time  # 更新全局时间
        self.buffer_download.consume(total_time)  # 消费下载缓冲
        self.buffer_enhance.consume(total_time)  # 消费增强缓冲
        self.buffer_download.push(0, 0, self.seg_time)  # 将第0段推入下载缓冲
        
        # 不将起播时延计入卡顿
        self.buffer_download.idle_time = 0  # 清零下载缓冲空转时间
        self.buffer_enhance.idle_event = 0  # 清零增强缓冲空转事件
        
        # 更新索引
        self.idx_down = 1  # 下一个要下载的分段索引
        self.idx_enh = 1  # 下一个要增强的分段索引
        
        return  # 方法结束

    def _estimate_quality(self):
        """
        无增强文件时的质量估计：以对数码率近似质量，并与最低码率对齐，增强时间置0
        
        当没有提供增强性能文件时，使用简单的对数模型估计质量
        """
        # 计算下载质量（基于码率对数）
        quality_down_avg = np.log(self.bitrate)  # 计算各码率的对数
        self.vmaf_down_avg = quality_down_avg - quality_down_avg[0]  # 以最低码率为基准对齐
        
        # 计算分段级下载质量
        quality_down_seg = np.log(self.seg_size / self.seg_time)  # 计算各分段的对数
        self.vmaf_down_seg = quality_down_seg - quality_down_avg[0]  # 以最低码率为基准对齐
        
        # 增强耗时与提升置零
        self.enhance_time = np.zeros((1, ))  # 增强时间设为0
        self.vmaf_enh_avg = self.vmaf_down_avg[:, None]  # 增强质量等于下载质量
        self.vmaf_enh_seg = self.vmaf_down_seg[:, None]  # 增强质量等于下载质量
        
        return  # 方法结束

    def _parse_enhance(self, enhance_path, speed_multiplier=1):
        """
        解析增强性能文件：计算增强耗时矩阵并拼接"无增强"方法；
        解析分段级质量（VMAF/SSIM），最高码率视为满分并禁止增强
        
        :param enhance_path: 增强性能文件路径
        :param speed_multiplier: 增强速度倍率，用于调整增强时间
        """
        data = self.load_json(enhance_path)  # 加载增强性能文件
        frame_rate = data['frame_rate']  # 获取帧率
        enhance_speed = np.array(data['enhancement_fps'])  # 获取增强速度数组 (num_bitrate - 1, num_method)
        num_bitrate = enhance_speed.shape[0] + 1  # 计算码率总数
        num_method = enhance_speed.shape[1]  # 获取增强方法数

        def parse_quality(metric):
            """
            解析质量指标的内部函数
            
            :param metric: 质量指标名称（"vmaf"或"ssim"）
            :return: (下载质量, 增强质量) 元组
            """
            if metric == "ssim":  # SSIM指标
                max_score = 1  # 最大分数为1
            else:  # VMAF指标
                max_score = 100  # 最大分数为100
            
            # 解析下载质量
            quality_down_seg = np.array(data[f'base_quality_{metric}'])  # 获取基础质量 (num_bitrate - 1, num_segment)
            quality_enh_seg = np.array(data[f'enhanced_quality_{metric}'])  # 获取增强质量 (num_bitrate_enh, num_method, num_segment)
            num_bitrate_enh = quality_enh_seg.shape[0]  # 获取增强码率数
            num_segment = quality_down_seg.shape[1]  # 获取分段数

            # 最高码率得分设为满分，并禁止其增强（增强质量置-inf）
            quality_down_seg = quality_down_seg.transpose()  # 转置为 (num_segment, num_bitrate - 1)
            quality_down_seg = np.concatenate([quality_down_seg, np.full((num_segment, 1), max_score)], axis=1)  # 拼接最高码率满分
            
            # 处理增强质量
            quality_enh_seg = quality_enh_seg.transpose((2, 0, 1))  # 转置为 (num_segment, num_bitrate_enh, num_method)
            quality_enh_seg[quality_enh_seg < 0] = -math.inf  # 将负值设为无效增强
            quality_enh_seg = np.concatenate([quality_enh_seg, np.full((num_segment, num_bitrate - num_bitrate_enh, num_method), -math.inf)], axis=1)  # 拼接缺失码率
            quality_enh_seg = np.concatenate([quality_down_seg[:, :, None], quality_enh_seg], axis=2)  # 拼接无增强选项
            
            return quality_down_seg, quality_enh_seg  # 返回质量数组

        # 方法索引0代表"不增强"；增强耗时矩阵拼接"不增强"为0
        enhance_time = frame_rate * self.seg_time / enhance_speed / speed_multiplier  # 计算增强时间 (num_bitrate - 1, num_method)
        enhance_time[enhance_time < 0] = 0  # 将负值设为0
        enhance_time = np.concatenate([enhance_time, np.full((1, num_method), 0)], axis=0)  # 拼接最高码率（无增强）
        enhance_time = np.concatenate([np.full((num_bitrate, 1), 0), enhance_time], axis=1)  # 拼接无增强选项

        # 质量张量（SSIM/VMAF）与平均质量
        self.ssim_down_seg, self.ssim_enh_seg = parse_quality('ssim')  # 解析SSIM质量
        self.vmaf_down_seg, self.vmaf_enh_seg = parse_quality('vmaf')  # 解析VMAF质量
        self.vmaf_down_avg = np.mean(self.vmaf_down_seg, axis=0)  # 计算平均VMAF质量
        self.vmaf_enh_avg = np.mean(self.vmaf_enh_seg, axis=0)  # 计算平均增强VMAF质量

        self.enhance_time = enhance_time  # 保存增强时间矩阵
        return  # 方法结束

    def _parse_trace(self, trace_path, bandwidth_multiplier=1):
        """
        解析网络轨迹：读取每个周期的时长/带宽/时延，带宽可乘倍率以模拟不同环境
        
        :param trace_path: 轨迹文件路径
        :param bandwidth_multiplier: 带宽倍率，用于调整网络条件
        :return: 三个numpy向量
        """
        data = self.load_json(trace_path)  # 加载轨迹文件
        net_time = []  # 网络周期时长列表
        bandwidth = []  # 网络周期带宽列表
        latency = []  # 网络周期时延列表
        
        # 遍历轨迹数据，提取网络参数
        for i in range(len(data)):  # 遍历轨迹数据
            net_time.append(data[i]['duration_ms'])  # 添加周期时长
            bandwidth.append(data[i]['bandwidth_kbps'])  # 添加周期带宽
            latency.append(data[i]['latency_ms'])  # 添加周期时延
        
        # 转换为numpy数组并应用倍率
        net_time = np.array(net_time)  # 转换为numpy数组
        bandwidth = np.array(bandwidth) * bandwidth_multiplier  # 应用带宽倍率
        latency = np.array(latency)  # 转换为numpy数组

        # 保存到实例变量
        self.net_time = net_time  # 保存网络周期时长
        self.bandwidth = bandwidth  # 保存网络周期带宽
        self.latency = latency  # 保存网络周期时延
        
        return  # 方法结束

    def _parse_manifest(self, manifest_path, video_length):
        """
        解析视频清单：分段时长、码率梯级与每段在各码率的大小
        
        :param manifest_path: 清单路径
        :param video_length: 自定义视频时长（暂未启用截断/扩展）
        """
        manifest = self.load_json(manifest_path)  # 加载清单文件
        seg_time = manifest['segment_duration_ms']  # 获取分段时长
        bitrate = manifest['bitrates_kbps']  # 获取码率梯级
        bitrate = np.array(bitrate)  # 转换为numpy数组
        seg_size = manifest['segment_sizes_bits']  # 获取分段大小
        # 目前固定使用完整清单；未来可按video_length截断/扩展
        assert video_length is None  # 断言检查，确保video_length为None
        seg_size = np.array(seg_size)  # 转换为numpy数组

        # 保存到实例变量
        self.seg_time = seg_time  # 保存分段时长
        self.bitrate = bitrate  # 保存码率梯级
        self.seg_size = seg_size  # 保存分段大小
        
        return  # 方法结束

    def is_running(self):
        """
        判断仿真是否继续：下载与增强索引均未越过总分段数时继续
        
        :return: 如果仿真应该继续返回True，否则返回False
        """
        num_seg = self.seg_size.shape[0]  # 获取分段总数
        if (self.idx_down < num_seg) and (self.idx_enh < num_seg):  # 检查两个索引是否都未越界
            return True  # 仿真继续
        return False  # 仿真结束

    @staticmethod
    def load_json(path):
        """
        加载JSON文件的静态方法
        
        :param path: JSON文件路径
        :return: 解析后的JSON对象
        """
        with open(path) as file:  # 打开文件
            obj = json.load(file)  # 解析JSON
        return obj  # 返回解析结果

    def get_timestamp(self):
        """
        生成用于日志的时间戳串，格式为 [t<ms>-d<idx_down>-e<idx_enh>]
        
        :return: 时间戳字符串
        """
        timestamp = ""  # 初始化时间戳字符串
        timestamp += "[t" + str(round(self.global_time)) + "-"  # 添加全局时间
        timestamp += "d" + str(self.idx_down) + "-"  # 添加下载索引
        timestamp += "e" + str(self.idx_enh) + "] "  # 添加增强索引
        return timestamp  # 返回时间戳


if __name__ == '__main__':
    """
    命令行入口：当文件作为脚本直接运行时执行
    创建参数解析器并运行仿真
    """
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加命令行参数
    parser.add_argument('--log_path', default=None, help='Path the logging file.')
    parser.add_argument('--manifest_path', default='movie.json', help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='network.json', help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='', help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None, help='Customized video length, measured in seconds.')
    parser.add_argument('--throughput', default='ewma', help='Throughput estimator. Dual exponential window moving average by default.')
    parser.add_argument('--download_buffer', default=25, help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('--enhance_buffer', default=25, help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('--bandwidth_multiplier', default=1, type=float, help='Multiplier for the network bandwidth.')
    parser.add_argument('--speed_multiplier', default=1, type=float, help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true', help='Print verbose information.')
    
    args = parser.parse_args()  # 解析命令行参数

    # 当本文件作为脚本执行时，构造基础Simulator（父类不直接run，示例仅演示参数解析）
    simulator = Simulator(args)  # 创建仿真器实例