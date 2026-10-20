import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import math
import numpy as np
import argparse
import os


# BONES（Buffer Occupancy-based Neural-Enhanced Streaming）算法实现
# 继承自 Simulator，提供具体的联动下载/增强控制策略，支持监控下载与参数自适应。
class BONES(Simulator):
    def __init__(self, args):
        """
        BONES (Buffer Occupancy-based Neural-Enhanced Streaming) Algorithm Simulator
        :param args: parameter settings
        """
        super(BONES, self).__init__(args)
        # BONES 参数：
        # gamma_p：质量奖励项的偏置；V：漂移-奖励权衡系数；is_avg_size：是否按平均分段大小归一化目标
        self.gamma_p = args.gamma_p
        self.V = (args.download_buffer * 1000 - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * args.V_multiplier
        self.is_avg_size = False

        # Quick Start 相关：当下载缓冲达到阈值时优先下载增强模型权重，解锁真实增强收益矩阵
        self.quick_start = args.quick_start
        self.model_size = 310 * 1000 * 8  # IMDN low+medium 模型大小（比特）
        self.model_ready = False
        self.download_threshold = 2 * self.seg_time
        if self.quick_start and ("imdn_bbb" not in self.args.enhance_path):
            raise ValueError("Quick start currently only supports content-aware IMDN.")

        # 监控下载：将网络模型替换为 MonitorNetworkModel，并注入 monitor_hook 与缓冲句柄
        self.monitor = args.monitor
        if args.monitor:
            self.network = MonitorNetworkModel(self.logger, self.net_time, self.bandwidth, self.latency,
                                                   self.monitor_hook, self.buffer_enhance, self.buffer_download)

        # 自动调参与查表：根据吞吐/方差/时延估计选择 gamma_p 与 V_multiplier
        self.autotune = args.autotune
        if args.autotune:
            self._load_autotune_table(args)

        # 限制增强方法池（示例：IMDN 屏蔽部分等级，以控制搜索空间与性能）
        if "imdn" in self.args.enhance_path:
            # self.vmaf_enh_avg[:, -1:] = -np.inf  # 屏蔽 low, medium, high 里的最高等级
            # self.vmaf_enh_avg[:, -2:] = -np.inf  # 也可只保留更少等级
            pass

        # Quick Start：初期禁用增强（仅低/中等级），待模型下载完毕再恢复增强收益
        if self.quick_start:
            self.old_vmaf = self.vmaf_enh_avg.copy()
            self.old_vmaf[:, -2:] = -np.inf  # 仅允许低和中等级
            self.vmaf_enh_avg[:, 1:] = -np.inf  # 暂时禁用所有增强



    def control_download_start(self):
        """
        下载会话开始：基于当前缓冲/估计状态选择 (action_down, action_enh)
        """
        # 计算联合动作（下载码率索引 action_down，增强方法索引 action_enh）
        self.action_down, self.action_enh = self._bones_control()

        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {}".format(
                self.action_down, self.action_enh
            ))
        return

    def _bones_control(self):
        # 自动调参：根据当前估计的带宽/方差/时延查表更新超参数
        if self.autotune:
            self._tune_params()

        # 漂移部分（更倾向降低缓冲积压）：下载缓冲与增强缓冲的“时间负载”
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        obj_drift = buff_down * self.seg_time + buff_enh * self.enhance_time  # (num_bitrate, num_method + 1)

        # 奖励部分（更倾向更高画质）：平均增强后 VMAF 加偏置项 gamma_p
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)

        # 归一化：按当前段大小或平均大小评估“单位比特”的性价比
        if self.is_avg_size:
            seg_size = self.bitrate[:, None] * self.seg_time
        else:
            seg_size = self.seg_size[self.idx_down][:, None]  # (num_bitrate, 1)

        # 目标函数：最小化 (漂移 - V*奖励) / 分段大小
        obj = (obj_drift - self.V * obj_reward) / seg_size  # (num_bitrate, num_method + 1)

        # 来不及增强的组合直接禁用（置为 +inf）：避免增强落后于下载播放进度
        mask_late = ((buff_enh + self.enhance_time - buff_down) > 0)  # (num_bitrate, num_method + 1)
        obj[mask_late] = math.inf

        # 选择最小目标值对应的 (action_down, action_enh)
        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        # Quick Start：当下载缓冲达到阈值且模型未就绪，附加下载模型权重并解锁真实增强收益
        if self.quick_start and (not self.model_ready) and buff_down >= self.download_threshold:
            self.extra_download = self.model_size
            self.model_ready = True
            self.vmaf_enh_avg = self.old_vmaf
        return action_down, action_enh



    def _tune_params(self):
        """
        自动参数调优：根据当前估计的带宽/方差/时延离散化到表格，选择 (gamma_p, V_multiplier)
        """
        # 预定义离散网格参数
        num_bandwidth = 25
        min_bandwidth = 200
        step_bandwidth = 200
        num_variance = 10
        min_variance = 500
        step_variance = 500
        num_latency = 2
        min_latency = 20
        step_latency = 80

        # 从吞吐历史估计当前带宽与时延，计算带宽方差
        bandwidth, latency = self.tput_history.estimate()
        variance = np.var(self.tput_history.throughput_window)

        # 将连续值映射到离散表索引
        def get_idx(val, min_val, step_val, num_val):
            if val < min_val:
                idx_val = 0
            else:
                idx_val = int((val - min_val) // step_val)
            idx_val = min(num_val - 1, idx_val)
            return idx_val
        idx_bandwidth = get_idx(bandwidth, min_bandwidth, step_bandwidth, num_bandwidth)
        idx_variance = get_idx(variance, min_variance, step_variance, num_variance)
        idx_latency = get_idx(latency, min_latency, step_latency, num_latency)

        # 查表更新 gamma_p 与 V，并重算 V
        self.gamma_p = self.gamma_p_table[idx_bandwidth, idx_variance, idx_latency]
        self.V_multiplier = self.V_multiplier_table[idx_bandwidth, idx_variance, idx_latency]
        self.V = (self.buffer_download.max_level - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * self.V_multiplier

        if self.logger.is_verbose():
            self.logger.write("(autotune) gamma_p {} V_multiplier {}".format(
                self.gamma_p, self.V_multiplier
            ))
        return

    def control_download_finish(self):
        """
        下载完成收尾：
        - 唤醒增强时间线（若空）
        - 判断该段增强是否来得及，不行则回退为不增强
        - 入列增强任务并更新 idx_enh
        - 计算下载等待时间，避免缓冲接近溢出
        """
        # 若增强缓冲为空，唤醒增强
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        # 检查增强是否能按时完成（避免增强落后播放）
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            self.action_enh = 0

        # 若最终仍需要增强，则将该增强任务推入增强缓冲
        if self.action_enh != 0:
            self.buffer_enhance.push(segment_index=self.idx_enh, decision=self.action_enh,
                                 segment_time=self.enhance_time[self.action_down][self.action_enh])
        self.idx_enh += 1
        # 计算等待时间，避免下载缓冲溢出：buffer_level + seg_time 超过 max_level 时等待
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)
        return

    def control_enhance_start(self):
        """
        增强会话开始：BONES 默认不改动（增强任务及时长来自增强缓冲队首）
        """
        return

    def control_enhance_finish(self):
        """
        增强会话结束：BONES 默认不做额外处理
        """
        return

    def monitor_hook(self, buffer_download, buffer_enhance, throughput, latency, remain_size, total_time):
        """
        下载过程监控回调：判断是否中止当前下载并切换到更优的 (action_down, action_enh)
        :param buffer_download: 当前“想象的”下载缓冲水位
        :param buffer_enhance: 当前“想象的”增强缓冲水位
        :param throughput: 估计吞吐
        :param latency: 估计时延
        :param remain_size: 当前选项剩余大小
        :param total_time: 累计下载时间（含首包）
        :return: 新的分段大小（表示切换选项），或 None（继续当前）
        """
        # 计算当前选项的目标值（以剩余大小归一化，越小越好）
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time[self.action_down][self.action_enh]
        obj_reward = self.vmaf_enh_avg[self.action_down][self.action_enh] + self.gamma_p
        score_now = (obj_drift - self.V * obj_reward) / remain_size

        # 计算其他所有可选项的目标值，并掩蔽“来不及增强”的组合
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time  # (num_bitrate, num_method + 1)
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)
        seg_size = self.seg_size[self.idx_down][:, None]  # (num_bitrate, 1)
        score_other = (obj_drift - self.V * obj_reward) / seg_size  # (num_bitrate, num_method + 1)
        mask_late = ((buffer_enhance + self.enhance_time - buffer_download) > 0)  # (num_bitrate, num_method + 1)
        score_other[mask_late] = math.inf
        decision = score_other.argmin()
        action_down = decision // score_other.shape[1]
        action_enh = decision % score_other.shape[1]

        # 若发现更优且与当前不同的组合，则请求切换（MonitorNetworkModel 将据此重启下载并计入新一次首包时延）
        if (action_down != self.action_down) and (action_enh != self.action_enh) and (score_other[action_down, action_enh] < score_now):
            self.action_down = action_down
            self.action_enh = action_enh
            new_size = self.seg_size[self.idx_down][self.action_down]
            if self.logger.is_verbose():
                self.logger.write("(monitor) download {} enhance {}".format(
                    self.action_down, self.action_enh
                ))
            return new_size
        return None

    def _load_autotune_table(self, args):
        # 根据增强路径选择对应的离线参数表（.npz），加载 gamma_p 与 V_multiplier 查找表
        path2name = {
            "../data/bbb/nas_1080ti.json": "nas",
            "../data/bbb/imdn_div2k_2080ti.json": "div2080",
            "../data/bbb/imdn_bbb_2080ti.json": "bbb2080",
            "../data/bbb/imdn_div2k_3060ti.json": "div3060",
            "../data/bbb/imdn_bbb_3060ti.json": "bbb3060",
        }
        abs_path = os.path.dirname(os.path.abspath(__file__)) + "/"
        tables = np.load(abs_path + f"autotune_bones_{path2name[args.enhance_path]}.npz")
        self.gamma_p_table, self.V_multiplier_table = tables["gamma_p_table"], tables["V_multiplier_table"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulator for neural-enhanced streaming system.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="bones_test.txt",
                        help='Path the logging file.')
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json',
                        help='Specify the .json file describing the video manifest.')
    parser.add_argument('--trace_path', default='../data/trace/dummy.json',
                        help='Specify the .json file describing the network trace.')
    parser.add_argument('--enhance_path', default='../data/bbb/nas_1080ti.json',
                        help='Specify the .json file describing the enhancement performance.')
    parser.add_argument('--video_length', default=None,
                        help='Customized video length, measured in seconds.')
    parser.add_argument('--throughput', default='ewma',
                        help='Throughput estimator. Dual exponential window moving average by default.')
    parser.add_argument('-db', '--download_buffer', default=60, type=int,
                        help='Maximum download buffer size, measured in seconds.')
    parser.add_argument('-eb', '--enhance_buffer', default=60, type=int,
                        help='Maximum enhance buffer size, measured in seconds.')
    parser.add_argument('-bm', '--bandwidth_multiplier', default=1, type=float,
                        help='Multiplier for the network bandwidth.')
    parser.add_argument('-sm', '--speed_multiplier', default=1, type=float,
                        help='Multiplier for the enhancement speed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print verbose information.')

    parser.add_argument('-gp', '--gamma_p', type=float, default=10,
                        help='BONES gamma*p parameter.')
    parser.add_argument('-vm', '--V_multiplier', type=float, default=1,
                        help='BONES gamma*p parameter.')
    parser.add_argument('-m', '--monitor', action='store_true',
                        help='Monitor download process.')
    parser.add_argument('-a', '--autotune', action='store_true',
                        help='Automatic parameter tuning.')
    parser.add_argument('-q', '--quick_start', action='store_true',
                        help='Quick start.')


    args = parser.parse_args()


    simulator = BONES(args)
    simulator.run()
    report = simulator.report()
