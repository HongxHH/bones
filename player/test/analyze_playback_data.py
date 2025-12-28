"""
播放数据分析可视化脚本

该脚本用于分析 analyzer.py 生成的 JSON 数据文件，并生成综合的可视化图表。
图表包含：
- 缓冲水位折线
- 卡顿时间段标记
- 段的下载/增强决策信息
- 增强时间信息
- 中止原因标记
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import platform

# 配置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 如果在Windows系统上，尝试设置中文字体
if platform.system() == 'Windows':
    try:
        # 尝试使用Windows系统字体
        import matplotlib.font_manager as fm
        # 查找可用的中文字体
        font_list = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
        for font in chinese_fonts:
            if font in font_list:
                plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
                break
    except Exception:
        pass


class PlaybackDataAnalyzer:
    """播放数据分析器"""
    
    def __init__(self, json_path: str):
        """
        初始化分析器
        
        Args:
            json_path: JSON 数据文件路径
        """
        self.json_path = json_path
        self.data: Dict = {}
        self.start_time: Optional[float] = None  # 播放开始时间（绝对时间戳）
        
    def load_data(self):
        """加载 JSON 数据"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 计算播放开始时间（使用第一个段的开始时间作为基准）
        if self.data.get('segments') and len(self.data['segments']) > 0:
            first_segment = self.data['segments'][0]
            if first_segment.get('start_time'):
                self.start_time = first_segment['start_time']
    
    def absolute_to_relative_time(self, absolute_time: Optional[float]) -> Optional[float]:
        """
        将绝对时间戳转换为相对时间（秒）
        
        Args:
            absolute_time: 绝对时间戳
            
        Returns:
            相对时间（秒），如果输入为 None 则返回 None
        """
        if absolute_time is None or self.start_time is None:
            return None
        return absolute_time - self.start_time
    
    def plot_analysis(self, output_path: Optional[str] = None):
        """
        绘制综合分析图表
        
        Args:
            output_path: 输出文件路径，如果为 None 则显示图表
        """
        fig = plt.figure(figsize=(30, 12))
        # # 计算数据的时间范围，动态调整图表宽度
        # max_time = 0
        # # 从buffer_level获取最大时间
        # buffer_levels = self.data.get('buffer_level', [])
        # if buffer_levels:
        #     max_time = max(max_time, max(bl['time'] for bl in buffer_levels))
        # # 从segments获取最大时间
        # segments = self.data.get('segments', [])
        # if segments:
        #     for seg in segments:
        #         seg_stop = self.absolute_to_relative_time(seg.get('stop_time'))
        #         if seg_stop is not None:
        #             max_time = max(max_time, seg_stop)
        
        # # 根据时间范围动态计算宽度：每100秒约15英寸，最小30英寸
        # fig_width = max(30, int(max_time / 100 * 15) + 10)
        
        # # 创建图表和子图（动态宽度以避免重叠）
        # fig = plt.figure(figsize=(fig_width, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1], hspace=0.3)
        
        # 主图：缓冲水位 + 卡顿 + 段信息
        ax_main = fig.add_subplot(gs[0])
        
        # 子图1：下载决策
        ax_download = fig.add_subplot(gs[1], sharex=ax_main)
        
        # 子图2：增强决策和状态
        ax_enhance = fig.add_subplot(gs[2], sharex=ax_main)
        
        # 子图3：增强时间信息
        ax_enhance_time = fig.add_subplot(gs[3], sharex=ax_main)
        
        # ===== 绘制缓冲水位 =====
        self._plot_buffer_level(ax_main)
        
        # ===== 绘制卡顿区域 =====
        self._plot_stalls(ax_main)
        
        # ===== 绘制段信息 =====
        self._plot_segments_info(ax_main, ax_download, ax_enhance, ax_enhance_time)
        
        # 设置主图标签和标题
        ax_main.set_ylabel('缓冲水位 (秒)', fontsize=12, color='blue')
        ax_main.set_title('播放数据分析 - 缓冲水位、卡顿与段信息', fontsize=14, fontweight='bold')
        ax_main.grid(True, alpha=0.3)
        ax_main.legend(loc='upper left', fontsize=9)
        
        # 设置子图标签
        ax_download.set_ylabel('下载决策', fontsize=10)
        ax_download.set_title('段下载决策 (download_action)', fontsize=11)
        ax_download.grid(True, alpha=0.3)
        ax_download.legend(loc='upper right', fontsize=8)
        
        ax_enhance.set_ylabel('增强决策', fontsize=10)
        ax_enhance.set_title('段增强决策与状态 (enhance_action)', fontsize=11)
        ax_enhance.grid(True, alpha=0.3)
        ax_enhance.legend(loc='upper right', fontsize=8)
        
        ax_enhance_time.set_xlabel('播放时间 (秒)', fontsize=12)
        ax_enhance_time.set_ylabel('时间差 (秒)', fontsize=10)
        ax_enhance_time.set_title('增强完成到播放的时间 (enhance_end_to_play_time)', fontsize=11)
        ax_enhance_time.grid(True, alpha=0.3)
        ax_enhance_time.legend(loc='upper right', fontsize=8)
        
        # 调整布局
        plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08, hspace=0.3)
        
        # 保存或显示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图表已保存到: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _plot_buffer_level(self, ax):
        """绘制缓冲水位折线"""
        buffer_levels = self.data.get('buffer_level', [])
        if not buffer_levels:
            return
        
        times = [bl['time'] for bl in buffer_levels]
        levels = [bl['level'] for bl in buffer_levels]
        
        ax.plot(times, levels, color='blue', linewidth=2, label='缓冲水位', alpha=0.8)
    
    def _plot_stalls(self, ax):
        """绘制卡顿区域"""
        stalls = self.data.get('stalls', [])
        if not stalls:
            return
        
        # 获取Y轴范围
        y_min, y_max = ax.get_ylim()
        
        # 为每个卡顿绘制半透明红色矩形
        for stall in stalls:
            start = stall['time_start']
            end = stall['time_end']
            duration = end - start
            
            rect = Rectangle(
                (start, y_min),
                duration,
                y_max - y_min,
                facecolor='red',
                alpha=0.2,
                edgecolor='red',
                linewidth=1.5,
                label='卡顿' if stall == stalls[0] else ''  # 只为第一个添加图例
            )
            ax.add_patch(rect)
            
            # 在卡顿区域中间添加文本标注
            mid_time = (start + end) / 2
            ax.text(mid_time, y_max * 0.9, f'卡顿\n{duration:.2f}s', 
                   ha='center', va='top', fontsize=8, color='red', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    def _plot_segments_info(self, ax_main, ax_download, ax_enhance, ax_enhance_time):
        """绘制段信息"""
        segments = self.data.get('segments', [])
        if not segments:
            return
        
        # 按索引排序
        sorted_segments = sorted(segments, key=lambda s: s.get('index', 0))
        
        # 获取y轴范围（在方法开始时获取一次，避免重复调用）
        y_min, y_max = ax_main.get_ylim()
        y_range = y_max - y_min
        
        # 用于记录不同决策值的颜色映射
        download_colors = {}
        abort_reason_colors = {
            'Already played in enhance': 'orange',
            'Already played before enhance': 'purple',
            'UnknownReason': 'gray',
            'Buffer too low': 'brown',
            'Enhance timeout': 'pink'
        }
        
        # 收集数据用于绘制
        download_times = []
        download_actions = []
        enhance_times = []
        enhance_actions = []
        enhance_start_times = []
        enhance_end_times = []
        enhance_end_to_play_times = []
        enhance_end_times_for_plot = []  # 用于绘制 enhance_end_to_play_time
        
        # 用于图例去重
        legend_labels_main = set()
        legend_labels_enhance = set()
        
        # 用于跟踪enhance_end_to_play_time文本的上下位置，避免重叠
        text_offset_counter = 0
        
        for seg in sorted_segments:
            # 获取段的相对时间（使用下载开始时间）
            seg_start = self.absolute_to_relative_time(seg.get('start_time'))
            seg_stop = self.absolute_to_relative_time(seg.get('stop_time'))
            
            if seg_start is None:
                continue
            
            # 下载决策
            download_action = seg.get('download_action', 0)
            download_times.append(seg_start)
            download_actions.append(download_action)
            if download_action not in download_colors:
                download_colors[download_action] = plt.cm.tab10(len(download_colors))
            
            # 增强决策
            enhance_action = seg.get('enhance_action', 0)
            is_enhance = seg.get('is_enhance', False)
            abort_reason = seg.get('abort_reason')
            
            enhance_times.append(seg_start)
            enhance_actions.append(enhance_action)
            
            # 如果成功增强
            if is_enhance:
                enhance_start = self.absolute_to_relative_time(seg.get('enhance_start_time'))
                enhance_end = self.absolute_to_relative_time(seg.get('enhance_end_time'))
                enhance_end_to_play = seg.get('enhance_end_to_play_time')
                
                if enhance_start is not None:
                    enhance_start_times.append(enhance_start)
                if enhance_end is not None:
                    enhance_end_times.append(enhance_end)
                    if enhance_end_to_play is not None:
                        enhance_end_times_for_plot.append(enhance_end)
                        enhance_end_to_play_times.append(enhance_end_to_play)
                
                # 在主图上标记增强时间段（使用横线表示）
                if enhance_start is not None and enhance_end is not None:
                    y_pos = y_max * 0.7  # 使用预先获取的y_max
                    label_key = 'enhance_period'
                    if label_key not in legend_labels_main:
                        # 绘制横线表示增强时间段
                        ax_main.plot([enhance_start, enhance_end], [y_pos, y_pos], 
                                   color='green', linewidth=4, alpha=0.8, 
                                   label='增强时间段', zorder=3)
                        legend_labels_main.add(label_key)
                    else:
                        # 绘制横线表示增强时间段
                        ax_main.plot([enhance_start, enhance_end], [y_pos, y_pos], 
                                   color='green', linewidth=4, alpha=0.8, zorder=3)
                    
                    # 如果 enhance_end_to_play_time 存在，绘制竖向箭头
                    if enhance_end_to_play is not None:
                        # 计算箭头目标位置（播放开始时间）
                        play_start_x = enhance_end + enhance_end_to_play
                        # 箭头起点y坐标（在横线上）
                        arrow_y_start = y_pos
                        # 箭头终点y坐标：根据计数器上下错开
                        y_offset = (text_offset_counter % 2) * 2 - 1  # 交替：-1或1
                        arrow_y_end = y_pos + y_offset * y_range * 0.15  # 使用y_range而不是get_ylim()
                        
                        # 绘制竖向箭头
                        ax_main.annotate('', 
                                       xy=(play_start_x, arrow_y_end),  # 箭头终点（播放开始位置，y坐标上下错开）
                                       xytext=(play_start_x, arrow_y_start),  # 箭头起点（增强结束位置，y坐标在横线上）
                                       arrowprops=dict(arrowstyle='->', color='darkgreen', 
                                                     lw=2, alpha=0.7),
                                       zorder=4)
                        
                        # 添加文本标注（在箭头旁边，根据计数器上下错开）
                        text_y = arrow_y_end + (y_offset * y_range * 0.03)  # 使用y_range
                        ax_main.text(play_start_x, text_y, 
                                   f'{enhance_end_to_play:.2f}s',
                                   ha='center', va='bottom' if y_offset > 0 else 'top',
                                   fontsize=6, color='darkgreen', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8),
                                   zorder=5)
                        
                        # 增加计数器，下次文本会在另一侧
                        text_offset_counter += 1
            
            # 如果增强被中止
            if abort_reason:
                abort_time = seg_start
                
                # 在主图上标记中止位置（使用预先获取的y_max）
                y_pos = y_max * 0.5
                color = abort_reason_colors.get(abort_reason, 'red')
                label_key = f'abort_{abort_reason}'
                if label_key not in legend_labels_main:
                    ax_main.scatter([abort_time], [y_pos], 
                                  color=color, s=150, marker='X', zorder=5,
                                  label=f'中止: {abort_reason}')
                    legend_labels_main.add(label_key)
                else:
                    ax_main.scatter([abort_time], [y_pos], 
                                  color=color, s=150, marker='X', zorder=5)
                # # 添加文本标注
                # ax_main.text(abort_time, y_pos + ax_main.get_ylim()[1] * 0.05,
                #            f'中止\n{abort_reason[:15]}...' if len(abort_reason) > 15 else f'中止\n{abort_reason}',
                #            ha='center', fontsize=7, color=color, fontweight='bold',
                #            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # 绘制下载决策子图
        if download_times:
            ax_download.scatter(download_times, download_actions, 
                               c=[download_colors.get(a, 'blue') for a in download_actions],
                               s=50, alpha=0.7, edgecolors='black', linewidths=0.5)
            # 添加图例
            for action, color in sorted(download_colors.items()):
                ax_download.scatter([], [], color=color, s=50, label=f'下载决策 {action}')
        
        # 绘制增强决策子图
        if enhance_times:
            # 收集成功、失败和无增强的段
            success_times = []
            success_actions = []
            fail_times = []
            fail_actions = []
            no_enhance_times = []
            no_enhance_actions = []
            
            for i, seg in enumerate(sorted_segments):
                seg_start = self.absolute_to_relative_time(seg.get('start_time'))
                if seg_start is None:
                    continue
                
                enhance_action = seg.get('enhance_action', 0)
                is_enhance = seg.get('is_enhance', False)
                
                if is_enhance:
                    success_times.append(seg_start)
                    success_actions.append(enhance_action)
                elif enhance_action > 0:
                    fail_times.append(seg_start)
                    fail_actions.append(enhance_action)
                else:
                    no_enhance_times.append(seg_start)
                    no_enhance_actions.append(enhance_action)
            
            # 成功的增强
            if success_times:
                label_key = 'success_enhance'
                if label_key not in legend_labels_enhance:
                    ax_enhance.scatter(success_times, success_actions, 
                                     c='green', s=80, marker='o', alpha=0.7, 
                                     edgecolors='darkgreen', linewidths=1.5,
                                     label='成功增强', zorder=3)
                    legend_labels_enhance.add(label_key)
                else:
                    ax_enhance.scatter(success_times, success_actions, 
                                     c='green', s=80, marker='o', alpha=0.7, 
                                     edgecolors='darkgreen', linewidths=1.5, zorder=3)
            
            # 失败的增强（有中止原因）
            if fail_times:
                label_key = 'fail_enhance'
                if label_key not in legend_labels_enhance:
                    ax_enhance.scatter(fail_times, fail_actions,
                                     c='red', s=80, marker='X', alpha=0.7,
                                     edgecolors='darkred', linewidths=1.5,
                                     label='增强中止', zorder=3)
                    legend_labels_enhance.add(label_key)
                else:
                    ax_enhance.scatter(fail_times, fail_actions,
                                     c='red', s=80, marker='X', alpha=0.7,
                                     edgecolors='darkred', linewidths=1.5, zorder=3)
            
            # 没有增强决策的段
            if no_enhance_times:
                label_key = 'no_enhance'
                if label_key not in legend_labels_enhance:
                    ax_enhance.scatter(no_enhance_times, no_enhance_actions,
                                     c='gray', s=30, marker='.', alpha=0.5,
                                     label='无增强', zorder=1)
                    legend_labels_enhance.add(label_key)
                else:
                    ax_enhance.scatter(no_enhance_times, no_enhance_actions,
                                     c='gray', s=30, marker='.', alpha=0.5, zorder=1)
        
        # 绘制增强完成到播放的时间
        if enhance_end_times_for_plot and enhance_end_to_play_times:
            # 使用增强结束时间作为X轴
            ax_enhance_time.scatter(enhance_end_times_for_plot, enhance_end_to_play_times,
                                  c='darkgreen', s=100, marker='o', alpha=0.7,
                                  edgecolors='green', linewidths=2,
                                  label='enhance_end_to_play_time', zorder=3)
            # 添加趋势线
            if len(enhance_end_times_for_plot) > 1:
                z = np.polyfit(enhance_end_times_for_plot, enhance_end_to_play_times, 1)
                p = np.poly1d(z)
                ax_enhance_time.plot(enhance_end_times_for_plot, p(enhance_end_times_for_plot), 
                                   "g--", alpha=0.5, linewidth=1, label='趋势线')
        
        # 设置Y轴范围
        if download_actions:
            ax_download.set_ylim(min(download_actions) - 0.5, max(download_actions) + 0.5)
            ax_download.set_yticks(sorted(set(download_actions)))
        
        if enhance_actions:
            ax_enhance.set_ylim(min(enhance_actions) - 0.5, max(enhance_actions) + 0.5)
            ax_enhance.set_yticks(sorted(set(enhance_actions)))
        
        if enhance_end_to_play_times:
            ax_enhance_time.set_ylim(0, max(enhance_end_to_play_times) * 1.1)


def main():
    """主函数"""


    json_file = r'E:\dev\Code\Python\bones\player\output\data\bones_imdn_always_fast-False_20251223-221930.json'
    output_path = r'E:\dev\Code\Python\bones\player\output\plots-2'
    
    # 检查文件是否存在
    if not Path(json_file).exists():
        print(f"错误: 文件不存在: {json_file}")
        return
    
    # 创建分析器
    analyzer = PlaybackDataAnalyzer(json_file)
    
    # 加载数据
    print(f"正在加载数据: {json_file}")
    analyzer.load_data()
        
    # 绘制图表
    print("正在生成图表...")
    analyzer.plot_analysis(output_path)
    print("完成!")


if __name__ == '__main__':
    main()

