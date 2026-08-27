import logging
import os
import re
from abc import ABC, abstractmethod
from math import ceil
from typing import Dict, Optional
from xml.etree import ElementTree
from xml.etree.ElementTree import Element

from istream_player.models.mpd_objects import MPD, AdaptationSet, Representation, Segment


# MPD解析异常类 - 继承自BaseException
# 用于处理MPD文件解析过程中出现的各种异常情况
class MPDParsingException(BaseException):
    pass


# MPD解析器抽象基类 - 定义MPD解析器的接口规范
class MPDParser(ABC):
    # 抽象解析方法 - 子类必须实现此方法
    @abstractmethod
    def parse(self, content: str, url: str) -> MPD:
        pass


# 默认MPD解析器实现类 - 继承自MPDParser抽象基类
# 负责解析DASH标准的MPD XML文件，将其转换为播放器可用的MPD对象
class DefaultMPDParser(MPDParser):
    # 获取日志记录器实例，用于记录MPD解析相关的日志信息
    log = logging.getLogger("DefaultMPDParser")

    # 解析ISO8601时间格式的静态方法
    @staticmethod
    def parse_iso8601_time(duration: Optional[str]) -> float:
        """
        将ISO8601时间字符串解析为秒数
        支持格式：PT[小时H][分钟M][秒S]，例如：PT1H30M45S
        """
        # 如果持续时间为空或None，返回0
        if duration is None or duration == "":
            return 0
        # 定义ISO8601时间格式的正则表达式模式
        # 匹配PT开头，可选的小时、分钟、秒部分
        pattern = r"^PT(?:(\d+(?:.\d+)?)H)?(?:(\d+(?:.\d+)?)M)?(?:(\d+(?:.\d+)?)S)?$"
        # 使用正则表达式匹配时间字符串
        results = re.match(pattern, duration)
        if results is not None:
            # 提取匹配的小时、分钟、秒数值，None值转换为0
            dur = [float(i) if i is not None else 0 for i in results.group(1, 2, 3)]
            # 将时间转换为总秒数：小时*3600 + 分钟*60 + 秒
            dur = 3600 * dur[0] + 60 * dur[1] + dur[2]
            return dur
        else:
            # 如果格式不匹配，返回0
            return 0

    # 移除XML命名空间的静态方法
    @staticmethod
    def remove_namespace_from_content(content):
        """
        从XML字符串中移除命名空间声明
        这简化了后续的XML解析过程
        """
        # 使用正则表达式移除xmlns属性
        content = re.sub('xmlns="[^"]+"', "", content, count=1)
        return content

    # 解析MPD文件内容的主要方法
    def parse(self, content: str, url: str) -> MPD:
        # 移除XML命名空间，简化解析过程
        content = self.remove_namespace_from_content(content)
        # 将XML字符串解析为ElementTree根元素
        root = ElementTree.fromstring(content)

        # 获取MPD类型（静态或动态）
        type_ = root.attrib["type"]
        # 断言确保MPD类型为静态或动态
        assert type_ == "static" or type_ == "dynamic"

        # 解析媒体演示持续时间
        # 从根元素的mediaPresentationDuration属性获取总时长
        media_presentation_duration = self.parse_iso8601_time(root.attrib.get("mediaPresentationDuration", ""))
        # 记录媒体演示持续时间到日志
        self.log.info(f"{media_presentation_duration=}")

        # 解析最小缓冲时间
        # 从根元素的minBufferTime属性获取最小缓冲时间
        min_buffer_time = self.parse_iso8601_time(root.attrib.get("minBufferTime", ""))
        # 记录最小缓冲时间到日志
        self.log.info(f"{min_buffer_time=}")

        # 解析最大段持续时间
        # 从根元素的maxSegmentDuration属性获取最大段持续时间
        max_segment_duration = self.parse_iso8601_time(root.attrib.get("maxSegmentDuration", ""))
        # 记录最大段持续时间到日志
        self.log.info(f"{max_segment_duration=}")

        # 查找Period元素（DASH标准中的时间段）
        period = root.find("Period")

        # 如果找不到Period元素，抛出解析异常
        if period is None:
            raise MPDParsingException('Cannot find "Period" tag')

        # 初始化自适应集字典，用于存储解析后的自适应集
        adaptation_sets: Dict[int, AdaptationSet] = {}

        # 构建基础URL，用于生成完整的段URL
        base_url = os.path.dirname(url) + "/"

        # 遍历Period下的所有子元素（自适应集）
        for index, adaptation_set_xml in enumerate(period):
            # 只处理视频类型的内容（默认contentType为video）
            if adaptation_set_xml.attrib.get("contentType", "video").lower() == "video":
                # 解析当前自适应集
                adaptation_set: AdaptationSet = self.parse_adaptation_set(
                    adaptation_set_xml, base_url, index, media_presentation_duration
                )
                # 将解析后的自适应集添加到字典中
                adaptation_sets[adaptation_set.id] = adaptation_set

        # 创建并返回MPD对象
        return MPD(content, url, type_, media_presentation_duration, max_segment_duration, min_buffer_time, adaptation_sets, root.attrib)

    # 解析自适应集的方法
    def parse_adaptation_set(
        self, tree: Element, base_url, index: Optional[int], media_presentation_duration: float
    ) -> AdaptationSet:
        # 获取自适应集ID，如果不存在则使用索引
        id_ = int(tree.attrib.get("id", str(index)))
        # 获取内容类型，默认为video
        content_type = tree.attrib.get("contentType", "video")
        # 断言确保内容类型为video或audio
        assert (
            content_type == "video" or content_type == "audio"
        ), f"Only 'video' or 'audio' content_type is supported, Got {content_type}"

        # 获取帧率信息
        frame_rate = tree.attrib.get("frameRate")
        # 如果frameRate不存在，尝试获取maxFrameRate
        if frame_rate is None:
            frame_rate = tree.attrib.get("maxFrameRate")
        # 获取最大宽度
        max_width = int(tree.attrib.get("maxWidth", 0))
        # 获取最大高度
        max_height = int(tree.attrib.get("maxHeight", 0))
        # 获取像素宽高比
        par = tree.attrib.get("par")
        # 记录调试信息到日志
        self.log.debug(f"{frame_rate=}, {max_width=}, {max_height=}, {par=}")

        # 初始化表示字典
        representations = {}
        # 查找段模板（GPAC MPD格式在自适应集内包含段模板）
        segment_template: Optional[Element] = tree.find("SegmentTemplate")

        # 遍历所有表示元素
        for representation_tree in tree.findall("Representation"):
            # 解析当前表示
            representation = self.parse_representation(
                representation_tree, id_, base_url, segment_template, media_presentation_duration
            )
            # 将解析后的表示添加到字典中
            representations[representation.id] = representation
        # 创建并返回自适应集对象
        return AdaptationSet(int(id_), content_type, frame_rate, max_width, max_height, par, representations, tree.attrib)

    # 解析表示的方法
    def parse_representation(
        self, tree: Element, as_id: int, base_url, segment_template: Optional[Element], media_presentation_duration: float
    ) -> Representation:
        # 如果自适应集级别的段模板为None，尝试在表示级别查找段模板
        segment_template = tree.find("SegmentTemplate") if segment_template is None else segment_template
        if segment_template is not None:
            # 使用段模板解析表示
            return self.parse_representation_with_segment_template(
                tree, as_id, base_url, segment_template, media_presentation_duration
            )
        else:
            # 如果没有段模板，抛出异常（当前实现不支持）
            raise MPDParsingException("The MPD support is not complete yet")

    # 使用段模板解析表示的方法
    def parse_representation_with_segment_template(
        self, tree: Element, as_id: int, base_url, segment_template: Element, media_presentation_duration: float
    ) -> Representation:
        # 获取表示ID
        id_ = tree.attrib["id"]
        # 获取MIME类型
        mime = tree.attrib["mimeType"]
        # 获取编解码器信息
        codec = tree.attrib["codecs"]
        # 获取带宽信息
        bandwidth = int(tree.attrib["bandwidth"])
        # 获取视频宽度
        width = int(tree.attrib["width"])
        # 获取视频高度
        height = int(tree.attrib["height"])

        # 断言确保段模板存在
        assert segment_template is not None, "Segment Template not found in representation"

        # 获取初始化段URL模板
        initialization = segment_template.attrib["initialization"]
        # 替换模板中的$RepresentationID$变量
        initialization = initialization.replace("$RepresentationID$", id_)
        # 构建完整的初始化段URL
        initialization = base_url + initialization
        # 初始化段字典
        segments: Dict[int, Segment] = {}

        # 获取时间刻度（每秒的刻度数）
        timescale = int(segment_template.attrib["timescale"])
        # 获取媒体段URL模板
        media = segment_template.attrib["media"].replace("$RepresentationID$", id_)
        # 获取起始段编号
        start_number = int(segment_template.attrib["startNumber"])

        # 查找段时间线元素
        segment_timeline = segment_template.find("SegmentTimeline")
        if segment_timeline is not None:
            # 使用段时间线解析段信息
            num = start_number
            start_time = 0
            # 遍历段时间线中的每个段
            for segment in segment_timeline:
                # 计算段持续时间（将时间刻度转换为秒）
                duration = float(segment.attrib["d"]) / timescale
                # 构建段URL（替换$Number$变量）
                url = base_url + re.sub(r"\$Number(%\d+d)\$", r"\1", media) % num
                # 如果段有时间偏移，使用指定的开始时间
                if "t" in segment.attrib:
                    start_time = float(segment.attrib["t"]) / timescale
                # 创建段对象并添加到段字典
                segments[num] = Segment(url, initialization, duration, start_time, as_id, int(id_))
                # 递增段编号
                num += 1
                # 更新下一个段的开始时间
                start_time += duration

                # 如果段有重复次数，创建重复的段
                if "r" in segment.attrib:  # repeat
                    for _ in range(int(segment.attrib["r"])):
                        # 构建重复段的URL
                        url = base_url + self.var_repl(media, {"Number": num})
                        # 创建重复段对象
                        segments[num] = Segment(url, initialization, duration, start_time, as_id, int(id_))
                        # 递增段编号
                        num += 1
                        # 更新下一个段的开始时间
                        start_time += duration
        else:
            # GPAC DASH格式（没有段时间线）
            num = start_number
            start_time = 0
            # 计算总段数（媒体演示持续时间 / 段持续时间）
            num_segments = ceil((media_presentation_duration * timescale) / int(segment_template.attrib["duration"]))
            # 计算段持续时间
            duration = float(segment_template.attrib["duration"]) / timescale
            # 记录调试信息
            self.log.debug(f"{num_segments=}, {duration=}")
            # 创建所有段
            for _ in range(num_segments):
                # 构建段URL
                url = base_url + self.var_repl(media, {"Number": num})
                # 创建段对象
                segments[num] = Segment(url, initialization, duration, start_time, as_id, int(id_))
                # 递增段编号
                num += 1
                # 更新下一个段的开始时间
                start_time += duration
            # 注释掉的段调试日志
            # self.log.debug(segments)

        # 创建并返回表示对象
        return Representation(int(id_), mime, codec, bandwidth, width, height, initialization, segments, tree.attrib)

    # 变量替换的静态方法
    @staticmethod
    def var_repl(s: str, vars: Dict[str, int | str]):
        # 内部替换函数，处理单个变量的替换
        def _repl(m) -> str:
            # 提取变量名（去掉$符号）
            m = m.group()[1:-1]
            # 如果变量在字典中，直接替换
            if m in vars:
                return str(vars[m])
            # 如果变量包含格式化字符串（如$Number%04d$）
            elif "%" in m:
                # 分割变量名和格式化字符串
                v, p = m.split("%", 1)
                # 使用格式化字符串格式化变量值
                return f"%{p}" % vars[v]
            else:
                # 如果无法替换变量，抛出异常
                raise Exception(f"Cannot replace {m} in {s}")

        # 使用正则表达式查找所有$...$格式的变量并替换
        return re.sub(r"\$.*\$", _repl, s)
