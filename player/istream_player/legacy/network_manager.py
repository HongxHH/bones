import json
import logging
import socket
import subprocess
from pprint import pprint
from threading import Thread
from time import sleep, time
from typing import List

from istream_player.analyzers.exp_events import ExpEvent_BwSwitch
from istream_player.analyzers.exp_recorder import ExpWriter

# 网络接口名称常量，用于Linux网络配置
IF_NAME = "eth0"
# NetEm队列限制常量，限制网络模拟器的队列大小
NETEM_LIMIT = 1000


# 网络配置类 - 封装单个网络配置参数和操作方法
# 负责管理带宽、延迟、丢包率等网络参数，并提供Docker容器内的网络配置功能
class NetworkConfig:
    def __init__(self, bw, latency, drop, sustain, recorder, log, server_container, target_ip):
        # 带宽参数（kbps），用于限制网络传输速率
        self.bw = bw
        # 延迟参数（ms），用于模拟网络延迟
        self.latency = latency
        # 丢包率参数（0-1），用于模拟网络丢包
        self.drop = drop
        # 持续时间参数（秒），表示该网络配置的持续时间
        self.sustain = sustain
        # 实验记录器实例，用于记录网络切换事件
        self.recorder: ExpWriter = recorder
        # 日志记录器实例，用于记录网络配置相关的日志
        self.log = log
        # 创建TCP套接字，用于与服务器容器通信（当前未使用）
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # 目标IP地址，用于网络流量过滤
        self.target_ip = target_ip
        # 服务器容器信息字典，包含容器ID、镜像名、容器名等
        self.server_container = server_container
        # 记录服务器容器信息到日志
        self.log.info(f"Server Container: {self.server_container}")
        # 注释掉的阻塞循环 - 用于调试时的阻塞测试
        # while True:
        #     self.log.info(f"Blocked")
        #     sleep(1)

    # 在Docker容器内执行脚本的方法
    def run_in_container(self, script):
        # 注释掉的套接字通信方式 - 当前使用Docker exec方式
        # self.server_socket.sendall(script.encode())
        # 构建Docker exec命令，在服务器容器内执行bash脚本
        cmd = ["docker", "exec", self.server_container['ID'], "bash", "-c", script]
        # 执行Docker命令，使用subprocess.check_call确保命令成功执行
        subprocess.check_call(cmd, stderr=subprocess.STDOUT)
        # 记录执行的命令到日志
        self.log.info("Running inside container: " + " ".join(cmd))

    # 在主机上执行脚本的方法
    def run_on_host(self, script):
        # 在主机上执行shell脚本，使用bash作为解释器
        subprocess.check_call(script, shell=True, executable="/bin/bash", stderr=subprocess.STDOUT)
        # 记录执行的脚本到日志
        self.log.info("Running on host: " + script)

    # 应用网络配置的方法 - 使用Linux tc命令配置网络参数
    def apply(self, if_name):
        # 构建tc命令脚本，用于修改现有的网络配置
        script = f'''
        set -e
        # 修改TBF（Token Bucket Filter）队列，设置带宽限制
        # rate: 带宽限制（kbps），limit: 队列长度，burst: 突发大小
        tc qdisc change dev {if_name} handle 2: tbf rate {self.bw}kbit limit 10000 burst 2000
        # 修改NetEm队列，设置延迟和丢包率
        # delay: 延迟（ms），loss: 丢包率（%），limit: 队列限制
        tc qdisc change dev {if_name} handle 3: netem limit {NETEM_LIMIT} delay {self.latency}ms 0ms loss {float(self.drop) * 0:.3f}%
        '''
        # 获取当前时间戳（毫秒），用于事件记录
        t = round(time() * 1000)
        # 在容器内执行网络配置脚本
        self.run_in_container(script)
        # 记录带宽切换事件到实验记录器
        self.recorder.write_event(ExpEvent_BwSwitch(t, float(self.bw), float(self.latency), float(self.drop)))

    # 初始化网络配置的方法 - 设置初始的网络队列和过滤器
    def setup(self, if_name):
        # 构建tc命令脚本，用于初始化网络配置
        script = f'''
        set -e
        # 删除现有的根队列（如果存在），忽略错误
        tc qdisc del dev {if_name} root || true
        # 添加优先级队列作为根队列，用于流量分类
        tc qdisc add dev {if_name} root handle 1: prio
        # 添加TBF队列，设置带宽限制
        # parent 1:3: 作为优先级队列的第3个子队列
        tc qdisc add dev {if_name} parent 1:3 handle 2: tbf rate {self.bw}kbit limit 10000 burst 2000
        # 添加NetEm队列，设置延迟和丢包率
        # parent 2:1: 作为TBF队列的子队列
        tc qdisc add dev {if_name} parent 2:1 handle 3: netem limit {NETEM_LIMIT} delay {self.latency}ms 0ms loss {float(self.drop) * 0:.3f}%
        # 添加流量过滤器，将目标IP的流量路由到第3个队列
        # u32: 使用u32分类器，match ip dst: 匹配目标IP
        tc filter add dev {if_name} protocol ip parent 1:0 prio 3 u32 match ip dst {self.target_ip} flowid 1:3
            '''
        # 在容器内执行网络配置脚本
        self.run_in_container(script)


# 网络管理器类 - 管理网络配置的时间线和执行
# 负责从配置文件读取网络参数，管理Docker容器，并执行网络配置切换
class NetworkManager:
    # 获取日志记录器实例，用于记录网络管理相关的日志信息
    log = logging.getLogger("NetworkManager")

    def __init__(self, bw_profile_path: str, recorder: ExpWriter):
        # 客户端IP地址，用于网络流量过滤
        self.client_ip = None
        # 当前容器信息字典，包含当前运行容器的详细信息
        self.current_container = None
        # 服务器容器信息字典，包含服务器容器的详细信息
        self.server_container = None
        # 强制停止标志，用于控制网络配置循环的执行
        self.force_stop = False
        # 带宽配置文件路径，包含网络参数的时间线
        self.bw_profile_path = bw_profile_path
        # 配置切换延迟（秒），用于控制配置切换的时间间隔
        self.delay = 1
        # 网络配置时间线列表，存储按时间顺序的网络配置
        self.timeline: List[NetworkConfig] = []
        # 实验记录器实例，用于记录网络切换事件
        self.recorder = recorder
        # 获取服务器容器信息
        self.get_server_container()
        # 获取客户端IP地址
        self.get_client_ip()

        # 读取带宽配置文件，构建网络配置时间线
        with open(bw_profile_path) as f:
            # 用于检测重复配置行的变量
            last_line = ""
            # 逐行读取配置文件
            for line in f:
                # 检查是否为重复的配置行
                if line == last_line:
                    # 如果是重复行，增加上一个配置的持续时间
                    self.timeline[-1].sustain += self.delay
                    continue
                # 更新最后一行记录
                last_line = line
                # 解析配置行，提取带宽、延迟、丢包率参数
                [bw, latency, drop] = line.strip().split(" ")
                # 创建新的网络配置对象并添加到时间线
                self.timeline.append(NetworkConfig(bw, latency, drop, self.delay, self.recorder, log=self.log,
                                                   server_container=self.server_container, target_ip=self.client_ip))

    # 获取服务器容器信息的方法
    def get_server_container(self):
        # 获取当前容器的信息，使用Docker ps命令和JSON格式输出
        self.current_container = json.loads(subprocess.check_output(
            'docker ps --format \'{"ID":"{{ .ID }}", "Image": "{{ .Image }}", "Names":"{{ .Names }}"}\' --filter id=$(cat /etc/hostname)',
            shell=True, executable="/bin/bash"))
        # 验证当前容器是否为客户端容器（以-client-1结尾）
        if not self.current_container["Names"].endswith("-client-1"):
            # 如果不是客户端容器，打印容器信息并抛出异常
            pprint(self.current_container)
            raise Exception("Failed to get current container")
        # 从容器名中提取Docker Compose项目名
        # 移除"-client-1"后缀得到项目名
        docker_compose_proj = self.current_container["Names"][:-len("-client-1")]
        # 构建服务器容器名，添加"-server-1"后缀
        server_container_name = docker_compose_proj + "-server-1"
        # 获取服务器容器的信息
        self.server_container = json.loads(subprocess.check_output(
            'docker ps --format \'{"ID":"{{ .ID }}", "Image": "{{ .Image }}", "Names":"{{ .Names }}"}\' --filter name=' + server_container_name,
            shell=True, executable="/bin/bash"))
        # 验证服务器容器获取是否成功（这里应该是检查server_container）
        if not self.current_container["Names"].endswith("-client-1"):
            # 如果不是客户端容器，打印容器信息并抛出异常
            pprint(self.current_container)
            raise Exception("Failed to get server container")

    # 获取客户端IP地址的方法
    def get_client_ip(self):
        # 断言确保服务器容器信息已获取
        assert self.server_container is not None
        # 在服务器容器内执行dig命令，获取客户端IP地址
        # dig +short client: 查询client域名的IP地址
        self.client_ip = subprocess.check_output(["docker", "exec", self.server_container["ID"], "bash", "-c", "dig +short client"]).decode().strip()
        # 记录检测到的客户端IP地址到日志
        self.log.info(f"Detected current container IP: {self.client_ip}")

    # 开始网络配置的方法 - 按时间线顺序应用网络配置
    def start(self, if_name):
        # 遍历网络配置时间线
        for config in self.timeline:
            # 应用当前网络配置
            config.apply(if_name)
            # 记录配置持续时间到日志
            self.log.info(f"Sustain Network Config for {config.sustain} seconds")
            # 按秒循环，维持当前配置
            for s in range(config.sustain):
                # 检查是否收到强制停止信号
                if self.force_stop:
                    return
                # 休眠1秒，等待下一个配置周期
                sleep(1)

    # 在后台启动网络管理器的方法
    def start_bg(self):
        # 注释掉的动态网络接口检测代码
        # 原本用于自动检测Docker容器的网络接口
        # if_name = subprocess.check_output(f'''
        # grep -l $(docker exec {os.environ["CONTAINER"]} bash -c 'cat /sys/class/net/eth0/iflink' | tr -d '\\r') /sys/class/net/veth*/ifindex | sed -e 's;^.*net/\\(.*\\)/ifindex$;\\1;'
        # ''', shell=True, executable="/bin/bash", stderr=subprocess.STDOUT).decode().strip()
        # 使用固定的网络接口名称
        if_name = "eth0"
        # 设置第一个网络配置的初始状态
        self.timeline[0].setup(if_name)
        # 记录后台启动信息到日志
        self.log.info("Starting Network Manager in background")
        # 创建后台线程，执行网络配置循环
        t = Thread(target=self.start, args=[if_name], daemon=True)
        # 启动后台线程
        t.start()

    # 停止后台网络管理器的方法
    def stop_bg(self):
        # 记录停止信息到日志
        self.log.info("Stopping Network Manager in background")
        # 断言确保服务器容器信息已获取
        assert self.server_container is not None
        # 设置强制停止标志，通知网络配置循环停止
        self.force_stop = True
        # 向服务器容器发送SIGINT信号，停止网络配置进程
        subprocess.check_call(["docker", "exec", self.server_container["ID"], "bash", "-c", "kill -s SIGINT 1"])
        # 方法结束标记
        pass
