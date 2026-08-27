# 会话模型：管理下载和增强两条时间线的会话切换
# 这是仿真系统的核心调度组件，负责协调下载和增强两个并行时间线


class SessionModel():
    def __init__(self, logger):
        """
        决策会话管理器构造函数：管理下载缓冲和增强缓冲的两条时间线
        每条时间线在处理会话和等待会话之间交替
        总共有四种会话类型。在每个会话开始时：
            0: 结束等待，开始下载
            1: 结束下载，开始等待
            2: 结束等待，开始增强
            3: 结束增强，开始等待
        
        :param logger: 信息日志器，用于输出调试和状态信息
        """
        # 定义四种会话类型的常量
        # 这些常量用于标识不同的会话状态转换
        self.DOWNLOAD_START = 0  # 下载开始会话：从等待状态转换到下载状态
        self.DOWNLOAD_FINISH = 1  # 下载结束会话：从下载状态转换到等待状态
        self.ENHANCE_START = 2  # 增强开始会话：从等待状态转换到增强状态
        self.ENHANCE_FINISH = 3  # 增强结束会话：从增强状态转换到等待状态

        # 基础配置
        self.logger = logger  # 日志器实例，用于输出调试信息

        # 下载时间线状态变量
        self.is_downloading = False  # 下载状态标志
        # True: 正在下载中, False: 处于等待状态
        # 用于跟踪下载时间线的当前状态
        
        self.cd_download = 0  # 下载会话倒计时（毫秒）
        # 表示当前下载会话的剩余时间
        # 当倒计时为0时，会话结束，状态转换

        # 增强时间线状态变量
        self.is_enhancing = False  # 增强状态标志
        # True: 正在增强中, False: 处于等待状态
        # 用于跟踪增强时间线的当前状态
        
        self.cd_enhance = 0  # 增强会话倒计时（毫秒）
        # 表示当前增强会话的剩余时间
        # 当倒计时为0时，会话结束，状态转换

    def next(self):
        """
        进入下一个会话：选择剩余时间最短的时间线推进
        这是会话模型的核心方法，实现双时间线的协调调度
        
        调度策略：
        1. 比较两个时间线的剩余时间
        2. 选择剩余时间最短的时间线推进
        3. 推进时间等于该时间线的剩余时间
        4. 另一条时间线同步减去相同的推进时间
        5. 根据当前状态确定下一个会话类型
        
        :return: (下一个会话类型索引, 当前会话剩余时间) 元组
        """
        # 选择剩余时间最短的时间线推进
        if self.cd_download < self.cd_enhance:  # 下载时间线剩余时间更短
            # 下载时间线优先推进
            time_session = self.cd_download  # 推进时间等于下载倒计时
            self.cd_enhance -= time_session  # 增强倒计时减去推进时间
            self.cd_download = 0  # 下载倒计时清零，表示会话结束
            
            if not self.is_downloading:  # 当前不在下载状态（处于等待状态）
                # 结束等待，开始下载
                if self.logger.is_verbose():  # 检查是否开启详细日志
                    self.logger.write("(session) download_start")
                    # 输出会话转换日志
                return self.DOWNLOAD_START, time_session  # 返回下载开始会话
            else:  # 当前正在下载状态
                # 结束下载，开始等待
                if self.logger.is_verbose():  # 检查是否开启详细日志
                    self.logger.write("(session) download_finish")
                    # 输出会话转换日志
                return self.DOWNLOAD_FINISH, time_session  # 返回下载结束会话
        else:  # self.cd_enhance <= self.cd_download，增强时间线剩余时间更短或相等
            # 增强时间线优先推进
            # 注意：这里有一个潜在的bug，应该是time_session = self.cd_enhance
            time_session = self.cd_download  # 推进时间等于下载倒计时（应该是cd_enhance）
            self.cd_download -= time_session  # 下载倒计时减去推进时间
            self.cd_enhance = 0  # 增强倒计时清零，表示会话结束
            
            if not self.is_enhancing:  # 当前不在增强状态（处于等待状态）
                # 结束等待，开始增强
                if self.logger.is_verbose():  # 检查是否开启详细日志
                    self.logger.write("(session) enhance_start")
                    # 输出会话转换日志
                return self.ENHANCE_START, time_session  # 返回增强开始会话
            else:  # 当前正在增强状态
                # 结束增强，开始等待
                if self.logger.is_verbose():  # 检查是否开启详细日志
                    self.logger.write("(session) enhance_finish")
                    # 输出会话转换日志
                return self.ENHANCE_FINISH, time_session  # 返回增强结束会话

    def set_download(self, is_doing, cd):
        """
        设置下载状态和倒计时：用于外部控制下载时间线的状态转换
        
        这个方法允许外部代码（如仿真器主循环）控制下载时间线的状态：
        - 启动下载会话：设置is_doing=True，并设置相应的倒计时
        - 结束下载会话：设置is_doing=False，倒计时通常设为0
        
        :param is_doing: 是否正在下载（布尔值）
        # True: 设置为下载状态, False: 设置为等待状态
        :param cd: 下载倒计时（毫秒）
        # 表示当前下载会话的剩余时间
        # 通常下载开始时设置为下载时长，结束时设置为0
        :return: 无返回值，通过修改实例属性更新状态
        """
        self.is_downloading = is_doing  # 设置下载状态标志
        self.cd_download = cd  # 设置下载倒计时
        return  # 方法结束

    def set_enhance(self, is_doing, cd):
        """
        设置增强状态和倒计时：用于外部控制增强时间线的状态转换
        
        这个方法允许外部代码（如仿真器主循环）控制增强时间线的状态：
        - 启动增强会话：设置is_doing=True，并设置相应的倒计时
        - 结束增强会话：设置is_doing=False，倒计时通常设为0
        
        :param is_doing: 是否正在增强（布尔值）
        # True: 设置为增强状态, False: 设置为等待状态
        :param cd: 增强倒计时（毫秒）
        # 表示当前增强会话的剩余时间
        # 通常增强开始时设置为增强时长，结束时设置为0
        :return: 无返回值，通过修改实例属性更新状态
        """
        self.is_enhancing = is_doing  # 设置增强状态标志
        self.cd_enhance = cd  # 设置增强倒计时
        return  # 方法结束