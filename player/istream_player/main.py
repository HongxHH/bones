import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import yaml

from istream_player.config.config import PlayerConfig
from istream_player.core.module_composer import PlayerComposer


def load_from_dict(d: Dict, config: PlayerConfig):
    for k, v in d.items():
        if isinstance(v, List):
            prev_list = config.__getattribute__(k)
            if prev_list is None:
                prev_list = []
                config.__setattr__(k, prev_list)
            prev_list.extend(v)
        elif isinstance(v, dict):
            prev_d = config.__getattribute__(k)
            if prev_d is None:
                prev_d = {}
                config.__setattr__(k, prev_d)
            prev_d.update(v)
        elif v is not None:
            config.__setattr__(k, v)
    return config


def load_from_config_file(config_path: str, config: PlayerConfig):
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path) as f:
            return load_from_dict(yaml.safe_load(f), config)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path) as f:
            return load_from_dict(json.load(f), config)
    else:
        raise Exception(f"Config file format not supported. Use JSON or YAML. Used : {config_path}")


def main():
    # 检查 Python 版本是否大于等于 3.3
    try:
        assert sys.version_info.major >= 3 and sys.version_info.minor >= 3
    except AssertionError:
        print("Python 3.3+ is required.")
        exit(-1)
    
    # 创建 PlayerComposer 实例，用于管理和组合各个模块
    composer = PlayerComposer()
    # 注册所有核心模块（如下载器、播放器、渲染器等）
    composer.register_core_modules()
    
    # 先创建默认配置（PlayerConfig 实例），用于命令行参数解析器的默认值
    config = PlayerConfig()
    # 创建命令行参数解析器，传入config实例以使用config.py中的默认值
    parser = composer.create_arg_parser(config)
    # 解析命令行参数，转换为字典
    args = vars(parser.parse_args())

    # 如果指定了配置文件，优先从配置文件加载参数
    if args["config"] is not None:
        load_from_config_file(args["config"], config)
        del args["config"]  # 移除 config 参数，避免后续重复处理
    verbose_flag = args.pop("verbose", False)

    # 用命令行参数覆盖配置文件或默认配置
    load_from_dict(args, config)

    # 设置日志输出到文件与控制台
    log_dir = config.log
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{config.mod_nes}_{config.mod_enhancer}_{timestamp}.log")
    log_level = logging.DEBUG if verbose_flag else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)20s %(levelname)8s:\t%(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True,
    )

    # 校验配置参数的有效性
    config.validate()

    # 启动异步事件循环，运行播放器主流程
    asyncio.run(composer.run(config))


if __name__ == "__main__":
    # 如果作为主程序运行，则调用 main() 入口
    main()
