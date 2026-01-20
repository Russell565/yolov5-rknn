#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import yaml
import logging
import time
import paramiko
from paramiko import SSHException


def setup_logging(log_level: str = 'INFO', log_file: str = 'sftp_push.log') -> logging.Logger:
    """
    设置日志配置
    :param log_level: 日志级别
    :param log_file: 日志文件路径
    :return: 日志记录器
    """
    logger = logging.getLogger('SFTPPush')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 创建日志目录（如果不存在）
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志器
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> dict:
    """
    加载配置文件
    :param config_path: 配置文件路径
    :return: 配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def push_model(config: dict, logger: logging.Logger, config_path: str) -> bool:
    """
    推送模型文件到远程设备
    :param config: 配置字典
    :param logger: 日志记录器
    :param config_path: 配置文件路径
    :return: 操作是否成功
    """
    sftp_config = config.get('sftp', {})
    
    # 获取连接参数
    host = sftp_config.get('host', 'localhost')
    port = sftp_config.get('port', 22)
    username = sftp_config.get('username', 'root')
    password = sftp_config.get('password', None)
    private_key = sftp_config.get('private_key', None)
    timeout = sftp_config.get('timeout', 30)
    
    logger.info(f"开始SFTP推送: {username}@{host}:{port}")
    
    # 获取文件列表 - 支持多文件配置
    files_list = sftp_config.get('files', [])
    
    # 兼容旧版配置格式
    if not files_list:
        # 单文件配置模式
        local_file = sftp_config.get('local_file', '')
        remote_dir = sftp_config.get('remote_dir', '/')
        remote_file = sftp_config.get('remote_file', '')
        
        if local_file:
            files_list = [{"local_file": local_file, "remote_dir": remote_dir, "remote_file": remote_file}]
        else:
            logger.error("未配置任何要推送的文件")
            return False
    
    try:
        # 创建 SSH 客户端
        ssh_client = paramiko.SSHClient()
        # 自动添加主机密钥（生产环境建议关闭，使用 known_hosts 文件）
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # 连接 SSH 服务器
        if private_key and os.path.exists(private_key):
            # 使用密钥认证
            logger.info(f"使用私钥认证: {private_key}")
            key = paramiko.RSAKey.from_private_key_file(private_key)
            ssh_client.connect(
                hostname=host,
                port=port,
                username=username,
                pkey=key,
                timeout=timeout
            )
        else:
            # 使用密码认证
            logger.info("使用密码认证")
            ssh_client.connect(
                hostname=host,
                port=port,
                username=username,
                password=password,
                timeout=timeout
            )
        
        # 打开 SFTP 会话
        sftp = ssh_client.open_sftp()
        logger.info("SFTP 连接成功")
        
        # 遍历所有要推送的文件
        for file_config in files_list:
            local_file = file_config.get('local_file', '')
            remote_dir = file_config.get('remote_dir', '/')
            remote_file = file_config.get('remote_file', '')
            
            # 如果未指定远程文件名，使用本地文件名
            if not remote_file and local_file:
                remote_file = os.path.basename(local_file)
            
            # 检查本地文件是否存在
            if not local_file:
                logger.warning("跳过空的本地文件配置")
                continue
            
            if not os.path.exists(local_file):
                logger.error(f"本地模型文件不存在: {local_file}")
                continue
            
            # 确保远程目录存在
            try:
                sftp.stat(remote_dir)
                logger.info(f"远程目录已存在: {remote_dir}")
            except IOError:
                logger.info(f"创建远程目录: {remote_dir}")
                sftp.mkdir(remote_dir)
            
            # 列出远程目录内容
            logger.info(f"列出远程目录: {remote_dir}")
            remote_files = sftp.listdir(remote_dir)
            logger.info(f"远程目录内容: {remote_files}")
            
            # 上传模型文件
            if local_file and remote_file:
                remote_path = os.path.join(remote_dir, remote_file)
                logger.info(f"上传本地文件: {local_file} -> {remote_path}")
                sftp.put(local_file, remote_path)
                logger.info(f"文件上传成功: {remote_path}")
                
                # 验证文件是否上传成功
                sftp.stat(remote_path)
                logger.info(f"文件验证成功: {remote_path}")
        
        # SFTP推送完成，立即发送微信通知
        logger.info("SFTP推送完成，发送微信通知")
        import subprocess
        try:
            # 获取当前脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 调用bash脚本发送SFTP推送通知
            cmd = f"bash -c 'source {script_dir}/../utils/wechat_notifier.sh && send_sftp_notification {config_path} success multiple_files multiple_remote_paths'"
            subprocess.run(cmd, shell=True, check=True)
        except Exception as e:
            logger.error(f"发送SFTP微信通知失败: {str(e)}")
        
        # 远程执行脚本
        remote_exec_config = sftp_config.get('remote_exec', {})
        remote_exec_enable = remote_exec_config.get('enable', 0)
        
        if remote_exec_enable:
            logger.info("开始执行远程脚本...")
            
            # 获取远程执行参数
            script_path = remote_exec_config.get('script_path', '')
            params_template = remote_exec_config.get('params', '')
            exec_timeout = remote_exec_config.get('timeout', 300)
            
            if not script_path:
                logger.error("远程脚本路径未配置")
                sftp.close()
                ssh_client.close()
                logger.info("SSH 连接已关闭")
                return False
            
            # 替换参数模板中的变量
            # 注意：如果需要针对不同文件执行不同脚本，需要修改此逻辑
            # 当前版本针对所有文件执行同一脚本
            params = params_template
            
            # 获取脚本所在目录
            script_dir = os.path.dirname(script_path)
            # 切换到脚本所在目录后执行命令
            full_command = f"cd {script_dir} && bash {os.path.basename(script_path)} {params}"
            logger.info(f"执行远程命令: {full_command}")
            
            try:
                # 执行远程命令
                stdin, stdout, stderr = ssh_client.exec_command(full_command, timeout=exec_timeout)
                
                # 读取命令输出
                stdout_output = stdout.read().decode('utf-8')
                stderr_output = stderr.read().decode('utf-8')
                
                # 获取命令退出码
                exit_status = stdout.channel.recv_exit_status()
                
                if exit_status == 0:
                    logger.info(f"远程脚本执行成功")
                    if stdout_output:
                        logger.info(f"脚本输出: {stdout_output}")
                    
                    # 发送微信通知 - 板子测试成功
                    logger.info("发送微信通知 - 板子测试成功")
                    import subprocess
                    try:
                        # 获取当前脚本所在目录
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        # 调用bash脚本发送微信通知，不传递详细输出
                        cmd = f"bash -c 'source {script_dir}/../utils/wechat_notifier.sh && send_board_test_notification {config_path} success'"
                        subprocess.run(cmd, shell=True, check=True)
                    except Exception as e:
                        logger.error(f"发送微信通知失败: {str(e)}")
                else:
                    logger.error(f"远程脚本执行失败，退出码: {exit_status}")
                    if stderr_output:
                        logger.error(f"错误输出: {stderr_output}")
                    if stdout_output:
                        logger.info(f"脚本输出: {stdout_output}")
                    
                    # 发送微信通知 - 板子测试失败
                    logger.info("发送微信通知 - 板子测试失败")
                    import subprocess
                    try:
                        # 获取当前脚本所在目录
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        # 调用bash脚本发送微信通知，不传递详细输出
                        cmd = f"bash -c 'source {script_dir}/../utils/wechat_notifier.sh && send_board_test_notification {config_path} failed'"
                        subprocess.run(cmd, shell=True, check=True)
                    except Exception as e:
                        logger.error(f"发送微信通知失败: {str(e)}")
                        
            except Exception as e:
                logger.error(f"远程执行脚本失败: {str(e)}")
        
        # 关闭 SFTP 会话
        sftp.close()
        logger.info("SFTP 会话已关闭")
        
        # 关闭 SSH 连接
        ssh_client.close()
        logger.info("SSH 连接已关闭")
        
        return True
        
    except paramiko.AuthenticationException:
        logger.error("SFTP 连接失败: 认证失败")
        return False
    except paramiko.SSHException as e:
        logger.error(f"SFTP 连接失败: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"SFTP 操作失败: {str(e)}")
        return False


def main():
    """
    主函数
    """
    # 获取配置文件路径
    if len(sys.argv) < 2:
        print("用法: python run_sftp_push.py <config_path>")
        return 1
    
    config_path = sys.argv[1]
    
    try:
        # 加载配置
        config = load_config(config_path)
        
        # 设置日志
        log_level = config.get('logging', {}).get('level', 'INFO')
        log_file = config.get('logging', {}).get('log_file', 'sftp_push.log')
        logger = setup_logging(log_level, log_file)
        logger.info(f"加载配置文件成功: {config_path}")
        
        # 执行SFTP推送
        start_time = time.time()
        success = push_model(config, logger, config_path)
        end_time = time.time()
        
        logger.info(f"SFTP推送 {'成功' if success else '失败'}")
        logger.info(f"操作耗时: {end_time - start_time:.2f} 秒")
        
        return 0 if success else 1
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1
    except yaml.YAMLError as e:
        print(f"配置文件解析错误: {e}")
        return 1
    except Exception as e:
        print(f"发生未知错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
