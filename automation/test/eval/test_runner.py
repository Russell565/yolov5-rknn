#!/usr/bin/env python3
"""
测试执行脚本
负责执行模型测试并保存推理结果
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from utils import remove_duplicate_args


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试执行脚本')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--test_script', type=str, required=True, help='测试脚本路径')
    parser.add_argument('--all_args', type=str, required=True, help='所有测试参数')
    parser.add_argument('--dataset_config', type=str, required=False, help='数据集配置(JSON)')
    parser.add_argument('--dataset_config_file', type=str, required=False, help='数据集配置文件路径')
    parser.add_argument('--weight_path', type=str, required=True, help='权重文件路径')
    parser.add_argument('--weight_output_dir', type=str, required=True, help='权重输出目录')
    parser.add_argument('--zip_enable', type=int, default=0, help='是否启用结果压缩')
    parser.add_argument('--conda_env', type=str, required=False, help='conda环境名称')
    
    return parser.parse_args()

def load_dataset_config(args):
    """加载数据集配置，支持从字符串或文件读取"""
    if args.dataset_config_file:
        # 从文件读取
        with open(args.dataset_config_file, 'r') as f:
            return json.loads(f.read())
    elif args.dataset_config:
        # 从字符串读取
        return json.loads(args.dataset_config)
    else:
        raise ValueError("必须提供 --dataset_config 或 --dataset_config_file 参数")


def process_dataset(dataset, test_script, weight_path, weight_output_dir, all_args, zip_enable):
    """
    处理单个数据集
    
    Args:
        dataset: 数据集配置字典
        test_script: 测试脚本路径
        weight_path: 权重文件路径
        weight_output_dir: 权重输出目录
        all_args: 所有测试参数
        zip_enable: 是否启用结果压缩
    """
    dataset_name = dataset['name']
    img_path = dataset['img_path']
    img_subdir = dataset.get('img_subdir', 'images')
    label_subdir = dataset.get('label_subdir', 'labels')
    
    print(f"\n[INFO] 处理数据集: {dataset_name}")
    
    # 自动检测是否有标签文件
    full_label_path = os.path.join(img_path, label_subdir)
    has_labels = 1 if os.path.exists(full_label_path) else 0
    if has_labels:
        label_files = [f for f in os.listdir(full_label_path) if f.endswith('.txt')]
        if len(label_files) == 0:
            has_labels = 0
    
    if has_labels:
        print(f"[INFO] 检测到标签文件，自动设置has_labels=1: {full_label_path}")
    else:
        print(f"[INFO] 未检测到标签文件，自动设置has_labels=0: {full_label_path}")
    
    # 构建完整的图片路径
    full_img_path = os.path.join(img_path, img_subdir)
    print(f"[INFO] 完整图片路径: {full_img_path}")
    
    if not os.path.exists(full_img_path):
        print(f"[WARNING] 图片路径不存在: {full_img_path}，跳过该数据集")
        return
    
    # 为每个数据集创建输出子目录
    dataset_output_dir = os.path.join(weight_output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 构建测试命令
    filtered_args = []
    args_list = all_args.split()
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg == '--project' or arg == '--name':
            i += 2
        else:
            filtered_args.append(arg)
            i += 1
    filtered_args_str = ' '.join(filtered_args)
    
    test_cmd = (
        f"python {test_script} "
        f"--source {full_img_path} "
        f"--weights {weight_path} "
        f"--project {weight_output_dir} "
        f"--name {dataset_name} "
        f"--save-txt --save-conf --exist-ok --save-crop "
        f"{filtered_args_str}"
    )
    
    # 确保--save-txt参数只出现一次
    test_cmd = remove_duplicate_args(test_cmd)
    
    # 确保--conf-thres参数被正确传递（从配置中获取）
    if "--conf-thres" not in test_cmd:
        import yaml
        config = yaml.safe_load(open(args.config))
        conf_thres = config['test']['core_params']['conf_thres']
        test_cmd += f" --conf-thres {conf_thres}"
    
    # 对于分割模型，确保添加--save-conf参数
    if "segment/predict.py" in test_script and "--save-conf" not in test_cmd:
        test_cmd += " --save-conf"
    
    print(f"[INFO] 执行测试命令: {test_cmd}")
    
    # 执行测试命令
    # 使用当前Python解释器（确保使用conda环境中的Python）
    print(f"[INFO] 开始执行测试命令...")
    print(f"[INFO] 当前Python路径: {sys.executable}")
    
    # 将命令中的python替换为当前Python解释器路径
    test_cmd_with_python = test_cmd.replace('python ', f'{sys.executable} ')
    print(f"[INFO] 使用当前Python解释器执行: {test_cmd_with_python}")
    
    # 执行测试命令
    result = subprocess.run(test_cmd_with_python, shell=True, capture_output=True, text=True)
    
    print(f"[INFO] 测试命令返回码: {result.returncode}")
    
    if result.returncode != 0:
        print(f"[ERROR] 测试命令执行失败!")
        print(f"[ERROR] 标准输出:")
        print(result.stdout)
        print(f"[ERROR] 错误输出:")
        print(result.stderr)
        print(f"[WARNING] 继续测试下一个数据集...")
        return
    
    print(f"[INFO] 测试命令执行成功")
    print(f"[INFO] 标准输出:")
    print(result.stdout[:2000] if len(result.stdout) > 2000 else result.stdout)
    if result.stderr:
        print(f"[INFO] 标准错误:")
        print(result.stderr[:2000] if len(result.stderr) > 2000 else result.stderr)
    
    # 检查输出目录是否生成了内容
    print(f"[INFO] 检查输出目录: {dataset_output_dir}")
    if os.path.exists(dataset_output_dir):
        contents = os.listdir(dataset_output_dir)
        print(f"[INFO] 输出目录内容: {contents}")
        
        # 检查labels目录
        labels_dir = os.path.join(dataset_output_dir, 'labels')
        print(f"[INFO] labels目录是否存在: {os.path.exists(labels_dir)}")
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            print(f"[INFO] labels目录文件数量: {len(label_files)}")
            if label_files:
                print(f"[INFO] 前3个labels文件: {label_files[:3]}")
        else:
            print(f"[WARNING] labels目录不存在!")
    else:
        print(f"[ERROR] 输出目录不存在: {dataset_output_dir}")
    
    if result.returncode == 0:
        print(f"[SUCCESS] 测试完成: {dataset_name}")
        
        # 打印生成的目录结构，帮助调试
        print(f"[INFO] 测试结果目录结构:")
        subprocess.run(f"find '{weight_output_dir}' -type d | sort", shell=True, text=True)
        print(f"[INFO] 生成的所有文件:")
        subprocess.run(f"find '{weight_output_dir}' -type f | sort", shell=True, text=True)
        print(f"[INFO] 生成的txt文件:")
        subprocess.run(f"find '{weight_output_dir}' -name '*.txt' | head -20", shell=True, text=True)
        print(f"[INFO] 检查具体数据集目录:")
        subprocess.run(f"ls -la '{dataset_output_dir}'", shell=True, text=True)
        
        # 检查是否生成了labels目录
        labels_dir = os.path.join(dataset_output_dir, 'labels')
        print(f"[INFO] 检查labels目录: {labels_dir}")
        print(f"[INFO] labels目录是否存在: {os.path.exists(labels_dir)}")
        if os.path.exists(labels_dir):
            print(f"[INFO] labels目录内容: {os.listdir(labels_dir)[:10]}")
        else:
            print(f"[WARNING] labels目录不存在，检查测试命令是否正确")
            print(f"[WARNING] 测试命令: {test_cmd}")
        
        # 压缩结果（如果启用）
        if zip_enable:
            print(f"[INFO] 压缩测试结果...")
            zip_cmd = f"zip -r {weight_output_dir}/{dataset_name}.zip {weight_output_dir}/{dataset_name} > /dev/null 2>&1"
            zip_result = subprocess.run(zip_cmd, shell=True, capture_output=True, text=True)
            
            if zip_result.returncode == 0:
                print(f"[SUCCESS] 结果已压缩: {weight_output_dir}/{dataset_name}.zip")
            else:
                print(f"[WARNING] 结果压缩失败: {zip_result.stderr}")
    else:
        print(f"[ERROR] 测试失败: {dataset_name}")
        print(f"[ERROR] 错误信息: {result.stderr}")
        print(f"[WARNING] 继续测试下一个数据集...")


def main():
    """主函数"""
    global args
    args = parse_arguments()
    
    print("[INFO] 开始执行测试...")
    print(f"[INFO] 配置文件: {args.config}")
    print(f"[INFO] 测试脚本: {args.test_script}")
    print(f"[INFO] 权重文件: {args.weight_path}")
    print(f"[INFO] 输出目录: {args.weight_output_dir}")
    
    # 加载数据集配置（支持从文件或字符串读取）
    dataset_config = load_dataset_config(args)
    datasets = dataset_config['datasets']
    
    # 处理每个数据集
    for dataset in datasets:
        process_dataset(
            dataset=dataset,
            test_script=args.test_script,
            weight_path=args.weight_path,
            weight_output_dir=args.weight_output_dir,
            all_args=args.all_args,
            zip_enable=args.zip_enable
        )
    
    print("[SUCCESS] 所有数据集测试完成")
    return 0


if __name__ == '__main__':
    sys.exit(main())
