#!/usr/bin/env python3
"""
调试检查脚本
检查测试结果目录结构和文件内容
"""

import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_check.py <result_output_dir>")
        sys.exit(1)
    
    result_output_dir = sys.argv[1]
    
    print("=" * 60)
    print("调试检查脚本")
    print("=" * 60)
    print(f"结果输出目录: {result_output_dir}")
    print(f"目录是否存在: {os.path.exists(result_output_dir)}")
    
    if os.path.exists(result_output_dir):
        print("\n目录内容:")
        for item in os.listdir(result_output_dir):
            item_path = os.path.join(result_output_dir, item)
            if os.path.isdir(item_path):
                print(f"  [DIR] {item}")
            else:
                print(f"  [FILE] {item}")
        
        # 检查权重目录（如best）
        weight_dirs = [d for d in os.listdir(result_output_dir) if os.path.isdir(os.path.join(result_output_dir, d))]
        
        for weight_dir in weight_dirs:
            weight_path = os.path.join(result_output_dir, weight_dir)
            print(f"\n{'='*60}")
            print(f"检查权重目录: {weight_path}")
            print(f"目录内容:")
            
            if os.path.exists(weight_path):
                for item in os.listdir(weight_path):
                    item_path = os.path.join(weight_path, item)
                    if os.path.isdir(item_path):
                        num_files = len([f for f in os.listdir(item_path) if f.endswith('.txt') or f.endswith('.jpg')])
                        print(f"  [DIR] {item} (文件数: {num_files})")
                    else:
                        print(f"  [FILE] {item}")
                
                # 检查labels目录
                labels_dir = os.path.join(weight_path, 'labels')
                print(f"\n检查labels目录: {labels_dir}")
                print(f"目录是否存在: {os.path.exists(labels_dir)}")
                
                if os.path.exists(labels_dir):
                    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
                    print(f"labels文件数量: {len(label_files)}")
                    
                    if label_files:
                        print("\n前3个标签文件内容:")
                        for i, label_file in enumerate(label_files[:3]):
                            label_path = os.path.join(labels_dir, label_file)
                            print(f"\n文件 {i+1}: {label_file}")
                            with open(label_path, 'r') as f:
                                content = f.read()
                                print(f"内容:\n{content}")
            
            # 检查visualization目录
            viz_dir = os.path.join(result_output_dir, 'visualization')
            print(f"\n检查visualization目录: {viz_dir}")
            print(f"目录是否存在: {os.path.exists(viz_dir)}")
            
            if os.path.exists(viz_dir):
                viz_files = []
                for root, dirs, files in os.walk(viz_dir):
                    for f in files:
                        if f.endswith('.jpg'):
                            viz_files.append(os.path.relpath(os.path.join(root, f), viz_dir))
                
                print(f"对比图数量: {len(viz_files)}")
                if viz_files:
                    print("前3个对比图:")
                    for f in viz_files[:3]:
                        print(f"  {f}")
        
        # 检查Excel文件
        excel_files = [f for f in os.listdir(result_output_dir) if f.endswith('.xlsx')]
        print(f"\n{'='*60}")
        print("检查Excel文件:")
        for excel_file in excel_files:
            excel_path = os.path.join(result_output_dir, excel_file)
            print(f"  文件: {excel_file}")
            print(f"  大小: {os.path.getsize(excel_path)} bytes")
    else:
        print("目录不存在！")

if __name__ == '__main__':
    main()
