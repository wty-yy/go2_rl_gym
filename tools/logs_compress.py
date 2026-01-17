# -*- coding: utf-8 -*-
'''
@File    : batch_compress.py
@Time    : 2026/01/09 17:49:54
@Author  : wty-yy, Gemini 3
@Version : 1.0
@Blog    : https://wty-yy.github.io/
@Desc    : None
'''
import os
import subprocess
from pathlib import Path

def smart_compress(logs_root):
    logs_path = Path(logs_root).resolve()
    if not logs_path.exists():
        print(f"❌ 找不到目录: {logs_root}")
        return

    # 遍历 logs 下的第一层子目录 (cts_vanilla, go2_moe_cts 等)
    projects = [d for d in logs_path.iterdir() if d.is_dir()]

    for project in projects:
        project_name = project.name
        print(f"\n🚀 正在处理项目: {project_name}")

        # 1. 搜寻需要包含的内容
        include_items = []

        # 检查是否存在 exported 文件夹
        if (project / "exported").exists():
            include_items.append("exported")

        # 搜寻所有包含 tfevents 的文件夹 (如 Jan04_15-55-59_)
        # 我们寻找 events 文件，然后取其父目录名（相对于项目根目录）
        event_folders = set()
        for event_file in project.rglob("events.out.tfevents*"):
            # 计算相对于项目根目录的路径
            relative_folder = event_file.parent.relative_to(project)
            event_folders.add(str(relative_folder))
        
        include_items.extend(list(event_folders))

        if not include_items:
            print(f"⚠️  跳过 {project_name}: 未发现符合条件的训练数据或 exported 文件夹")
            continue

        # 2. 构造压缩命令
        output_zst = logs_path / f"{project_name}.tar.zst"
        
        # 命令解释:
        # -C: 切换到项目所在目录，这样压缩包内的路径不会带一堆无用的父级前缀
        # --exclude='*.pt': 显式排除所有模型权重文件
        # -T0: zstd 开启全核并行
        tar_cmd = [
            "tar",
            "-I", "zstd -T0 -3",
            "-C", str(project), 
            "--exclude=*.pt",
            "--exclude=*.pth", # 预防万一有 .pth
            "-cf", str(output_zst)
        ] + include_items

        print(f"📦 正在打包 (已排除 .pt 文件)...")
        
        try:
            # 执行压缩
            subprocess.run(tar_cmd, check=True)
            
            # 统计结果
            final_size = output_zst.stat().st_size / (1024 * 1024)
            print(f"✅ 完成! 压缩包: {output_zst.name} ({final_size:.2f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"❌ {project_name} 压缩失败: {e}")

if __name__ == "__main__":
    # 执行目录
    TARGET_LOGS_DIR = "./logs"
    smart_compress(TARGET_LOGS_DIR)
