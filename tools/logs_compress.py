import os
import subprocess
from pathlib import Path

def smart_compress(logs_root):
    logs_path = Path(logs_root).resolve()
    if not logs_path.exists():
        print(f"❌ Can't find directory: {logs_root}")
        return

    projects = [d for d in logs_path.iterdir() if d.is_dir()]

    for project in projects:
        project_name = project.name
        print(f"\n🚀 Processing project: {project_name}")

        include_items = []

        if (project / "exported").exists():
            include_items.append("exported")

        event_folders = set()
        for event_file in project.rglob("events.out.tfevents*"):
            relative_folder = event_file.parent.relative_to(project)
            event_folders.add(str(relative_folder))
        
        include_items.extend(list(event_folders))

        if not include_items:
            print(f"⚠️  Skipping {project_name}: No eligible training data or exported folder found")
            continue

        output_zst = logs_path / f"{project_name}.tar.zst"
        
        tar_cmd = [
            "tar",
            "-I", "zstd -T0 -3",
            "-C", str(project), 
            "--exclude=*.pt",
            "--exclude=*.pth",
            "-cf", str(output_zst)
        ] + include_items

        print(f"📦 Packaging (excluding .pt files and .pth files)...")
        
        try:
            subprocess.run(tar_cmd, check=True)
            
            final_size = output_zst.stat().st_size / (1024 * 1024)
            print(f"✅ Done! Archive: {output_zst.name} ({final_size:.2f} MB)")
        except subprocess.CalledProcessError as e:
            print(f"❌ {project_name} Compression failed: {e}")

if __name__ == "__main__":
    # Execution directory
    TARGET_LOGS_DIR = "./logs"
    smart_compress(TARGET_LOGS_DIR)
