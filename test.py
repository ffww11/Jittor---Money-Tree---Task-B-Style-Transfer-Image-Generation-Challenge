import subprocess

# 定义要运行的.sh文件列表
sh_files = [
    "sample_lx.sh",
    "sample_yar.sh",
    "sample_fw.sh"
]

# 依次运行每个.sh文件
for sh_file in sh_files:
    try:
        print(f"Running {sh_file}...")
        subprocess.run(['bash', sh_file], check=True)
        print(f"{sh_file} finished successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {sh_file}: {e}")
        break

print("All scripts processed.")