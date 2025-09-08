import os
import subprocess
import sys
from pathlib import Path

def build_project():
    root_path = Path(__file__).parent.parent.parent.absolute()
    build_path = Path(root_path, "build", "Debug").absolute()
    print(f"Project root: {root_path}")

    # Docker run command. "-v" mounts a volume from the host machine. "-w" sets the working directory.
    docker_cmd = [
        "docker", "run", "--rm",
        "-v", f"{root_path}:/workspace",
        "-w", "/workspace",
        "seeplusplus/cpp-opencv:latest",
        "bash", "-c",
        "mkdir -p build && cd build && cmake .. && make -j$(nproc)"
    ]

    print("Running build inside Docker...")

    try:
        subprocess.check_call(docker_cmd)
        print(f"\n✅ Build finished.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with exit code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    build_project()