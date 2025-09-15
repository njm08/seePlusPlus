# Utility functions for build scripts in the seePlusPlus project.
#
# Author: Niklas Meier

import os
import subprocess
import sys
from pathlib import Path

from enum import Enum

class BuildTarget(Enum):
    DEBUG = "Debug"
    RELEASE = "Release"

class BuildSite(Enum):
    LOCAL = "Local"
    CONTAINER = "Container"

def get_root_path() -> Path:
    """Get the root path of the project.

    Returns:
        Path: Path object pointing to the root (platform independent).
    """
    root_path = Path(__file__).parent.parent.parent.absolute()
    return root_path


def get_source_path() -> Path:
    """Get the src dir.

    Returns:
        Path: Path object pointing to the src dir.
    """
    src_path = Path(get_root_path(), "src").absolute()
    return src_path

def get_build_path(root_path, build_target: BuildTarget, build_site: BuildSite) -> Path:
    """Get the build dir.

    Args:
        root_path (Path, str): Path to the root directory.
        build_target (BuildTarget): Build target (Debug or Release).
        build_site (BuildSite): Build site (Local or Container).

    Returns:
        Path: Path object pointing to the build dir.
    """
    build_path = Path(root_path, "build", build_site.value, build_target.value).absolute()
    return build_path

def create_build_dir(root_path, build_target: BuildTarget, build_site: BuildSite):
    """Creates a build directory if it does not yet exist (platform independent).

    Args:
        root_path(Path, str): Root path.
        build_target (BuildTarget): Build target (Debug or Release).
        build_site (BuildSite): Build site (Local or Container).
    """
    build_path = get_build_path(root_path, build_target, build_site)
    build_path.mkdir(parents=True, exist_ok=True)

def create_cmake_build_command(root_path, build_target: BuildTarget, build_site: BuildSite) -> str:
    """Create commands to compile the project with CMake.

    Args:
        root_path(Path, str): Root path.
        build_target (BuildTarget): Build target (Debug or Release).
        build_site (BuildSite): Build site (Local or Container).

    Returns:
        str: Command to build and command to compile.
    """
    build_path = get_build_path(root_path, build_target, build_site)
    return f"cmake -S {root_path} -B {build_path} -DCMAKE_BUILD_TYPE={build_target.value} && cmake --build {build_path} --parallel"

def run_build(build_command):
    """Run the build.

    Args:
        build_command (str): Build command used to build the project.
    """
    print("Running build ...")
    print(f"CMD: {build_command}")
    try:
        subprocess.check_call(build_command, shell=True)
        print(f"\n✅ Build finished.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def build_project_local(build_target: BuildTarget):
    """Build the project locally.

    Args:
        build_target (BuildTarget): Build either debug or release.
    """
    # Settings
    build_site = BuildSite.LOCAL

    # Make the build directory.
    create_build_dir(get_root_path(), build_target, build_site)

    # Run the build locally with CMake.
    build_command = create_cmake_build_command(get_root_path(), build_target, build_site) 
    run_build(build_command)

def pull_container():
    """Pull the container on Github.
    """
    print("Pulling the container...")
    subprocess.run(["docker", "pull", "ghcr.io/njm08/seeplusplus-opencv:latest"], check=True)

def build_containerized(build_target: BuildTarget):
    """Build in a container.

    Args:
        build_target (BuildTarget): Build either debug or release.
    """

    # Settings
    build_site = BuildSite.CONTAINER

    # Pull the container first. If it is already up-to-date this will not take much time.
    pull_container()

    # Make the build directory.
    create_build_dir(get_root_path(), build_target, build_site)

    docker_workspace_path = "/workspace"
    build_command = create_cmake_build_command(docker_workspace_path, build_target, build_site)
    # Docker run command. "-v" mounts a volume from the host machine. "-w" sets the working directory.
    docker_cmd = f"docker run --rm -v {get_root_path()}:{docker_workspace_path} -w {docker_workspace_path} seeplusplus/cpp-opencv:latest bash -c '{build_command}'"
    # Run the container and build the project.
    run_build(docker_cmd)
