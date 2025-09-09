# Build project locally.
#
# Author: Niklas Meier

from build_utilities import BuildTarget, build_project_local

if __name__ == "__main__":
    build_project_local(BuildTarget.RELEASE)