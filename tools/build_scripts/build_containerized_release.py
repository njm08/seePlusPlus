# Build project in a container.
#
# Author: Niklas Meier

from build_utilities import build_containerized, BuildTarget

if __name__ == "__main__":
    build_containerized(BuildTarget.RELEASE)