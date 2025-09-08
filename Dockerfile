# Use an Ubuntu image as base. 
# An alpine image is smaller but harder to get everything to run with OpenCV. So it was chosen to use Ubuntu, since the size doesnt really matter.
FROM ubuntu:22.04

# Prevent interactive installations, for example setting the time zone during installation of OpenCV.
ENV DEBIAN_FRONTEND=noninteractive 
ENV OPENCV_VERSION=4.12.0

# Install required dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    ca-certificates \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libjpeg8 libpng16-16 libtiff5 ffmpeg v4l-utils libgtk-3-0 pkg-config bash \
    && rm -rf /var/lib/apt/lists/*

# Download and build OpenCV
WORKDIR /opt

# Download OpenCV source
RUN wget -q https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.zip -O opencv_${OPENCV_VERSION}.zip \
    && unzip opencv_${OPENCV_VERSION}.zip \
    && rm opencv_${OPENCV_VERSION}.zip

WORKDIR /opt/opencv-${OPENCV_VERSION}/build

# Create build files for OpenCV
RUN cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D BUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,dnn \
          -D BUILD_SHARED_LIBS=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_JAVA=OFF \
          -D BUILD_opencv_python3=OFF \
          -D WITH_IPP=OFF \
          -D WITH_TBB=ON \
          -D WITH_OPENMP=ON \
          -D WITH_OPENCL=OFF \
          -D WITH_FFMPEG=ON \
          -D WITH_GSTREAMER=OFF \
          -D WITH_QT=OFF \
          -D WITH_GTK=ON \
          ..

# Compile OpenCV
RUN make -j$(nproc)
# Install OpenCV and update the systems cache of shared libraries.
RUN make install && ldconfig
# Check that OpenCV was installed correctly and output the version.
RUN pkg-config --modversion opencv4

# Cleanup sources to reduce image size
WORKDIR /opt
# Remove sources to reduce image size
RUN rm -rf opencv-${OPENCV_VERSION}

# Set the working directory
WORKDIR /app

# Create a non-root user to run the application. This is considered a best practice for security.
ARG USERNAME=shs
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# Print an entry point message with the tool versions.
COPY tools/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
# Set the user to run all next commands.
USER $USERNAME

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]