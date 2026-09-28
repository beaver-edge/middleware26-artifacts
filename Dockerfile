FROM continuumio/miniconda3:24.11.1-0@sha256:6a66425f001f739d4778dd732e020afeb06175f49478fafc3ec673658d61550b

RUN conda create -y -n datasci2 --override-channels -c conda-forge python=3.11 pip && conda clean -afy
ENV PATH=/opt/conda/envs/datasci2/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    ARTIFACT_CONTAINER=1 \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl build-essential openssh-client time ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /artifact
COPY requirements.txt requirements-optional.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt -r requirements-optional.txt

# The existing workflow uses arduino:mbed:nano33ble (core 3.3.0).
ARG TARGETARCH
RUN case "$TARGETARCH" in \
        amd64) arduino_arch=64bit ;; \
        arm64) arduino_arch=ARM64 ;; \
        *) echo "Unsupported Docker architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl --http1.1 -fsSL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 180 \
    "https://downloads.arduino.cc/arduino-cli/arduino-cli_1.1.1_Linux_${arduino_arch}.tar.gz" \
    -o /tmp/arduino-cli.tar.gz \
    && tar -xzf /tmp/arduino-cli.tar.gz -C /usr/local/bin arduino-cli \
    && rm /tmp/arduino-cli.tar.gz
RUN arduino-cli core update-index && arduino-cli core install arduino:mbed@3.3.0
RUN arduino-cli lib install 'Arduino_APDS9960@1.0.4' 'Arduino_LSM9DS1@1.1.1' 'ArduinoBLE@1.3.7'
COPY vendor/ /opt/artifact-vendor/
RUN mkdir -p /root/Arduino/libraries && tar -xzf /opt/artifact-vendor/Arduino_TensorFlowLite-2.4.0-ALPHA.tar.gz -C /root/Arduino/libraries
COPY src/ src/
COPY tests/ tests/
COPY scripts/ scripts/
COPY inputs/ inputs/
COPY README.md example.env ./
RUN mkdir -p logs tmp work/data work/convert work/ardsketch \
    work/pysketch/generated work/tpusketch/generated \
    && python -m pip check
CMD ["bash", "scripts/container/run-task.sh", "all"]
