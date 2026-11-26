FROM python:3.12-slim-bookworm@sha256:44cf1793efa7c7a476bb0f2cff81443f4a072c90c1cb127e2924933daa3e8c64
COPY install.py nvrtc-wheels.json /build/
RUN python -I /build/install.py
ENV LD_LIBRARY_PATH=/opt/nvrtc
LABEL org.weakbench.kernel-cuda="cuda-native-v1"
ENV CUDA_CACHE_DISABLE=1
ENV NVIDIA_DRIVER_CAPABILITIES=compute
COPY worker.py /opt/worker.py
USER 65534:65534
ENTRYPOINT ["python","-I","-B","/opt/worker.py"]
