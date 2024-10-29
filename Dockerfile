FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

ENV DEBIAN_FRONTEND noninteractive

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# https://github.com/google-ai-edge/ai-edge-torch/tree/v0.2.0?tab=readme-ov-file#update-ld_library_path-if-necessary
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# https://github.com/google-ai-edge/ai-edge-torch/issues/145#issuecomment-2283272821
ENV PJRT_DEVICE=CPU

ENV PYTHONPATH $PYTHONPATH:/workspace
WORKDIR /workspace
