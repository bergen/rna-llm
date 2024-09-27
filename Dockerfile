FROM nvcr.io/nvidia/pytorch:22.07-py3

WORKDIR /wdr

COPY requirements.txt .

RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY . .

RUN git submodule update --init && cd flash-attention && git submodule update --init && pip install . --no-build-isolation && \
    cd csrc/layer_norm && pip install . --no-build-isolation