FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LEXEUROPA_PUBLIC=1 \
    LEXEUROPA_DATA_DIR=/app/data \
    HF_HOME=/home/user/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user

# The code is pinned to one commit of the public repository (substituted by scripts/deploy_space.py).
ARG LEXEUROPA_REF=main
RUN git clone https://github.com/lumenworksco/-LexEuropa.git /app/src \
 && cd /app/src && git checkout "$LEXEUROPA_REF"

# CPU-only torch first, so the sentence-transformers install does not pull the multi-gigabyte CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -e "/app/src[rerank]"

# The library: chunks, instruments and the dense index (plain arrays, no pickle). No credentials are baked in.
COPY --chown=user data /app/data

USER user
WORKDIR /app/src
EXPOSE 7860
CMD ["python", "-m", "lexeuropa.cli", "serve-public", "--port", "7860"]
