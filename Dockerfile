ARG PYTHON_VERSION=3.14.5
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY constraints-docker-cpu.txt .
# Install Torch first from the official CPU-only index. A subsequent normal
# requirements install treats this exact installed version as satisfied instead
# of selecting PyPI's CUDA local-version wheel.
RUN python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --constraint constraints-docker-cpu.txt \
    torch==2.14.1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --requirement requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
