# ─────────────────────────────────────────────────────────────────────────
# Lambda container image for the FSI Digital Bank demo FastAPI app.
# Adapted from public.ecr.aws/lambda/python:3.12 — the official AWS Lambda
# Python runtime image. Mangum bridges API Gateway events into ASGI calls.
# ─────────────────────────────────────────────────────────────────────────
FROM public.ecr.aws/lambda/python:3.12

# Install OS deps occasionally needed for pyfpdf / reportlab font rendering.
# (Cheap install; safe to skip on most demos but harmless here.)
RUN dnf install -y libjpeg-turbo zlib && dnf clean all

# Copy and install Python dependencies first so this layer caches between
# code edits.
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt --target ${LAMBDA_TASK_ROOT} \
 && pip install --no-cache-dir mangum>=0.17.0 --target ${LAMBDA_TASK_ROOT}

# Copy application code.
COPY app.py             ${LAMBDA_TASK_ROOT}/
COPY graph_agent.py     ${LAMBDA_TASK_ROOT}/
COPY lambda_handler.py  ${LAMBDA_TASK_ROOT}/

# Static files are served from S3 in production but FastAPI mounts the
# directory at import time, so we ship it. Keeping it tiny — index.html only.
COPY static/            ${LAMBDA_TASK_ROOT}/static/

# Data files (graph_seeds, faq_policies, etc.) — referenced by some endpoints
# at runtime for fallback rendering. Small enough to bundle.
COPY data/              ${LAMBDA_TASK_ROOT}/data/

# Mangum entry point — strips the /fsi_digital_bank_demo base path and hands
# the rest off to the FastAPI ASGI app.
CMD ["lambda_handler.handler"]
