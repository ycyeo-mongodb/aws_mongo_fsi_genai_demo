"""
AWS Lambda entry point. Wraps the FastAPI ASGI app with Mangum so it can
respond to API Gateway HTTP API v2 events.

The CloudFront distribution forwards requests under the path
    /fsi_digital_bank_demo/api/*
to the asean_sa_yc API Gateway, which proxies them here. Mangum's
`api_gateway_base_path` parameter strips that prefix before the request
hits FastAPI, so the app's own routes (defined as /api/...) match cleanly.

Local development is unaffected — uvicorn never imports this module.
"""

from __future__ import annotations

import logging

from mangum import Mangum

from app import app

logging.getLogger().setLevel(logging.INFO)

# `lifespan="off"` is mandatory: Lambda spins each container up cold and
# the FastAPI startup event would otherwise block the first invocation
# without ever firing the shutdown event.
handler = Mangum(
    app,
    lifespan="off",
    api_gateway_base_path="fsi_digital_bank_demo",
)
