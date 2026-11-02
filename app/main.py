"""FastAPI application factory.

Run locally:  uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.errors import AppError
from app.logging_config import configure_logging
from app.services import Services, get_services

logger = logging.getLogger(__name__)


def create_app(services_factory: Callable[[], Services] | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Hybrid-RAG + text-to-SQL research agent orchestrated with LangGraph.",
    )
    # Services are built lazily on first request so the app starts even when
    # models or the database are not yet available.
    app.state.get_services = services_factory or get_services

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.code, "message": exc.message, "details": exc.details},
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "message": "An unexpected error occurred.", "details": {}},
        )

    app.include_router(router)
    return app


app = create_app()
