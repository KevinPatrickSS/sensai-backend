import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
from os.path import exists
from api.config import UPLOAD_FOLDER_NAME
from api.utils.logging import logger
from api.routes import (
    auth,
    batch,
    code,
    cohort,
    course,
    org,
    task,
    chat,
    user,
    milestone,
    hva,
    file,
    ai,
    scorecard,
    integration,
    evalgraph,
)
from api.websockets import router as websocket_router
from api.scheduler import scheduler
from api.settings import settings
import sentry_sdk


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application")
    scheduler.start()
    os.makedirs(settings.local_upload_folder, exist_ok=True)
    yield
    logger.info("Shutting down application")
    scheduler.shutdown()


if settings.sentry_dsn:
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.sentry_environment,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(
        f"Incoming request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    start_time = asyncio.get_event_loop().time()
    try:
        response = await call_next(request)
        process_time = asyncio.get_event_loop().time() - start_time
        logging.info(
            f"Request completed: {request.method} {request.url.path} "
            f"- Status: {response.status_code} - Duration: {process_time:.4f}s"
        )
        return response
    except Exception as e:
        process_time = asyncio.get_event_loop().time() - start_time
        logging.error(
            f"Error processing request: {request.method} {request.url.path} "
            f"- Error: {str(e)} - Duration: {process_time:.4f}s",
            exc_info=True,
        )
        raise


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if exists(settings.local_upload_folder):
    app.mount(
        f"/{UPLOAD_FOLDER_NAME}",
        StaticFiles(directory=settings.local_upload_folder),
        name="uploads",
    )

app.include_router(file.router,           prefix="/file",           tags=["file"])
app.include_router(ai.router,             prefix="/ai",             tags=["ai"])
app.include_router(auth.router,           prefix="/auth",           tags=["auth"])
app.include_router(batch.router,          prefix="/batches",        tags=["batches"])
app.include_router(task.router,           prefix="/tasks",          tags=["tasks"])
app.include_router(chat.router,           prefix="/chat",           tags=["chat"])
app.include_router(user.router,           prefix="/users",          tags=["users"])
app.include_router(org.router,            prefix="/organizations",  tags=["organizations"])
app.include_router(cohort.router,         prefix="/cohorts",        tags=["cohorts"])
app.include_router(course.router,         prefix="/courses",        tags=["courses"])
app.include_router(milestone.router,      prefix="/milestones",     tags=["milestones"])
app.include_router(scorecard.router,      prefix="/scorecards",     tags=["scorecards"])
app.include_router(code.router,           prefix="/code",           tags=["code"])
app.include_router(hva.router,            prefix="/hva",            tags=["hva"])
app.include_router(websocket_router,      prefix="/ws",             tags=["websockets"])
app.include_router(integration.router,    prefix="/integrations",   tags=["integrations"])
app.include_router(evalgraph.router,      prefix="/api/evalgraph",  tags=["evalgraph"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(
        f"Unhandled exception: {type(exc).__name__}: {str(exc)} "
        f"on {request.method} {request.url.path}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logging.warning(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code >= 500:
        logging.error(
            f"HTTP {exc.status_code} error on {request.method} {request.url.path}: {exc.detail}"
        )
    else:
        logging.info(
            f"HTTP {exc.status_code} on {request.method} {request.url.path}: {exc.detail}"
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "ok"}


@app.api_route("/sentry-debug", methods=["GET"])
async def sentry_debug():
    raise Exception("Sentry test error")