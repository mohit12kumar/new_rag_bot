import time
import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

# Set up logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_agent_api")

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    HTTP Middleware that logs details about incoming requests,
    execution duration, and HTTP response codes.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()
        
        # Log request receipt
        logger.info(f"Incoming request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            # Log response metrics
            logger.info(
                f"Response: {request.method} {request.url.path} "
                f"Status: {response.status_code} "
                f"Duration: {process_time:.2f}ms"
            )
            return response
        except Exception as exc:
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"Request Failed: {request.method} {request.url.path} "
                f"Duration: {process_time:.2f}ms "
                f"Error: {str(exc)}",
                exc_info=True
            )
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "An internal server error occurred while processing the request.",
                    "error": str(exc)
                }
            )

def setup_cors(app) -> None:
    """
    Configures Cross-Origin Resource Sharing (CORS) on the FastAPI app.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins for local testing and flexibility
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
