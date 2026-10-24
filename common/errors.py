"""Shared JSON error handling for all services.

Response shape (backward compatible with the original ``{"error": "..."}``)::

    {"error": "<human-readable message>", "code": "<stable_code>"}
    {"error": "Internal server error", "code": "internal_error", "error_id": "<id>"}  # 5xx only

Exception text is written to the log, never returned to the client.
"""

import uuid

from flask import jsonify
from werkzeug.exceptions import HTTPException


class ServiceError(Exception):
    """An expected error that maps to a client-facing response."""

    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def error_response(code: str, message: str, status: int, error_id: str | None = None):
    body = {"error": message, "code": code}
    if error_id:
        body["error_id"] = error_id
    return jsonify(body), status


def register_error_handlers(app, logger):
    """Attach JSON error handlers to a Flask app."""

    @app.errorhandler(ServiceError)
    def _service_error(e: ServiceError):
        logger.warning("Request rejected: %s (%s)", e.code, e.status)
        return error_response(e.code, e.message, e.status)

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        if e.code == 413:
            return error_response("payload_too_large", "Upload exceeds the size limit", 413)
        return error_response(e.name.lower().replace(" ", "_"), e.name, e.code)

    @app.errorhandler(Exception)
    def _unexpected(e: Exception):
        error_id = uuid.uuid4().hex[:12]
        logger.exception("Unhandled error (error_id=%s)", error_id)
        return error_response("internal_error", "Internal server error", 500, error_id)
