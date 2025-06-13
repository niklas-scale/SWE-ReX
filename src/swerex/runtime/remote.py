import logging
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel
from typing_extensions import Self

from swerex.exceptions import SwerexException
from swerex.runtime.abstract import (
    AbstractRuntime,
    Action,
    CloseResponse,
    CloseSessionRequest,
    CloseSessionResponse,
    Command,
    CommandResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    IsAliveResponse,
    Observation,
    ReadFileRequest,
    ReadFileResponse,
    UploadRequest,
    UploadResponse,
    WriteFileRequest,
    WriteFileResponse,
    _ExceptionTransfer,
)
from swerex.runtime.config import RemoteRuntimeConfig
from swerex.utils.log import get_logger
from swerex.utils.wait import _wait_until_alive

__all__ = ["RemoteRuntime", "RemoteRuntimeConfig"]


class RemoteRuntime(AbstractRuntime):
    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        max_retries: int = 4,
        retry_delay: float = 1.0,
        retry_backoff: float = 2.0,
        **kwargs: Any,
    ):
        """A runtime that connects to a remote server.
        Args:
            max_retries: Maximum number of retries for HTTP requests (default: 3)
            retry_delay: Initial delay between retries in seconds (default: 1.0)
            retry_backoff: Backoff multiplier for retry delays (default: 2.0)
            **kwargs: Keyword arguments to pass to the `RemoteRuntimeConfig` constructor.
        """
        self._config = RemoteRuntimeConfig(**kwargs)
        self.logger = logger or get_logger("rex-runtime")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff = retry_backoff

        if not self._config.host.startswith("http"):
            self.logger.warning("Host %s does not start with http, adding http://", self._config.host)
            self._config.host = f"http://{self._config.host}"

    @classmethod
    def from_config(cls, config: RemoteRuntimeConfig) -> Self:
        return cls(**config.model_dump())

    def _get_timeout(self, timeout: float | None = None) -> float:
        if timeout is None:
            return self._config.timeout
        return timeout

    @property
    def _headers(self) -> dict[str, str]:
        """Request headers to use for authentication."""
        if self._config.auth_token:
            return {"X-API-Key": self._config.auth_token}
        return {}

    @property
    def _api_url(self) -> str:
        if self._config.port is None:
            return self._config.host
        return f"{self._config.host}:{self._config.port}"

    def _should_retry_exception(self, exc: Exception) -> bool:
        """Determine if an exception is retryable."""
        if isinstance(exc, requests.exceptions.ConnectionError):
            return True
        if isinstance(exc, requests.exceptions.Timeout):
            return True
        if isinstance(exc, requests.exceptions.SSLError):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            # Retry on 5xx server errors, but not 4xx client errors
            if hasattr(exc, 'response') and exc.response is not None:
                return 500 <= exc.response.status_code < 600
        return False

    def _should_retry_status_code(self, status_code: int) -> bool:
        """Determine if a status code is retryable."""
        # Retry on 5xx server errors and some network-related codes
        return 500 <= status_code < 600 or status_code in [408, 429]

    def _retry_request(self, request_func, *args, safe_to_retry: bool = True, **kwargs):
        """Execute a request function with retries."""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                response = request_func(*args, **kwargs)

                # Check if we should retry based on status code
                if (attempt < self.max_retries and 
                    safe_to_retry and 
                    self._should_retry_status_code(response.status_code)):

                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    self.logger.warning(
                        "Request failed with status %d, retrying in %.2f seconds (attempt %d/%d)",
                        response.status_code, delay, attempt + 1, self.max_retries + 1
                    )
                    time.sleep(delay)
                    continue

                return response

            except Exception as exc:
                last_exception = exc

                if (attempt < self.max_retries and 
                    safe_to_retry and 
                    self._should_retry_exception(exc)):

                    delay = self.retry_delay * (self.retry_backoff ** attempt)
                    self.logger.warning(
                        "Request failed with %s, retrying in %.2f seconds (attempt %d/%d): %s",
                        type(exc).__name__, delay, attempt + 1, self.max_retries + 1, str(exc)
                    )
                    time.sleep(delay)
                    continue

                # If not retryable or out of retries, re-raise
                raise

        # If we get here, we've exhausted retries
        if last_exception:
            raise last_exception

    def _handle_transfer_exception(self, exc_transfer: _ExceptionTransfer) -> None:
        """Reraise exceptions that were thrown on the remote."""
        if exc_transfer.traceback:
            self.logger.critical("Traceback: \n%s", exc_transfer.traceback)
        module, _, exc_name = exc_transfer.class_path.rpartition(".")
        print(module, exc_name)
        if module == "builtins":
            module_obj = __builtins__
        else:
            if module not in sys.modules:
                self.logger.debug("Module %s not in sys.modules, trying to import it", module)
                try:
                    __import__(module)
                except ImportError:
                    self.logger.debug("Failed to import module %s", module)
                    exc = SwerexException(exc_transfer.message)
                    raise exc from None
            module_obj = sys.modules[module]
        try:
            if isinstance(module_obj, dict):
                # __builtins__, sometimes
                exception = module_obj[exc_name](exc_transfer.message)
            else:
                exception = getattr(module_obj, exc_name)(exc_transfer.message)
        except (AttributeError, TypeError):
            self.logger.error(
                f"Could not initialize transferred exception: {exc_transfer.class_path!r}. "
                f"Transfer object: {exc_transfer}"
            )
            exception = SwerexException(exc_transfer.message)
        exception.extra_info = exc_transfer.extra_info
        raise exception from None

    def _handle_response_errors(self, response: requests.Response) -> None:
        """Raise exceptions found in the request response."""
        if response.status_code == 511:
            exc_transfer = _ExceptionTransfer(**response.json()["swerexception"])
            self._handle_transfer_exception(exc_transfer)
        try:
            response.raise_for_status()
        except Exception:
            self.logger.critical("Received error response: %s", response.json())
            raise

    async def is_alive(self, *, timeout: float | None = None) -> IsAliveResponse:
        """Checks if the runtime is alive.
        Internal server errors are thrown, everything else just has us return False
        together with the message.
        """
        try:
            response = self._retry_request(
                requests.get,
                f"{self._api_url}/is_alive",
                headers=self._headers,
                timeout=self._get_timeout(timeout),
                safe_to_retry=True  # is_alive is safe to retry
            )
            if response.status_code == 200:
                return IsAliveResponse(**response.json())
            elif response.status_code == 511:
                exc_transfer = _ExceptionTransfer(**response.json()["swerexception"])
                self._handle_transfer_exception(exc_transfer)
            msg = (
                f"Status code {response.status_code} from {self._api_url}/is_alive. "
                f"Message: {response.json().get('detail')}"
            )
            return IsAliveResponse(is_alive=False, message=msg)
        except requests.RequestException:
            msg = f"Failed to connect to {self._config.host}\n"
            msg += traceback.format_exc()
            return IsAliveResponse(is_alive=False, message=msg)
        except Exception:
            msg = f"Failed to connect to {self._config.host}\n"
            msg += traceback.format_exc()
            return IsAliveResponse(is_alive=False, message=msg)

    async def wait_until_alive(self, *, timeout: float = 60.0):
        return await _wait_until_alive(self.is_alive, timeout=timeout)

    def _request(self, endpoint: str, request: BaseModel | None, output_class: Any, safe_to_retry: bool = True):
        """Small helper to make requests to the server and handle errors and output."""
        response = self._retry_request(
            requests.post,
            f"{self._api_url}/{endpoint}",
            json=request.model_dump() if request else None,
            headers=self._headers,
            safe_to_retry=safe_to_retry
        )
        self._handle_response_errors(response)
        return output_class(**response.json())

    async def create_session(self, request: CreateSessionRequest) -> CreateSessionResponse:
        """Creates a new session."""
        return self._request("create_session", request, CreateSessionResponse, safe_to_retry=True)

    async def run_in_session(self, action: Action) -> Observation:
        """Runs a command in a session."""
        # This is potentially unsafe to retry as it might execute commands twice
        return self._request("run_in_session", action, Observation, safe_to_retry=False)

    async def close_session(self, request: CloseSessionRequest) -> CloseSessionResponse:
        """Closes a shell session."""
        return self._request("close_session", request, CloseSessionResponse, safe_to_retry=True)

    async def execute(self, command: Command) -> CommandResponse:
        """Executes a command (independent of any shell session)."""
        # This is potentially unsafe to retry as it might execute commands twice
        return self._request("execute", command, CommandResponse, safe_to_retry=False)

    async def read_file(self, request: ReadFileRequest) -> ReadFileResponse:
        """Reads a file"""
        return self._request("read_file", request, ReadFileResponse, safe_to_retry=True)

    async def write_file(self, request: WriteFileRequest) -> WriteFileResponse:
        """Writes a file"""
        # File writes could be unsafe to retry depending on the write mode
        # Being conservative and not retrying by default
        return self._request("write_file", request, WriteFileResponse, safe_to_retry=False)

    async def upload(self, request: UploadRequest) -> UploadResponse:
        """Uploads a file"""
        source = Path(request.source_path).resolve()
        self.logger.debug("Uploading file from %s to %s", request.source_path, request.target_path)

        def _upload_request(files, data):
            return requests.post(f"{self._api_url}/upload", files=files, data=data, headers=self._headers)

        if source.is_dir():
            # Ignore cleanup errors: See https://github.com/SWE-agent/SWE-agent/issues/1005
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
                zip_path = Path(temp_dir) / "zipped_transfer.zip"
                shutil.make_archive(str(zip_path.with_suffix("")), "zip", source)
                self.logger.debug("Created zip file at %s", zip_path)
                files = {"file": zip_path.open("rb")}
                data = {"target_path": request.target_path, "unzip": "true"}

                # Uploads might be unsafe to retry if they partially succeed
                response = self._retry_request(
                    _upload_request, files, data, safe_to_retry=False
                )
                self._handle_response_errors(response)
                return UploadResponse(**response.json())

        elif source.is_file():
            self.logger.debug("Uploading file from %s to %s", source, request.target_path)
            files = {"file": source.open("rb")}
            data = {"target_path": request.target_path, "unzip": "false"}

            # Uploads might be unsafe to retry if they partially succeed
            response = self._retry_request(
                _upload_request, files, data, safe_to_retry=False
            )
            self._handle_response_errors(response)
            return UploadResponse(**response.json())
        else:
            msg = f"Source path {source} is not a file or directory"
            raise ValueError(msg)

    async def close(self) -> CloseResponse:
        """Closes the runtime."""
        return self._request("close", None, CloseResponse, safe_to_retry=True)
