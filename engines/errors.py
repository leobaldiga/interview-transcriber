"""Exception taxonomy shared by all transcription engines.

Each failure mode gets its own type so callers can react differently: a
`ServerUnreachableError` means "start the container", a `ModelNotLoadedError`
means "the server is up but `-m` pointed at nothing usable", and an
`InferenceHTTPError` means the request itself was rejected.
"""


class TranscriptionError(RuntimeError):
    """Base class for every engine failure."""


class EngineUnavailableError(TranscriptionError):
    """The engine cannot serve requests right now (health check territory)."""


class ServerUnreachableError(EngineUnavailableError):
    """Connection refused / DNS failure / socket error — the server is not running."""


class ModelNotLoadedError(EngineUnavailableError):
    """The server answered, but it has no model loaded (or is still loading one)."""


class InferenceHTTPError(TranscriptionError):
    """The server returned a non-2xx status for an inference request."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class TranscriptionTimeoutError(TranscriptionError):
    """Connect or read timeout while talking to the engine."""


class MalformedResponseError(TranscriptionError):
    """The engine replied with something we could not parse into a TranscriptResult."""
