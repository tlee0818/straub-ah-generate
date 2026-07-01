"""Domain-specific exceptions for the Podcast Worker Service."""


class PodcastWorkerError(Exception):
    """Base exception for all podcast worker errors."""
    pass


class ConfigurationError(PodcastWorkerError):
    """Raised when required configuration is missing or invalid."""
    pass


class LLMError(PodcastWorkerError):
    """Raised when an LLM provider call fails."""
    pass


class TTSError(PodcastWorkerError):
    """Raised when TTS synthesis fails."""
    pass


class BeatGenerationError(PodcastWorkerError):
    """Raised when beat generation fails."""
    pass


class AudioMixError(PodcastWorkerError):
    """Raised when audio mixing fails."""
    pass


class JobNotFoundError(PodcastWorkerError):
    """Raised when a job ID is not found."""
    pass


class JobNotCompleteError(PodcastWorkerError):
    """Raised when trying to access results of an incomplete job."""
    pass


class ScriptNotFoundError(PodcastWorkerError):
    """Raised when a script ID is not found."""
    pass


class FileOperationError(PodcastWorkerError):
    """Raised when file read/write/convert operations fail."""
    pass
