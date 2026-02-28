"""Audio utilities for BiasInEar benchmark."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def concat_audio(
    question: bytes,
    options: list[bytes],
    pause_ms: int = 500,
    labels: list[str] | None = None,
) -> bytes:
    """Concatenate question and option audio clips into a single audio file.

    Produces: ``[question] [pause] [label_A] [pause] [option_A] [pause] ...``

    Args:
        question: Question audio as WAV bytes.
        options: List of option audio clips as WAV bytes.
        pause_ms: Pause duration between segments in milliseconds.
        labels: Optional label audio clips as WAV bytes (e.g. spoken
            "A", "B", "C", "D"). If ``None``, labels are omitted and
            options are concatenated directly after the question.

    Returns:
        Combined audio as WAV bytes.

    Raises:
        ImportError: If ``pydub`` is not installed. Install it with
            ``pip install biasinear[audio]``.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "The 'pydub' package is required for audio utilities. "
            "Install it with: pip install biasinear[audio]"
        )

    pause = AudioSegment.silent(duration=pause_ms)
    audio = AudioSegment.from_file(io.BytesIO(question), format="wav") + pause

    for i, opt in enumerate(options):
        if labels is not None and i < len(labels):
            audio += AudioSegment.from_file(io.BytesIO(labels[i]), format="wav") + pause
        audio += AudioSegment.from_file(io.BytesIO(opt), format="wav") + pause

    buf = io.BytesIO()
    audio.export(buf, format="wav")
    return buf.getvalue()


def compress_audio(audio: bytes, target_format: str = "mp3") -> bytes:
    """Compress audio to a target format.

    Useful for reducing file size before sending to APIs with input size
    limits.

    Args:
        audio: Input audio as WAV bytes.
        target_format: Output format (default ``"mp3"``).

    Returns:
        Compressed audio as bytes.
    """
    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "The 'pydub' package is required for audio utilities. "
            "Install it with: pip install biasinear[audio]"
        )

    seg = AudioSegment.from_file(io.BytesIO(audio), format="wav")
    buf = io.BytesIO()
    seg.export(buf, format=target_format)
    return buf.getvalue()


def chunk_audio(audio: bytes, max_bytes: int) -> list[bytes]:
    """Split audio into chunks that each fit within a byte-size limit.

    The audio is split at evenly-spaced points in time so that each
    chunk, when exported as WAV, stays under *max_bytes*.

    Args:
        audio: Input audio as WAV bytes.
        max_bytes: Maximum byte size per chunk.

    Returns:
        List of WAV byte chunks.
    """
    if len(audio) <= max_bytes:
        return [audio]

    try:
        from pydub import AudioSegment
    except ImportError:
        raise ImportError(
            "The 'pydub' package is required for audio utilities. "
            "Install it with: pip install biasinear[audio]"
        )

    seg = AudioSegment.from_file(io.BytesIO(audio), format="wav")
    total_ms = len(seg)

    # Estimate number of chunks needed
    num_chunks = max(2, -(-len(audio) // max_bytes))  # ceiling division

    chunks: list[bytes] = []
    chunk_ms = total_ms // num_chunks

    for start in range(0, total_ms, chunk_ms):
        part = seg[start : start + chunk_ms]
        buf = io.BytesIO()
        part.export(buf, format="wav")
        chunks.append(buf.getvalue())

    return chunks
