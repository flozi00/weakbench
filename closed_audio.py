"""Private audio tasks accept a bounded canonical 16 kHz mono signed-PCM WAV."""
import base64
import struct


SAMPLE_RATE = 16_000
MAX_FRAMES = 15 * SAMPLE_RATE
HEADER = struct.Struct("<4sI4s4sIHHIIHH4sI")
MAX_WAV_BYTES = HEADER.size + 2 * MAX_FRAMES


def validate_media(media):
    """Accept a single uncompressed PCM data chunk, with no ancillary chunks or trailing bytes."""
    if not isinstance(media, dict) or set(media) != {"mime_type", "data"} or media["mime_type"] != "audio/wav":
        raise ValueError("invalid private WAV media")
    encoded = media["data"]
    if not isinstance(encoded, str) or not 0 < len(encoded) <= 4 * ((MAX_WAV_BYTES + 2) // 3):
        raise ValueError("invalid private WAV size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError:
        raise ValueError("invalid private WAV encoding") from None
    if not HEADER.size < len(raw) <= MAX_WAV_BYTES or base64.b64encode(raw).decode("ascii") != encoded:
        raise ValueError("invalid private WAV encoding or size")
    pcm_bytes = len(raw) - HEADER.size
    expected_header = (b"RIFF", len(raw) - 8, b"WAVE", b"fmt ", 16, 1, 1,
                       SAMPLE_RATE, 2 * SAMPLE_RATE, 2, 16, b"data", pcm_bytes)
    if pcm_bytes % 2 or HEADER.unpack_from(raw) != expected_header:
        raise ValueError("invalid private PCM WAV")
