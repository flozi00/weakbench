"""Validate the embedded, static PNG raster used by optional private visual tasks."""
import base64
import binascii
import struct
import zlib

MAX_PNG_BYTES = 1 << 20
MAX_DIMENSION = 2048
MAX_RASTER_BYTES = 8 << 20
SIGNATURE = b"\x89PNG\r\n\x1a\n"
# Bits per sample allowed for each PNG colour type and the number of channels per pixel.
FORMATS = {0: ((1, 2, 4, 8, 16), 1), 2: ((8, 16), 3), 3: ((1, 2, 4, 8), 1),
           4: ((8, 16), 2), 6: ((8, 16), 4)}
ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def _passes(w, h, bits_per_pixel, interlace):
    patterns = ADAM7 if interlace else ((0, 0, 1, 1),)
    for x, y, dx, dy in patterns:
        if x < w and y < h:
            width, rows = (w - x + dx - 1) // dx, (h - y + dy - 1) // dy
            yield 1 + (width * bits_per_pixel + 7) // 8, rows


def validate_media(media):
    """Return the validated raster dimensions. Reject malformed media without echoing it."""
    if not isinstance(media, dict) or set(media) != {"mime_type", "data"} or media["mime_type"] != "image/png":
        raise ValueError("invalid private PNG media")
    encoded = media["data"]
    if not isinstance(encoded, str) or not 0 < len(encoded) <= 4 * ((MAX_PNG_BYTES + 2) // 3):
        raise ValueError("invalid private PNG size")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ValueError("invalid private PNG encoding") from None
    if (len(png) > MAX_PNG_BYTES or base64.b64encode(png).decode("ascii") != encoded
            or not png.startswith(SIGNATURE)):
        raise ValueError("invalid private PNG encoding")
    pos, palette, seen_data = len(SIGNATURE), 0, False
    previous = None
    transparency = False
    compressed = bytearray()
    while pos + 12 <= len(png):
        length, kind = struct.unpack_from(">I4s", png, pos)
        end = pos + 8 + length
        if end + 4 > len(png) or not kind.isalpha() or kind[2] & 32:
            raise ValueError("invalid private PNG chunk")
        payload = memoryview(png)[pos + 8:end]
        if struct.unpack_from(">I", png, end)[0] != zlib.crc32(payload, zlib.crc32(kind)):
            raise ValueError("invalid private PNG checksum")
        if previous is None and kind != b"IHDR":
            raise ValueError("missing private PNG header")
        pos = end + 4
        if kind == b"IHDR":
            if previous is not None or length != 13:
                raise ValueError("invalid private PNG header")
            w, h, depth, color, method, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (not 1 <= w <= MAX_DIMENSION or not 1 <= h <= MAX_DIMENSION or color not in FORMATS
                    or depth not in FORMATS[color][0] or method or filtering or interlace not in (0, 1)):
                raise ValueError("unsupported private PNG header")
            passes = list(_passes(w, h, depth * FORMATS[color][1], interlace))
            raster_length = sum(stride * rows for stride, rows in passes)
            if raster_length > MAX_RASTER_BYTES:
                raise ValueError("oversize private PNG raster")
        elif kind == b"PLTE":
            if palette or transparency or seen_data or color in (0, 4) or not 0 < length <= 768 or length % 3:
                raise ValueError("invalid private PNG palette")
            palette = length // 3
            if color == 3 and palette > 2 ** depth:
                raise ValueError("invalid private PNG palette")
        elif kind == b"tRNS":
            allowed = ((color == 0 and length == 2) or (color == 2 and length == 6)
                       or (color == 3 and 0 < length <= palette))
            if transparency or seen_data or not allowed:
                raise ValueError("invalid private PNG transparency")
            transparency = True
        elif kind == b"IDAT":
            if (seen_data and previous != b"IDAT") or (color == 3 and not palette):
                raise ValueError("invalid private PNG image data")
            seen_data = True
            compressed.extend(payload)
        elif kind == b"IEND":
            if length or pos != len(png) or not seen_data:
                raise ValueError("invalid private PNG end")
            break
        elif kind in (b"acTL", b"fcTL", b"fdAT", b"zTXt", b"iCCP", b"iTXt") or not kind[0] & 32:
            # Images in the release are static; compressed ancillary streams are outside its raster budget.
            raise ValueError("unsupported private PNG chunk")
        previous = kind
    else:
        raise ValueError("truncated private PNG")
    decoder = zlib.decompressobj()
    try:
        pixels = decoder.decompress(compressed, raster_length + 1)
    except zlib.error:
        raise ValueError("invalid private PNG compressed data") from None
    if len(pixels) != raster_length or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("invalid private PNG raster length")
    offset = 0
    for stride, rows in passes:
        end = offset + stride * rows
        if any(filter_type > 4 for filter_type in pixels[offset:end:stride]):
            raise ValueError("invalid private PNG row filter")
        offset = end
    return w, h
