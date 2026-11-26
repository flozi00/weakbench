"""Render arena answers in disposable offline Linux containers. Only PNGs leave the runtime."""

import base64
import binascii
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import struct
import subprocess
import tempfile
import time
import uuid
import zlib

SCHEMA = "weakbench.arena.png.v1"
IMAGE = "weakbench-arena-render:1"
VIEWS = {
    "web": (("desktop", 1120, 760), ("mobile", 390, 844)),
    "three": (("front", 720, 600), ("side", 720, 600), ("rear", 720, 600)),
}
ALIASES = {"web": "web", "website": "web", "three": "three", "threejs": "three"}
MAX_INPUT = 256_000
MAX_WIRE = 18_000_000
MAX_SECONDS = 45
ERRORS = {"empty_answer", "answer_too_large", "unsupported_format", "javascript_error",
          "missing_scene_factory", "invalid_scene", "empty_scene", "external_resource"}
INFRA = {"runtime_unavailable", "runtime_failure", "render_timeout", "view_capture_failed", "webgl_unavailable"}


def _runtime():
    cli = shutil.which("docker")
    socket = os.environ.get("WEAKBENCH_ARENA_DOCKER_SOCKET", "/var/run/docker.sock")
    if not cli or not os.path.isabs(socket):
        return None, None
    command = [cli, "--host", "unix://" + socket]
    try:
        probe = subprocess.run(command + ["image", "inspect", "--format",
                               '{{.Id}} {{index .Config.Labels "org.weakbench.arena-render"}}', IMAGE],
                               capture_output=True, timeout=5, check=False)
        fields = probe.stdout.decode("ascii").strip().split()
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        return None, None
    if probe.returncode or len(fields) != 2 or fields[1] != "1" or not re.fullmatch(r"sha256:[a-f0-9]{64}", fields[0]):
        return None, None
    return command, fields[0]


def _key(image_id):
    revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:20]
    return f"arena-png-v1:offline-oci:{revision}:{image_id}"


def renderer_key():
    """Identify the local immutable image and host-side render contract."""
    image_id = _runtime()[1]
    if image_id is None:
        raise RuntimeError("Local offline renderer image unavailable")
    return _key(image_id)


def _result(kind, key, status, diagnostic, views=()):
    return {"schema": SCHEMA, "renderer_key": key, "kind": kind, "status": status,
            "diagnostic": diagnostic, "views": list(views)}


def _source(raw_text, kind):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return "", "empty_answer"
    if len(raw_text) > MAX_INPUT or len(raw_text.encode("utf-8", errors="replace")) > MAX_INPUT:
        return "", "answer_too_large"
    text = raw_text.strip()
    if text.startswith("```"):
        allowed = "html?" if kind == "web" else "javascript|js"
        block = re.fullmatch(r"```(?:(?:" + allowed + r"))?[ \t]*\r?\n(.*)\r?\n```", text, re.I | re.S)
        if not block:
            return "", "unsupported_format"
        text = block.group(1).strip()
    if not text or (kind == "web" and not text.startswith("<")):
        return "", "unsupported_format"
    return text, None


def _view_valid(view, expected):
    name, width, height = expected
    if not isinstance(view, dict) or set(view) != {"name", "width", "height", "png_base64"}:
        return False
    if (view["name"], view["width"], view["height"]) != expected or not isinstance(view["png_base64"], str):
        return False
    if type(view["width"]) is not int or type(view["height"]) is not int:
        return False
    if len(view["png_base64"]) > (width * height * 6 + 4096):
        return False
    try:
        png = base64.b64decode(view["png_base64"], validate=True)
    except (ValueError, binascii.Error):
        return False
    if not (len(png) >= 45 and png.startswith(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
            and struct.unpack(">II", png[16:24]) == (width, height)):
        return False
    pos = 8
    has_pixels = False
    while pos + 12 <= len(png):
        length = struct.unpack_from(">I", png, pos)[0]
        tag = png[pos + 4:pos + 8]
        end = pos + length + 8
        if end + 4 > len(png) or zlib.crc32(png[pos + 4:end]) != struct.unpack_from(">I", png, end)[0]:
            return False
        if tag == b"IEND":
            return length == 0 and has_pixels and end + 4 == len(png)
        has_pixels |= tag == b"IDAT" and length > 0
        pos = end + 4
    return False


def _run(command, image_id, payload):
    """Bound output even if the worker fails to follow its protocol."""
    name = "weakbench-arena-" + uuid.uuid4().hex
    args = command + ["run", "--rm", "--pull=never", "--name", name, "--interactive", "--init",
                      "--network=none", "--read-only", "--user=10001:10001", "--cap-drop=ALL",
                      "--security-opt=no-new-privileges", "--cpus=1", "--memory=896m", "--memory-swap=896m",
                      "--pids-limit=160", "--ulimit=nofile=1024:1024", "--ipc=private", "--shm-size=128m",
                      "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=192m,mode=1777", "--log-driver=none", image_id]
    proc = None
    try:
        with tempfile.TemporaryFile() as input_file, selectors.DefaultSelector() as events:
            input_file.write(payload)
            input_file.seek(0)
            proc = subprocess.Popen(args, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            events.register(proc.stdout, selectors.EVENT_READ)
            output = bytearray()
            deadline = time.monotonic() + MAX_SECONDS
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not events.select(remaining):
                    raise TimeoutError
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > MAX_WIRE:
                    raise ValueError("oversized worker response")
            if proc.wait(timeout=max(0.1, deadline - time.monotonic())):
                raise OSError("worker exit")
            return json.loads(output)
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()
            try:
                # On kill/disconnect Docker can keep the container running after the CLI exited.
                subprocess.run(command + ["rm", "--force", name], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=5, check=False)
            except (OSError, subprocess.TimeoutExpired):
                pass


def prepare_artifact(case, raw_text):
    """Return a neutral serializable PNG bundle; infrastructure failures are retryable."""
    value = case.get("kind") if isinstance(case, dict) else None
    kind = ALIASES.get(value) if isinstance(value, str) else None
    if kind is None:
        raise ValueError("Visual arena kind must be web or three")
    command, image_id = _runtime()
    key = _key(image_id) if image_id else None
    source, error = _source(raw_text, kind)
    if error:
        return _result(kind, key, "invalid_artifact", error)
    if not command:
        return _result(kind, key, "render_error", "runtime_unavailable")
    payload = json.dumps({"kind": kind, "source": source, "views": VIEWS[kind]}).encode()
    try:
        outcome = _run(command, image_id, payload)
        if not isinstance(outcome, dict):
            raise ValueError
        status, diagnostic = outcome.get("status"), outcome.get("diagnostic")
        if status == "invalid_artifact" and diagnostic in ERRORS:
            return _result(kind, key, status, diagnostic)
        if status == "render_error" and diagnostic in INFRA:
            return _result(kind, key, status, diagnostic)
        views = outcome.get("views")
        if (status == "ok" and isinstance(views, list) and len(views) == len(VIEWS[kind])
                and all(_view_valid(view, spec) for view, spec in zip(views, VIEWS[kind]))):
            return _result(kind, key, "ok", "ok", views)
    except (TimeoutError, subprocess.TimeoutExpired):
        return _result(kind, key, "render_error", "render_timeout")
    except (ValueError, UnicodeError, RecursionError, TypeError, OSError):
        pass
    return _result(kind, key, "render_error", "runtime_failure")


def judge_parts(bundle):
    """Make blinded OpenAI-compatible visual parts; unavailable images cannot enter a visual match."""
    if not isinstance(bundle, dict) or not isinstance(bundle.get("kind"), str):
        raise ValueError("Visual arena bundle required")
    kind = bundle.get("kind")
    views = bundle.get("views")
    if (bundle.get("schema") != SCHEMA or bundle.get("status") != "ok" or kind not in VIEWS
            or not isinstance(views, list) or len(views) != len(VIEWS[kind])
            or not all(_view_valid(v, spec) for v, spec in zip(views, VIEWS[kind]))):
        raise ValueError("Complete rendered views required for visual judging")
    parts = []
    for v in views:
        parts.extend([{"type": "text", "text": f"{v['name']} ({v['width']} × {v['height']})"},
                      {"type": "image_url", "image_url": {"url": "data:image/png;base64," + v["png_base64"]}}])
    return parts
