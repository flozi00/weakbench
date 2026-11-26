"""Install NVIDIA's pinned NVRTC shared libraries into the compiler-only guest."""
import hashlib
import io
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

arch = {"aarch64": "arm64", "x86_64": "amd64"}[platform.machine()]
pin = json.loads(Path("/build/nvrtc-wheels.json").read_bytes())[arch]
with urllib.request.urlopen(pin["url"], timeout=180) as download:
    content = download.read()
if hashlib.sha256(content).hexdigest() != pin["digests"]["sha256"]:
    raise RuntimeError("NVRTC wheel digest mismatch")
dest = Path("/opt/nvrtc")
dest.mkdir()
with zipfile.ZipFile(io.BytesIO(content)) as wheel:
    libs = {Path(name).name: name for name in wheel.namelist()
            if "/lib/" in name and Path(name).name.startswith("libnvrtc") and ".so" in name}
    for lib, source in libs.items():
        (dest / lib).write_bytes(wheel.read(source))
    for name in wheel.namelist():
        if Path(name).name.startswith(("License", "LICENSE")) and len(wheel.read(name)) < 2**20:
            (dest / Path(name).name).write_bytes(wheel.read(name))
if not (dest / "libnvrtc.so.13").exists() or not list(dest.glob("libnvrtc-builtins.so*")):
    raise RuntimeError("Expected NVRTC libraries unavailable")
