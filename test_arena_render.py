import base64
import os
import struct
import unittest
from unittest.mock import patch
import zlib

import arena_render as render


WEB = """<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>html,body{margin:0}body{background:#243948;color:white;font:28px sans-serif}.box{width:50vw;height:45vh;
background:tomato}@media(max-width:600px){.box{width:90vw;background:gold}}</style></head><body>
<div class="box"></div><script>document.querySelector('.box').textContent='Spring archive';</script></body></html>"""
THREE = """function buildScene(THREE) {
  const scene = new THREE.Scene();
  const ink = new THREE.MeshStandardMaterial({color:0x225991});
  const clay = new THREE.MeshStandardMaterial({color:0xe08b5a});
  scene.add(new THREE.Mesh(new THREE.BoxGeometry(2,1,1), ink));
  const cone = new THREE.Mesh(new THREE.ConeGeometry(.45,1.6,6),clay);
  cone.position.set(.7,.8,.7);
  scene.add(cone);
  return scene;
}"""


def png(width, height):
    def chunk(name, data):
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", zlib.crc32(name + data))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = zlib.compress(b"\0" * ((1 + width * 3) * height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")


def synthetic_views(kind):
    return [{"name": name, "width": width, "height": height, "png_base64": base64.b64encode(png(width, height)).decode()}
            for name, width, height in render.VIEWS[kind]]


class TestRenderer(unittest.TestCase):
    def test_invalid_submission_and_unavailable_runtime_have_different_statuses(self):
        with patch.object(render, "_runtime", return_value=(None, None)):
            with self.assertRaises(RuntimeError):
                render.renderer_key()
            for src in ("", "```python\npass\n```", "This is an essay"):
                self.assertEqual(render.prepare_artifact({"kind": "website"}, src)["status"], "invalid_artifact")
            self.assertEqual(render.prepare_artifact({"kind": "web"}, WEB)["status"], "render_error")
            self.assertEqual(render.prepare_artifact({"kind": "three"}, THREE)["diagnostic"], "runtime_unavailable")

    def test_output_validation_and_blinding(self):
        with patch.object(render, "_runtime", return_value=(["docker"], "test")):
            positive = {"status": "ok", "views": synthetic_views("web")}
            with patch.object(render, "_run", return_value=positive):
                bundle = render.prepare_artifact({"kind": "web", "model_key": "hidden"}, WEB)
            self.assertEqual(bundle["status"], "ok")
            parts = render.judge_parts(bundle)
            self.assertEqual([part["type"] for part in parts], ["text", "image_url"] * 2)
            self.assertNotIn("hidden", str(parts))
            positive["views"][0]["width"] = 42
            with patch.object(render, "_run", return_value=positive):
                bad = render.prepare_artifact({"kind": "web"}, WEB)
            self.assertEqual(bad["status"], "render_error")
            with self.assertRaises(ValueError):
                render.judge_parts(bad)

    def test_status_allowlist_and_deadline(self):
        with patch.object(render, "_runtime", return_value=(["docker"], "test")):
            with patch.object(render, "_run", side_effect=TimeoutError):
                self.assertEqual(render.prepare_artifact({"kind":"threejs"}, THREE)["status"], "render_error")
            for returned, expected in [
                ({"status":"invalid_artifact", "diagnostic":"javascript_error"}, "invalid_artifact"),
                ({"status":"invalid_artifact", "diagnostic":"secret exception details"}, "render_error"),
                ({"status":"render_error", "diagnostic":"webgl_unavailable"}, "render_error"),
            ]:
                with patch.object(render, "_run", return_value=returned):
                    self.assertEqual(render.prepare_artifact({"kind":"three"}, THREE)["status"], expected)

    def test_wrapped_source_and_limits(self):
        self.assertEqual(render._source("```js\n"+THREE+"\n```", "three"), (THREE, None))
        self.assertEqual(render._source("```html\n"+WEB+"\n```", "web"), (WEB, None))
        self.assertEqual(render._source("ä" * render.MAX_INPUT, "three")[1], "answer_too_large")


@unittest.skipUnless(os.environ.get("WEAKBENCH_ARENA_RENDER_E2E") == "1", "local offline renderer image required")
class LiveRenderer(unittest.TestCase):
    def run_twice(self, kind, source):
        a = render.prepare_artifact({"kind":kind}, source)
        self.assertEqual((a["status"], a["diagnostic"]), ("ok","ok"))
        self.assertEqual(a, render.prepare_artifact({"kind":kind}, source))
        self.assertEqual(len(render.judge_parts(a)), 2 * len(render.VIEWS[kind]))
        return a["views"]

    def test_scripted_website(self):
        views = self.run_twice("web", WEB)
        changed = render.prepare_artifact({"kind":"web"}, WEB.replace("Spring archive","Winter archive"))
        self.assertEqual(changed["status"], "ok")
        for before, after in zip(views, changed["views"]):
            self.assertNotEqual(before["png_base64"], after["png_base64"])

    def test_three_geometry_and_camera_positions(self):
        views = self.run_twice("three", THREE)
        self.assertEqual(len({v["png_base64"] for v in views}), 3)
        wrong = render.prepare_artifact({"kind":"three"}, "function buildScene() {")
        self.assertEqual((wrong["status"],wrong["diagnostic"]), ("invalid_artifact","javascript_error"))

if __name__ == "__main__":
    unittest.main()
