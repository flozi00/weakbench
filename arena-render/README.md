# Offline visual renderer

The arena accepts a self-contained HTML document for `web`. Inline HTML, CSS and JavaScript run in Chromium. For `three` the answer is a JavaScript program defining `function buildScene(THREE)`, returning a synchronous `THREE.Scene` with finite renderable geometry. A single matching Markdown code fence is permitted. The runtime supplies Three.js 0.187.0, a studio light rig and a camera fitted to the scene. Each view has its own browser context and starts from the original source.

Build the isolated renderer once on a local Docker daemon. The build needs access to the pinned browser image and the two integrity-locked renderer packages. The Console frontend and Python runner need no additional packages. [Playwright Docker image](https://playwright.dev/docs/docker) · [Three.js documentation](https://threejs.org/docs/)

```sh
docker build -t weakbench-arena-render:1 arena-render
WEAKBENCH_ARENA_RENDER_E2E=1 python3 -BS -m unittest test_arena_render
```

A missing runtime raises `RuntimeError` from `renderer_key()` and gives `prepare_artifact` no cache key. Prior successful renders retain their own immutable keys. The host wrapper accepts a local Unix Docker socket, by default `/var/run/docker.sock`. Use `WEAKBENCH_ARENA_DOCKER_SOCKET=/path/to/local/docker.sock` to choose a different local daemon. Each invocation inspects the local image, selects its immutable ID, and starts a nonroot, read-only, disposable container with no network, no host mounts, no capabilities, no new privileges, private IPC, limited CPU, memory, processes, temporary storage and wall time. The browser and the untrusted page share the container boundary. Chromium's inner namespace sandbox is disabled for compatibility with unprivileged Docker. Inside the container the browser controller uses Playwright's transport pipe, serves the document and two local Three.js modules from a synthetic origin, and denies outbound browser requests. It saves screens in memory. The only returned payloads are bounded PNGs and fixed diagnostic codes.

Web views are 1120 × 760 and 390 × 844. The clock, initial pseudo-random seed, locale, time zone, display scale and reduced-motion preference are fixed. The static web frame is captured after 500 ms of virtual page time. A document can explicitly use nondeterministic sources, such as cryptographic randomness. Three.js views are 720 × 600 from three published directions with common framing. A renderer key combines the local immutable image ID (including architecture, browser, renderer code and npm lock) and the hash of the host-side contract.

`prepare_artifact(case, raw_text)` returns PNG views or a non-success status. `invalid_artifact` describes an identifiable candidate-format, JavaScript, scene or resource error. `render_error` covers unavailable infrastructure, timeouts, unsupported WebGL and failed captures; these results are retryable. `judge_parts(bundle)` requires a complete successful bundle and converts its images into blinded multimodal message parts. A visual matchup with a renderer failure cannot produce a win, loss or text-only fallback.
