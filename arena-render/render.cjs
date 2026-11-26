// Browser programs run only in an isolated page. The controller never evaluates answer text in Node.
const fs = require("node:fs");
const path = require("node:path");
const { chromium } = require("playwright-core");

const origin = "https://artifact.invalid";
const documentURL = origin + "/";
const fixedTime = Date.UTC(2024, 0, 15, 12);
const modules = new Map(["three.module.js", "three.core.js"].map(file =>
  [origin + "/" + file, fs.readFileSync(path.join(__dirname, "node_modules/three/build", file))]));
const failure = (status, diagnostic) => ({status, diagnostic});

async function deterministic(context, page) {
  await context.addInitScript(() => {
    let seed = 8167;
    Math.random = () => ((seed = seed * 16807 % 2147483647) - 1) / 2147483646;
  });
  await page.clock.install({time: fixedTime});
  await page.clock.pauseAt(fixedTime + 1000);
}

async function sceneView(page, source, view) {
  // Initialize the library and graphics device before running candidate code.
  const ready = await page.evaluate(async () => {
    try {
      const THREE = await import("/three.module.js");
      const renderer = new THREE.WebGLRenderer({antialias:true, alpha:false, preserveDrawingBuffer:true});
      renderer.setPixelRatio(1);
      renderer.setSize(innerWidth, innerHeight, false);
      const canvas = renderer.domElement;
      canvas.style.cssText = "width:100vw;height:100vh;display:block";
      document.body.append(canvas);
      globalThis.arenaStudio = {THREE, renderer};
      return true;
    } catch {
      return false;
    }
  });
  if (!ready) return failure("render_error", "webgl_unavailable");
  return page.evaluate(({source, view}) => {
    const {THREE, renderer} = globalThis.arenaStudio;
    // Snapshot the canvas, not a candidate-controlled HTML overlay.
    const canvas = renderer.domElement;
    const capture = HTMLCanvasElement.prototype.toDataURL.bind(canvas);
    const draw = renderer.render.bind(renderer);
    let scene;
    try {
      const factory = new Function("THREE", source + "\n;return typeof buildScene === 'function' ? buildScene : null;")(THREE);
      if (!factory) return {status:"invalid_artifact", diagnostic:"missing_scene_factory"};
      scene = factory(THREE);
    } catch {
      return {status:"invalid_artifact", diagnostic:"javascript_error"};
    }
    if (!(scene instanceof THREE.Scene)) return {status:"invalid_artifact", diagnostic:"invalid_scene"};
    try {
      scene.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(scene, true);
      if (box.isEmpty()) return {status:"invalid_artifact", diagnostic:"empty_scene"};
      const {center, radius} = box.getBoundingSphere(new THREE.Sphere());
      if (![center.x, center.y, center.z, radius].every(Number.isFinite))
        return {status:"invalid_artifact", diagnostic:"invalid_scene"};
      const scale = Math.max(radius, 0.01);
      const direction = {front:[1.4,0.9,2], side:[-2,0.65,1], rear:[0.85,1.5,-2]}[view];
      const camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, scale/100, scale*30);
      camera.position.copy(center).addScaledVector(new THREE.Vector3(...direction).normalize(), 3.45*scale);
      camera.lookAt(center);
      if (!scene.background) scene.background = new THREE.Color(0xecf0f3);
      const hemi = new THREE.HemisphereLight(0xffffff, 0xc8d2da, 1.45);
      const key = new THREE.DirectionalLight(0xfff8e8, 2.0);
      key.position.copy(center).addScaledVector(new THREE.Vector3(2,3,2), scale);
      key.target.position.copy(center);
      scene.add(hemi, key, key.target);
      draw(scene, camera);
      renderer.getContext().finish();
      if (renderer.getContext().isContextLost()) return {status:"render_error", diagnostic:"view_capture_failed"};
      return {status:"ok", png:capture("image/png").split(",")[1]};
    } catch {
      return {status:"invalid_artifact", diagnostic:"javascript_error"};
    }
  }, {source, view});
}

async function renderView(browser, kind, source, name, width, height) {
  const context = await browser.newContext({viewport:{width,height}, deviceScaleFactor:1,
    locale:"en-US", timezoneId:"UTC", colorScheme:"light", reducedMotion:"reduce", acceptDownloads:false,
    serviceWorkers:"block", permissions:[], offline:true,
    isMobile:kind==="web" && name==="mobile", hasTouch:kind==="web" && name==="mobile"});
  try {
    const page = await context.newPage();
    page.setDefaultTimeout(7000);
    let jsError = false, external = false, mainServed = false;
    page.on("pageerror", () => { jsError = true; });
    page.on("dialog", dialog => dialog.dismiss().catch(() => {}));
    context.on("page", child => { if (child !== page) child.close().catch(() => {}); });
    await deterministic(context, page);
    const policy = "default-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none'; object-src 'none'; " +
      "style-src 'unsafe-inline'; font-src data:; img-src data: blob:; media-src data: blob:; " +
      (kind==="three" ? "script-src 'self' 'unsafe-eval'" : "script-src 'unsafe-inline' 'unsafe-eval'");
    await context.route("**/*", async route => {
      const req = route.request();
      if (req.url() === documentURL && req.isNavigationRequest() && !mainServed && req.frame() === page.mainFrame()) {
        mainServed = true;
        const body = kind === "web" ? source :
          '<!doctype html><meta charset="utf-8"><style>*{margin:0}body{overflow:hidden}</style><body></body>';
        return route.fulfill({status:200, contentType:"text/html; charset=utf-8", body,
          headers:{"content-security-policy":policy, "referrer-policy":"no-referrer", "x-content-type-options":"nosniff"}});
      }
      if (kind==="three" && req.resourceType()==="script" && modules.has(req.url())) {
        return route.fulfill({status:200, contentType:"text/javascript", body:modules.get(req.url())});
      }
      external = true;
      return route.abort("blockedbyclient");
    });
    await page.goto(documentURL, {waitUntil:"load", timeout:7000});
    let scenePng;
    if (kind === "three") {
      const result = await sceneView(page, source, name);
      if (result.status !== "ok") return result;
      scenePng = result.png;
    } else {
      await page.clock.runFor(500);
    }
    if (page.url() !== documentURL) return failure("invalid_artifact", "external_resource");
    if (jsError) return failure("invalid_artifact", "javascript_error");
    if (external) return failure("invalid_artifact", "external_resource");
    if (kind==="three") return {status:"ok", view:{name, width, height, png_base64:scenePng}};
    const png = await page.screenshot({type:"png", animations:"disabled", caret:"hide",
      scale:"css", timeout:7000, fullPage:false});
    return {status:"ok", view:{name, width, height, png_base64:png.toString("base64")}};
  } finally {
    await context.close();
  }
}

async function main() {
  const job = JSON.parse(fs.readFileSync(0, "utf8"));
  const browser = await chromium.launch({headless:true, chromiumSandbox:false,
    args:["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
          "--disable-background-networking","--disable-extensions","--disable-sync","--no-first-run"]});
  try {
    const views = [];
    for (const [name,width,height] of job.views) {
      const result = await renderView(browser, job.kind, job.source, name, width, height);
      if (result.status !== "ok") return result;
      views.push(result.view);
    }
    return {status:"ok", diagnostic:"ok", views};
  } finally {
    await browser.close();
  }
}
main().then(value => process.stdout.write(JSON.stringify(value)))
  .catch(() => process.stdout.write(JSON.stringify(failure("render_error","view_capture_failed"))));
