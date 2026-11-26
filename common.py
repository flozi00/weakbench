"""Shared plumbing for the benchmark suites (stdlib only).

Every suite talks to one OpenAI-compatible server (vLLM / SGLang) at --url and
writes one run directory:

    runs/<UTC stamp>-<kind>-<arm>/
        manifest.json     server build, model, max_model_len, args, git sha
        <name>.jsonl      per-item (quality) / per-request (speed) records
        summary.json      what the suite measured, ready for compare.py

and appends its summary rows to results/<kind>.jsonl (the ledger
that is committed). compare.py pairs two run directories item by item.
"""
import ast
import functools
import collections
import collections.abc
import datetime
import fcntl
import hashlib
import http.client
import json
import os
import re
import socket
import ssl
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Overridable for read-only installs (the suite mounted read-only into a container).
RUNS = Path(os.environ.get("BENCH_RUNS", HERE / "runs"))
RESULTS = Path(os.environ.get("BENCH_RESULTS", HERE / "results"))
CACHE = Path(os.environ.get("BENCH_CACHE", Path.home() / ".cache" / "weakbench"))

# macOS negative-caches a hostname queried before its DNS record existed. For hosts ending in
# WEAKBENCH_DNS_FALLBACK_SUFFIX (unset = off) a failed lookup is retried through `host`.
_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_fallback(host, *a, **kw):
    try:
        return _getaddrinfo(host, *a, **kw)
    except socket.gaierror:
        suffix = os.environ.get("WEAKBENCH_DNS_FALLBACK_SUFFIX")
        if not (suffix and isinstance(host, str) and host.endswith(suffix)):
            raise
        try:  # no `host` binary (minimal containers): keep the original error
            out = subprocess.run(["host", host], capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            raise socket.gaierror(socket.EAI_NONAME, f"cannot resolve {host}") from None
        ip = next((ln.split()[-1] for ln in out.splitlines() if "has address" in ln), None)
        if ip is None:
            raise
        return _getaddrinfo(ip, *a, **kw)


socket.getaddrinfo = _getaddrinfo_fallback
CTX = ssl._create_unverified_context()
RETRY_STATUS = {429, 500, 502, 503, 504}


def verify_tls():
    """Require the system trust store for model and judge calls carrying private data."""
    global CTX
    CTX = ssl.create_default_context()
    # An embedded caller may have already contacted this host in another profile.
    with _POOL_LOCK:
        for key in list(_POOL):
            if key[0] == "https":
                for conn in _POOL.pop(key):
                    conn.close()


class HTTPError(RuntimeError):
    def __init__(self, status, body):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status, self.body = status, body


# Keep-alive connection pool: a measured TTFT must not include a fresh TCP +
# TLS handshake per request (urllib sends `Connection: close`), and OpenAI
# SDK clients reuse connections too. Idle connections are shared across
# threads, so a new cell's workers pick up the previous cell's sockets.
_POOL, _POOL_LOCK = {}, threading.Lock()
_STALE = (http.client.RemoteDisconnected, http.client.BadStatusLine, ConnectionResetError, BrokenPipeError)


def _send(url, body, timeout):
    """(pool key, connection, response) for one request; raises HTTPError on
    >= 400 (body read, connection back in the pool). A pooled socket the
    server closed while idle drops every idle socket of that server and is
    replaced by a fresh one."""
    u = urllib.parse.urlsplit(url)
    key, path = (u.scheme, u.hostname, u.port), u.path + (f"?{u.query}" if u.query else "")
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"content-type": "application/json"}
    if os.environ.get("BENCH_API_KEY"):
        hdrs["authorization"] = "Bearer " + os.environ["BENCH_API_KEY"]
    for attempt in (0, 1):
        with _POOL_LOCK:
            idle = _POOL.setdefault(key, [])
            conn = idle.pop() if idle else None
        if conn is None:
            conn = (http.client.HTTPSConnection(u.hostname, u.port, timeout=timeout, context=CTX)
                    if u.scheme == "https" else http.client.HTTPConnection(u.hostname, u.port, timeout=timeout))
        reused, conn.timeout = conn.sock is not None, timeout
        if reused:
            conn.sock.settimeout(timeout)
        try:
            conn.request("POST" if data is not None else "GET", path or "/", body=data, headers=hdrs)
            resp = conn.getresponse()
        except _STALE:
            conn.close()
            if reused and attempt == 0:
                # Its idle peers went stale with it (vLLM closes keep-alive
                # sockets after 5 s idle): retrying on ANOTHER pooled one failed
                # ~14 requests of every c8 cell after a c32 cell (2026-10-16).
                with _POOL_LOCK:
                    stale = _POOL.pop(key, [])
                for other in stale:
                    other.close()
                continue
            raise
        except BaseException:
            conn.close()
            raise
        if resp.status >= 400:
            raw = resp.read().decode(errors="replace")
            _release(key, conn, resp)
            raise HTTPError(resp.status, raw)
        return key, conn, resp


def _release(key, conn, resp):
    if resp.will_close:
        conn.close()
    else:
        with _POOL_LOCK:
            _POOL.setdefault(key, []).append(conn)


# -------------------------------------------------------------------------- throttle
# A scheduled job shares one set of slots across candidate and judge. A/B speed
# runs leave this disabled so their requested concurrency remains measurable.
_THROTTLE = {"urls": set(), "yield_above": 0, "poll_s": 2.0}
_SLOTS = None


def throttle_configure(url, max_parallel=0, yield_above=0, poll_s=2.0, judge_url=None):
    global _SLOTS
    max_parallel = int(max_parallel or 0)
    _THROTTLE.update(urls={u.rstrip("/") for u in (url, judge_url) if u},
                     yield_above=int(yield_above or 0), poll_s=max(0.5, float(poll_s)))
    _SLOTS = threading.BoundedSemaphore(max_parallel) if max_parallel else None


def _throttle_enter(url, body):
    """Hold a slot from admission until the response is consumed, including SSE streams."""
    slots = _SLOTS
    ceiling = _THROTTLE["yield_above"]
    if body is None or (slots is None and not ceiling) or url.rstrip("/") not in _THROTTLE["urls"]:
        return lambda: None
    if slots is not None:
        slots.acquire()
    try:
        while ceiling:
            load = metrics(url)
            running = load.get("num_requests_running")
            # A server without these gauges still has the hard client-side cap.
            if running is None or running + load.get("num_requests_waiting", 0) < ceiling:
                break
            time.sleep(_THROTTLE["poll_s"])
    except BaseException:
        if slots is not None:
            slots.release()
        raise
    return slots.release if slots is not None else (lambda: None)


def request(url, path, body=None, timeout=900, retries=3, *, reject_redirects=False):
    """One JSON call. Retries gateway/overload errors and dropped connections
    with backoff (a 30-minute suite must survive one 502); a 4xx is the
    request's fault and raises at once."""
    for attempt in range(retries + 1):
        release = _throttle_enter(url, body)
        try:
            key, conn, resp = _send(url.rstrip("/") + path, body, timeout)
            try:
                raw = resp.read().decode()
            except BaseException:
                conn.close()
                raise
            _release(key, conn, resp)
            if reject_redirects and 300 <= resp.status < 400:
                raise HTTPError(resp.status, "unexpected redirect from endpoint")
            return json.loads(raw) if raw.lstrip()[:1] in ("{", "[") else raw
        except HTTPError as exc:
            if exc.status not in RETRY_STATUS or attempt == retries:
                raise
        except (OSError, http.client.HTTPException):  # refused, reset, timeout, truncated reply
            if attempt == retries:
                raise
        finally:
            release()
        time.sleep(2 ** attempt)


def stream(url, path, body, timeout=900):
    """Yield (monotonic time, parsed chunk) for each SSE data line. The
    connection returns to the pool only after a fully read response."""
    release = _throttle_enter(url, body)
    try:
        key, conn, resp = _send(url.rstrip("/") + path, {**body, "stream": True}, timeout)
        complete = False
        try:
            while raw := resp.readline():
                line = raw.decode(errors="replace").strip()
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    yield time.monotonic(), json.loads(line[5:])
            complete = True
        finally:  # an abandoned or failed stream leaves the socket mid-response: close it
            if complete:
                _release(key, conn, resp)
            else:
                conn.close()
    finally:
        release()


def metrics(url):
    """Selected Prometheus counters/gauges summed over label sets, {} when the
    route does not expose /metrics. Behind a multi-replica pool a scrape lands
    on one replica — read the deltas as that replica's sample."""
    try:
        text = request(url, "/metrics", timeout=15, retries=1)
    except Exception:  # noqa: BLE001 — metrics are a diagnostic, never fatal
        return {}
    keep = ("spec_decode_num_drafts", "spec_decode_num_draft_tokens", "spec_decode_num_accepted_tokens_total",
            "prefix_cache_queries", "prefix_cache_hits", "num_preemptions", "generation_tokens",
            "prompt_tokens_total", "num_requests_running", "num_requests_waiting")
    out = {}
    for line in str(text).splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{")[0].split(":")[-1]
        if any(base.startswith(k) for k in keep) and "_per_pos" not in base:
            try:
                out[base] = out.get(base, 0.0) + float(value)
            except ValueError:
                pass
    return out


def server_info(url, model=None):
    """What is serving: model id + max_model_len from /v1/models, build from
    /version. `model` overrides the id (multi-model servers)."""
    listing = request(url, "/v1/models", timeout=30)
    data = (listing or {}).get("data") or [{}]
    entry = next((m for m in data if m.get("id") == model), data[0])
    try:
        version = request(url, "/version", timeout=10, retries=0)
    except Exception:  # noqa: BLE001 — SGLang has /get_server_info instead
        version = None
    return {"model": model or entry.get("id"), "max_model_len": entry.get("max_model_len"),
            "version": version, "root": entry.get("root")}


def _git_sha():
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except OSError:
        return None


def _code_sha():
    """Hash of the suite code that shapes a run (every non-test .py incl. vendored
    scorers): two runs with equal hashes ran the same prompts, defaults and
    scorers, committed or not."""
    h = hashlib.sha256()
    # Installed environments, run artifacts and private caches are outside the suite.
    sources = [*HERE.glob("*.py"), *(HERE / "vendor").rglob("*.py")]
    for f in sorted(p for p in sources if not p.name.startswith("test_")):
        h.update(str(f.relative_to(HERE)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def new_run(kind, arm, url, model, args):
    """Create the run directory and its manifest; returns (path, manifest)."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RUNS / f"{stamp}-{kind}-{arm}"
    path.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": path.name, "kind": kind, "arm": arm, "url": url,
                "started_at": stamp, "server": server_info(url, model),
                "bench_git_sha": _git_sha(), "bench_code_sha": _code_sha(), "args": args}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return path, manifest


def write_atomic(path, data):
    """Write via a per-process temp file + rename: a Ctrl-C or a second run
    filling the same cache entry never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data if isinstance(data, bytes) else data.encode())
    os.replace(tmp, path)


HF_RETRIES = int(os.environ.get("BENCH_HF_RETRIES", "4"))


def _retrying(fn):
    """fn() with backoff on Hugging Face's transient failures (the dataset-viewer
    API answers 503 for minutes at a time): waits 5, 15, 45, 120 s; a 4xx other
    than 429 is the request's fault and raises at once."""
    for attempt in range(HF_RETRIES + 1):
        try:
            return fn()
        except urllib.error.HTTPError as exc:
            if exc.code not in RETRY_STATUS or attempt == HF_RETRIES:
                raise
        except (urllib.error.URLError, ConnectionError, TimeoutError, socket.timeout):
            if attempt == HF_RETRIES:
                raise
        time.sleep(min(120, 5 * 3 ** attempt))


def hf_rows(dataset, config, split, offset, length):
    """Rows of a Hugging Face dataset over the public dataset-viewer API
    (stdlib, no parquet), cached on disk. Max 100 rows per page."""
    out = []
    for start in range(offset, offset + length, 100):
        n = min(100, offset + length - start)
        cached = CACHE / "hf-rows" / f"{dataset.replace('/', '__')}-{config}-{split}-{start}-{n}.json"
        if not cached.exists():
            url = ("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
                {"dataset": dataset, "config": config, "split": split, "offset": start, "length": n}))
            def fetch():
                with urllib.request.urlopen(url, timeout=60) as resp:
                    return [r["row"] for r in json.loads(resp.read())["rows"]]
            write_atomic(cached, json.dumps(_retrying(fetch), ensure_ascii=False))
        out += json.loads(cached.read_text())
    return out


def _hf_token():
    tok = os.environ.get("HF_TOKEN")
    path = Path.home() / ".cache" / "huggingface" / "token"
    return tok or (path.read_text().strip() if path.exists() else None)


def _hf_get(url, timeout=120):
    req = urllib.request.Request(url, headers={"authorization": f"Bearer {_hf_token()}"} if _hf_token() else {})
    return urllib.request.urlopen(req, timeout=timeout)


def hf_list(repo, revision, repo_type="datasets"):
    """Every file path of a Hugging Face repo at a pinned revision."""
    cached = CACHE / "hf" / repo.replace("/", "__") / revision / ".files.json"
    if not cached.exists():
        def fetch():
            with _hf_get(f"https://huggingface.co/api/{repo_type}/{repo}/revision/{revision}") as resp:
                return [s["rfilename"] for s in json.loads(resp.read())["siblings"]]
        write_atomic(cached, json.dumps(_retrying(fetch)))
    return json.loads(cached.read_text())


# our own hand-written / translated sets, private, pinned
DATA_REPO, DATA_REV = "flozi00/weakbench-data", "ec7c8efbb4d63416ab2ae28163e1ce7555b04d14"


def distinct(load, pool=600):
    """Wrap a generated-task loader so it yields only items with a prompt not seen before (in item order, so the quick
    tier stays a prefix of the full tier). It draws 3n candidates first and falls back to `pool` when that is not
    enough; with n=None it returns every distinct item among `pool` draws."""
    def take(lang, draws, n, seed):
        out, seen = [], set()
        for x in load(lang, draws, seed):
            key = json.dumps(x.get("turns") or x["messages"], sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                out.append(x)
                if n and len(out) == n:
                    break
        return out

    def wrapped(lang, n, seed):
        out = take(lang, min(pool, 3 * n), n, seed) if n else []
        return out if n and len(out) == n else take(lang, pool, n, seed)
    return wrapped


def redraws(n, items, pair=False):
    """Loop of a generated loader that never repeats a prompt: yields (i, attempt) and the body seeds its rng with
    the attempt (attempt 0 = the old seed, so items without a repeat stay as they were) and appends at most one item.
    An item whose prompt (turns or messages) an earlier item already had is dropped and drawn again with the next
    attempt; a body that appends nothing skips that id. pair=True: twins 2k, 2k+1 share one draw (control and trap),
    so the odd twin reuses the even twin's attempt."""
    seen, last = set(), 0
    for i in range(n):
        for att in range(1000):
            if pair and i % 2:
                yield i, last
                break
            before = len(items)
            yield i, att
            if len(items) == before:
                break
            key = json.dumps(items[-1].get("turns") or items[-1]["messages"], sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                last = att
                break
            items.pop()
        else:
            raise RuntimeError(f"no new prompt for item {i} in 1000 draws")


class OwnSet(collections.abc.Sequence):
    """A hand-written set from the private DATA_REPO (`sets/<module>.<NAME>.py`, a Python literal), fetched on
    first use so a run without access fails loudly only in the tasks that need it."""

    def __init__(self, name):
        self.name = name

    @functools.cached_property
    def value(self):
        return ast.literal_eval(hf_file(DATA_REPO, DATA_REV, f"sets/{self.name}.py").read_text(encoding="utf-8"))

    def __getitem__(self, k):
        return self.value[k]

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getattr__(self, attr):  # dict methods (.items, .keys) of a dict-valued set
        if attr in ("name", "value") or attr.startswith("__"):
            raise AttributeError(attr)
        return getattr(self.value, attr)


def hf_file(repo, revision, filename, repo_type="datasets"):
    """One file of a Hugging Face repo at a pinned revision, downloaded once
    into the cache (gated repos use the local `hf auth login` token)."""
    dest = CACHE / "hf" / repo.replace("/", "__") / revision / filename
    if not dest.exists():
        prefix = "datasets/" if repo_type == "datasets" else ""
        url = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{urllib.parse.quote(filename)}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")
        def fetch():  # a retry rewrites the partial file from the start
            with _hf_get(url, timeout=600) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        _retrying(fetch)
        os.replace(tmp, dest)
    return dest


def gh_file(repo, commit, path):
    """One file of a GitHub repo at a pinned commit, downloaded once into the cache.

    Same shape as `hf_file` (atomic replace, retried), so a cache mirror can treat both alike. Used for
    the benchmarks whose prompts live in a GitHub repo rather than on Hugging Face (EQ-Bench).
    """
    dest = CACHE / "gh" / repo.replace("/", "__") / commit / path
    if not dest.exists():
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(path)}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.part")

        def fetch():  # a retry rewrites the partial file from the start
            # plain urlopen: raw.githubusercontent.com 404s when an HF token rides along in Authorization
            with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
        _retrying(fetch)
        os.replace(tmp, dest)
    return dest


def read_rows(path):
    """Rows of a downloaded data file: .parquet (needs pyarrow — run the
    quality suite with `uv run`), .jsonl, .json (list) or .csv."""
    path = str(path)
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    if path.endswith(".csv"):
        import csv
        with open(path, newline="") as fh:
            return list(csv.DictReader(fh))
    with open(path) as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else [data]


def wiki_texts(lang, n, min_chars=6000, offset=20000):
    """n long Wikipedia articles (20231101 snapshot) in `lang` — the fixed,
    natural-prose corpus behind long prompts (speed) and drift texts (quality).
    Same (lang, n, min_chars, offset) -> same texts, run after run."""
    texts, start = [], offset
    while len(texts) < n and start < offset + 20000:
        texts += [r["text"] for r in hf_rows("wikimedia/wikipedia", f"20231101.{lang}", "train", start, 100)
                  if len(r["text"]) >= min_chars]
        start += 100
    return texts[:n]


def chrf(hyp, ref, order=6, beta=2):
    """Segment chrF in [0, 1] with sacrebleu semantics (char 1..6-grams, whitespace
    removed, precision/recall averaged over the orders both sides have, beta 2).
    Lives here, not in quality.py, because tasks_text.py scores with it too."""
    hyp, ref = "".join(hyp.split()), "".join(ref.split())
    p = r = eff = 0
    for n in range(1, order + 1):
        h = collections.Counter(hyp[i:i + n] for i in range(len(hyp) - n + 1))
        g = collections.Counter(ref[i:i + n] for i in range(len(ref) - n + 1))
        if h and g:
            m = sum((h & g).values())
            p, r, eff = p + m / h.total(), r + m / g.total(), eff + 1
    if not eff or not p + r:
        return 0.0
    p, r = p / eff, r / eff
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def append_jsonl(path, row):
    with open(path, "a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    """Rows of a JSONL file. A truncated LAST line (a writer killed mid-append, e.g. a time budget stopping
    quality.py) is dropped; a bad line anywhere else is corruption and raises."""
    with open(path) as fh:
        lines = [line for line in fh if line.strip()]
    rows = []
    for k, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if k != len(lines) - 1:
                raise
    return rows


def finish_run(path, manifest, summary_rows):
    """summary.json in the run dir + one ledger row per summary row."""
    (path / "summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False))
    RESULTS.mkdir(exist_ok=True)
    ledger = RESULTS / f"{manifest['kind']}.jsonl"
    version = manifest["server"]["version"]  # vLLM's /version is {"version": "..."}
    version = version.get("version") if isinstance(version, dict) else version
    new = [{"run_id": manifest["run_id"], "arm": manifest["arm"], "model": manifest["server"]["model"],
            "version": version, **row} for row in summary_rows]
    # Several suites may finish at once (A and B arms, quality + speed): one
    # writer at a time; a resumed run's earlier rows are replaced, not doubled.
    with open(RESULTS / ".ledger.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        old = read_jsonl(ledger) if ledger.exists() else []
        if any(r.get("run_id") == manifest["run_id"] for r in old):
            rows = [r for r in old if r.get("run_id") != manifest["run_id"]] + new
            write_atomic(ledger, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        else:
            for row in new:
                append_jsonl(ledger, row)


# ---------------------------------------------------------------- language id (quality + tasks_lang)
DE_WORDS = frozenset("der die das und ist nicht ein eine einen einem dem den des zu mit von sich auf für im auch es "
                     "ich wir sie werden wird sind oder aber wenn dass als wie nach bei aus noch nur um durch über "
                     "vom zum zur hat haben kann können diese dieser dieses wurde wurden sein seine ihre".split())
EN_WORDS = frozenset("the and is are not of to with from that this these those it for on as be by were has have had "
                     "can which or but if than then there their they we you he she would should could into about "
                     "been at".split())  # words shared with German (in, an, was, so, will) are left out


def lang_ok(text, lang, min_words=20):
    """German vs English by function-word counts; None when undecidable (other
    language, fewer than min_words words, no function words, a tie)."""
    words = re.findall(r"[^\W\d_]+", text.lower())
    if lang not in ("de", "en") or len(words) < min_words:
        return None
    de, en = sum(w in DE_WORDS for w in words), sum(w in EN_WORDS for w in words)
    return None if de == en else (de > en) == (lang == "de")
