#!/usr/bin/env python3
"""20 cold-start cycles of a MANAGED MLX server on :8091 plus a decoy-listener reap test on :8092. Never touches :8080."""
import json, os, re, signal, socket, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
MGR = REPO / "plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/mlx_server_manager.py"
PORT, DECOY = 8091, 8092
N = int(os.environ.get("N", "20"))
MODEL = os.environ.get("MLX_MODEL", "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")


def mgr(*a, timeout=180):
    return subprocess.run([sys.executable, str(MGR), *a], capture_output=True, text=True, timeout=timeout)


def swap_free_mb():
    m = re.search(r"free = ([\d.]+)M", subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout)
    return float(m.group(1)) if m else None


def mem_free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    m = re.search(r"free percentage:\s*(\d+)%", out)
    return int(m.group(1)) if m else None


def disk_free_gb():
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 2**30


def guard_ok():
    """Gate on memory-free % and disk; swap is logged only (macOS grows swap on demand; the pilot showed free swap is not a limit)."""
    f = mem_free_pct()
    return (f is None or f >= 20) and disk_free_gb() >= 20


def avail_gib():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1))
    return (g("Pages free") + g("Pages inactive") + g("Pages speculative") + g("Pages purgeable")) * page / 2**30


def rss_gib(pid):
    r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return int(r) / 2**20 if r else 0.0


def first_token(timeout=60):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 1}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    return bool(d.get("choices"))


def cycle(i):
    row = {"i": i}
    if not guard_ok():
        row["aborted"] = f"guard: mem_free_pct={mem_free_pct()} disk_free_gb={disk_free_gb():.0f}"; return row
    row["avail_before_gib"] = round(avail_gib(), 2)
    t0 = time.perf_counter()
    r = mgr("start", "--port", str(PORT), "--model", MODEL, "--idle-timeout", "300")
    row["boot_s"] = round(time.perf_counter() - t0, 2)
    st = json.loads(mgr("status", "--port", str(PORT)).stdout or "{}")
    row["managed"] = st.get("managed"); pid = st.get("pid")
    try:
        ok = first_token()
        row["ok"] = ok
    except Exception as e:
        row["ok"] = False; row["error"] = str(e)[:120]
    row["first_token_s"] = round(time.perf_counter() - t0, 2)
    row["rss_gib"] = round(rss_gib(pid), 2) if pid else None
    row["avail_loaded_gib"] = round(avail_gib(), 2)
    time.sleep(3)  # idle period before the reap
    rp = mgr("reap", "--port", str(PORT), "--timeout", "2", "--only-managed")
    try:
        row["reap_status"] = json.loads(rp.stdout).get("status")
    except Exception:
        row["reap_status"] = "unparsed"
    if row["reap_status"] != "reaped":
        mgr("stop", "--port", str(PORT))
    time.sleep(2)
    row["avail_after_gib"] = round(avail_gib(), 2)
    row["swap_free_mb"] = swap_free_mb()
    row["stopped"] = not socket.socket().connect_ex(("127.0.0.1", PORT)) == 0
    return row


def decoy_test():
    out = {}
    for flag in ([], ["--only-managed"]):
        stub = ("import http.server,json\n"
                "class H(http.server.BaseHTTPRequestHandler):\n"
                "    def do_GET(self):\n"
                "        b=json.dumps({'data':[]}).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)\n"
                "    def log_message(self,*a): pass\n"
                f"http.server.HTTPServer(('127.0.0.1',{DECOY}),H).serve_forever()\n")
        p = subprocess.Popen([sys.executable, "-c", stub], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        out.setdefault("decoy_status", json.loads(mgr("status", "--port", str(DECOY)).stdout or "{}"))
        time.sleep(3)
        r = mgr("reap", "--port", str(DECOY), "--timeout", "1", *flag)
        time.sleep(1.5)
        alive = p.poll() is None
        out["default" if not flag else "only_managed"] = {"reap_output": (r.stdout or r.stderr).strip()[:200], "decoy_survived": alive}
        if alive:
            p.send_signal(signal.SIGTERM); p.wait(5)
    return out


if __name__ == "__main__":
    assert subprocess.run(["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN"], capture_output=True, text=True).stdout.strip() == "", "port busy"
    res = {"n": N, "swap_free_before_mb": swap_free_mb(), "mem_free_pct_before": mem_free_pct(), "cycles": [cycle(i) for i in range(N)]}
    res["decoy"] = decoy_test()
    c = [x for x in res["cycles"] if "first_token_s" in x]
    ts = sorted(x["first_token_s"] for x in c)
    res["summary"] = {"completed": len(c), "failed": sum(1 for x in c if not x.get("ok")),
                      "first_token_s": {"median": statistics.median(ts), "p95": ts[min(len(ts) - 1, int(round(0.95 * (len(ts) - 1))))], "max": ts[-1]} if ts else None,
                      "boot_s_median": statistics.median(x["boot_s"] for x in c) if c else None,
                      "rss_gib_median": statistics.median(x["rss_gib"] for x in c if x.get("rss_gib")) if c else None,
                      "avail_recovered_gib_median": statistics.median(x["avail_after_gib"] - x["avail_loaded_gib"] for x in c) if c else None,
                      "unmanaged_flag": sum(1 for x in c if x.get("managed") is not True),
                      "aborted": [x for x in res["cycles"] if "aborted" in x]}
    print(json.dumps(res, indent=1))
