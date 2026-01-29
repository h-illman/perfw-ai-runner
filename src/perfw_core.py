from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import platform
import socket
import statistics
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import psutil
import requests

try:
    import pynvml  # type: ignore
    _HAS_NVML = True
except requests.RequestException:
    _HAS_NVML = False


# ----------------------------
# Helpers
# ----------------------------

def utc_now_iso() -> str:
    # timezone-aware UTC
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ns_now() -> int:
    return time.time_ns()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.pstdev(vals))


def pct_change(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def _lp_escape_tag(v: str) -> str:
    # escape commas, spaces, equals per line protocol tag rules
    return v.replace("\\", "\\\\").replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


# ----------------------------
# Config
# ----------------------------

@dataclasses.dataclass
class InfluxCfg:
    enabled: bool = True
    url: str = "http://localhost:8181"
    database: str = "perfw"
    precision: str = "nanosecond"   # auto|second|millisecond|microsecond|nanosecond
    accept_partial: bool = True
    no_sync: bool = True


@dataclasses.dataclass
class GrafanaCfg:
    enabled: bool = False
    dashboard_url: str = "http://localhost:3000"
    open_on_start: bool = True


@dataclasses.dataclass
class RegressionCfg:
    enabled: bool = True
    warn_drop_pct: float = 5.0
    fail_drop_pct: float = 10.0
    baseline_policy: str = "latest_pass"  # future: "specific_run_id"


@dataclasses.dataclass
class RunCfg:
    output_dir: str = "runs"
    workload: str = "resnet50"      # resnet50 | matmul
    model: str = "resnet50"         # displayed in dashboards
    duration_s: float = 60.0
    warmup_s: float = 10.0
    repeats: int = 3
    sample_hz: float = 2.0

    batch_size: int = 32
    precision: str = "fp16"         # fp32 | fp16
    image_size: int = 224

    matmul_n: int = 2048
    allow_tf32: bool = True

    notes: str = ""

    influx: InfluxCfg = dataclasses.field(default_factory=InfluxCfg)
    grafana: GrafanaCfg = dataclasses.field(default_factory=GrafanaCfg)
    regression: RegressionCfg = dataclasses.field(default_factory=RegressionCfg)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RunCfg":
        influx = InfluxCfg(**d.get("influx", {}))
        grafana = GrafanaCfg(**d.get("grafana", {}))
        regression = RegressionCfg(**d.get("regression", {}))

        base = dict(d)
        base.pop("influx", None)
        base.pop("grafana", None)
        base.pop("regression", None)

        cfg = RunCfg(**base)
        cfg.influx = influx
        cfg.grafana = grafana
        cfg.regression = regression
        return cfg


# ----------------------------
# InfluxDB 3 Writer (Line Protocol)
# ----------------------------

class InfluxWriter:
    """
    Minimal InfluxDB 3 line protocol writer (HTTP).

    Uses:
      POST /api/v3/write_lp?db=<db>&precision=<...>&accept_partial=<...>&no_sync=<...>
      Authorization: Bearer <token>

    Token is read from env var:
      INFLUXDB3_AUTH_TOKEN
    """
    def __init__(self, cfg: InfluxCfg):
        self.cfg = cfg
        self.token = os.getenv("INFLUXDB3_AUTH_TOKEN", "").strip()
        self._enabled = bool(cfg.enabled and cfg.url and cfg.database and self.token)

        self._q: List[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session = requests.Session()

        # batching
        self.flush_every_lines = 200
        self.flush_every_s = 1.0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._enabled:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self.flush()

    def enqueue(self, line: str) -> None:
        if not self._enabled or not line:
            return
        with self._lock:
            self._q.append(line)

    def flush(self) -> None:
        if not self._enabled:
            return
        with self._lock:
            if not self._q:
                return
            batch = "\n".join(self._q)
            self._q.clear()

        params = {
            "db": self.cfg.database,
            "precision": self.cfg.precision,
            "accept_partial": "true" if self.cfg.accept_partial else "false",
            "no_sync": "true" if self.cfg.no_sync else "false",
        }
        url = self.cfg.url.rstrip("/") + "/api/v3/write_lp"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "text/plain; charset=utf-8",
        }

        try:
            r = self._session.post(url, params=params, headers=headers, data=batch.encode("utf-8"), timeout=5)
            if r.status_code not in (204, 200):
                # Best effort: don't crash the benchmark.
                pass
        except requests.RequestException:
            pass

    def _loop(self) -> None:
        last = time.time()
        while not self._stop.is_set():
            time.sleep(0.05)
            now = time.time()
            with self._lock:
                qlen = len(self._q)
            if qlen >= self.flush_every_lines or (now - last) >= self.flush_every_s:
                self.flush()
                last = now


# ----------------------------
# Telemetry Collector
# ----------------------------

@dataclasses.dataclass
class TelemetrySample:
    t_ns: int
    gpu_power_w: Optional[float]
    gpu_temp_c: Optional[float]
    gpu_util_pct: Optional[float]
    gpu_clock_mhz: Optional[float]
    mem_clock_mhz: Optional[float]
    vram_used_mb: Optional[float]
    vram_total_mb: Optional[float]
    cpu_util_pct: Optional[float]
    ram_used_mb: Optional[float]
    ram_total_mb: Optional[float]


class TelemetryCollector:
    def __init__(self, sample_hz: float, csv_path: Path, influx: Optional[InfluxWriter], tags: Dict[str, str]):
        self.sample_hz = max(0.5, float(sample_hz))
        self.csv_path = csv_path
        self.influx = influx
        self.tags = tags

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[TelemetrySample] = []

        self._nvml_handle = None
        self._nvml_init_ok = False

    def start(self) -> None:
        if _HAS_NVML:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_init_ok = True
            except requests.RequestException:
                self._nvml_init_ok = False

        safe_mkdir(self.csv_path.parent)
        if not self.csv_path.exists():
            self.csv_path.write_text(
                "t_ns,gpu_power_w,gpu_temp_c,gpu_util_pct,gpu_clock_mhz,mem_clock_mhz,"
                "vram_used_mb,vram_total_mb,cpu_util_pct,ram_used_mb,ram_total_mb\n",
                encoding="utf-8"
            )

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

        if self._nvml_init_ok:
            try:
                pynvml.nvmlShutdown()
            except requests.RequestException:
                pass

    def _loop(self) -> None:
        period = 1.0 / self.sample_hz
        while not self._stop.is_set():
            t0 = time.time()
            s = self._sample()
            self.samples.append(s)
            self._append_csv(s)
            self._write_influx(s)

            dt_sleep = period - (time.time() - t0)
            if dt_sleep > 0:
                time.sleep(dt_sleep)

    def _sample(self) -> TelemetrySample:
        t_ns = ns_now()

        # CPU/RAM
        cpu_util = None
        ram_used = None
        ram_total = None
        try:
            cpu_util = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            ram_used = float(vm.used) / (1024 * 1024)
            ram_total = float(vm.total) / (1024 * 1024)
        except requests.RequestException:
            pass

        # GPU via NVML
        gpu_power = gpu_temp = gpu_util = gpu_clock = mem_clock = vram_used = vram_total = None
        if self._nvml_init_ok and self._nvml_handle is not None:
            try:
                gpu_power = float(pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)) / 1000.0  # mW -> W
            except requests.RequestException:
                gpu_power = None
            try:
                gpu_temp = float(pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
            except requests.RequestException:
                gpu_temp = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                gpu_util = float(util.gpu)
            except requests.RequestException:
                gpu_util = None
            try:
                gpu_clock = float(pynvml.nvmlDeviceGetClockInfo(self._nvml_handle, pynvml.NVML_CLOCK_GRAPHICS))
            except requests.RequestException:
                gpu_clock = None
            try:
                mem_clock = float(pynvml.nvmlDeviceGetClockInfo(self._nvml_handle, pynvml.NVML_CLOCK_MEM))
            except requests.RequestException:
                mem_clock = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                vram_used = float(mem.used) / (1024 * 1024)
                vram_total = float(mem.total) / (1024 * 1024)
            except requests.RequestException:
                vram_used = vram_total = None

        return TelemetrySample(
            t_ns=t_ns,
            gpu_power_w=gpu_power,
            gpu_temp_c=gpu_temp,
            gpu_util_pct=gpu_util,
            gpu_clock_mhz=gpu_clock,
            mem_clock_mhz=mem_clock,
            vram_used_mb=vram_used,
            vram_total_mb=vram_total,
            cpu_util_pct=cpu_util,
            ram_used_mb=ram_used,
            ram_total_mb=ram_total,
        )

    def _append_csv(self, s: TelemetrySample) -> None:
        def f(x):
            return "" if x is None else f"{x:.4f}"
        line = (
            f"{s.t_ns},"
            f"{f(s.gpu_power_w)},{f(s.gpu_temp_c)},{f(s.gpu_util_pct)},{f(s.gpu_clock_mhz)},{f(s.mem_clock_mhz)},"
            f"{f(s.vram_used_mb)},{f(s.vram_total_mb)},"
            f"{f(s.cpu_util_pct)},{f(s.ram_used_mb)},{f(s.ram_total_mb)}\n"
        )
        try:
            with self.csv_path.open("a", encoding="utf-8") as fp:
                fp.write(line)
        except requests.RequestException:
            pass

    def _write_influx(self, s: TelemetrySample) -> None:
        if not self.influx or not self.influx.enabled:
            return

        tag_str = ",".join([f"{k}={_lp_escape_tag(v)}" for k, v in self.tags.items()])
        fields = {
            "gpu_power_w": s.gpu_power_w,
            "gpu_temp_c": s.gpu_temp_c,
            "gpu_util_pct": s.gpu_util_pct,
            "gpu_clock_mhz": s.gpu_clock_mhz,
            "mem_clock_mhz": s.mem_clock_mhz,
            "vram_used_mb": s.vram_used_mb,
            "vram_total_mb": s.vram_total_mb,
            "cpu_util_pct": s.cpu_util_pct,
            "ram_used_mb": s.ram_used_mb,
            "ram_total_mb": s.ram_total_mb,
        }
        field_parts = []
        for k, v in fields.items():
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                continue
            field_parts.append(f"{k}={float(v):.4f}")
        if not field_parts:
            return

        lp = f"ai_telemetry,{tag_str} " + ",".join(field_parts) + f" {s.t_ns}"
        self.influx.enqueue(lp)


# ----------------------------
# Workloads
# ----------------------------

class BaseWorkload:
    name: str = "base"

    def setup(self, cfg: RunCfg) -> None:
        raise NotImplementedError

    def warmup(self, stop_event: threading.Event) -> None:
        raise NotImplementedError

    def run_timed(self, duration_s: float, stop_event: threading.Event) -> Dict[str, Any]:
        raise NotImplementedError


class MatmulWorkload(BaseWorkload):
    name = "matmul"

    def setup(self, cfg: RunCfg) -> None:
        import torch  # type: ignore
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if (cfg.precision == "fp16" and self.device == "cuda") else torch.float32
        self.n = int(cfg.matmul_n)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        self.a = torch.randn(self.n, self.n, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.n, self.n, device=self.device, dtype=self.dtype)

    def warmup(self, stop_event: threading.Event) -> None:
        for _ in range(10):
            if stop_event.is_set():
                return
            _ = self.a @ self.b
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def run_timed(self, duration_s: float, stop_event: threading.Event) -> Dict[str, Any]:
        start = time.perf_counter()
        iters = 0
        while True:
            if stop_event.is_set():
                break
            _ = self.a @ self.b
            iters += 1
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            if (time.perf_counter() - start) >= duration_s:
                break

        elapsed = time.perf_counter() - start
        throughput = iters / elapsed if elapsed > 0 else 0.0
        return {
            "workload": self.name,
            "units": "matmuls",
            "iterations": iters,
            "duration_s": elapsed,
            "throughput_items_s": throughput,
            "notes": f"device={self.device},n={self.n},dtype={str(self.dtype).replace('torch.', '')}",
        }

    def __init__(self) -> None:
        self.torch = None
        self.device: str = "cpu"
        self.dtype = None
        self.n: int = 0
        self.a = None
        self.b = None


class ResNet50Workload(BaseWorkload):
    name = "resnet50"

    def setup(self, cfg: RunCfg) -> None:
        import torch  # type: ignore
        import torchvision  # type: ignore

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = int(cfg.batch_size)
        self.precision = str(cfg.precision).lower()
        self.image_size = int(cfg.image_size)

        weights = getattr(torchvision.models, "ResNet50_Weights", None)
        if weights is not None:
            model = torchvision.models.resnet50(weights=weights.DEFAULT)
        else:
            model = torchvision.models.resnet50(pretrained=True)

        model.eval().to(self.device)
        if self.device == "cuda" and self.precision == "fp16":
            model = model.half()

        self.model = model

        x = torch.randn(self.batch_size, 3, self.image_size, self.image_size, device=self.device)
        if self.device == "cuda" and self.precision == "fp16":
            x = x.half()
        self.x = x
        torch.set_grad_enabled(False)

    def warmup(self, stop_event: threading.Event) -> None:
        for _ in range(10):
            if stop_event.is_set():
                return
            _ = self.model(self.x)
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def run_timed(self, duration_s: float, stop_event: threading.Event) -> Dict[str, Any]:
        lat_ms: List[float] = []
        iters = 0
        start = time.perf_counter()

        while True:
            if stop_event.is_set():
                break
            t0 = time.perf_counter()
            _ = self.model(self.x)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            t1 = time.perf_counter()

            lat_ms.append((t1 - t0) * 1000.0)
            iters += 1

            if (t1 - start) >= duration_s:
                break

        elapsed = time.perf_counter() - start
        images = iters * self.batch_size
        throughput = images / elapsed if elapsed > 0 else 0.0

        lat_ms.sort()
        p50 = lat_ms[int(0.50 * (len(lat_ms) - 1))] if lat_ms else None
        p95 = lat_ms[int(0.95 * (len(lat_ms) - 1))] if lat_ms else None

        return {
            "workload": self.name,
            "units": "images",
            "iterations": iters,
            "duration_s": elapsed,
            "throughput_items_s": throughput,
            "latency_ms_p50": p50,
            "latency_ms_p95": p95,
            "notes": f"device={self.device},batch={self.batch_size},precision={self.precision},img={self.image_size}",
        }
    def __init__(self) -> None:
        self.torch = None
        self.device: str = "cpu"
        self.batch_size: int = 1
        self.precision: str = "fp32"
        self.image_size: int = 224
        self.model = None
        self.x = None

WORKLOAD_CLASSES: Dict[str, Type[BaseWorkload]] = {
    MatmulWorkload.name: MatmulWorkload,
    ResNet50Workload.name: ResNet50Workload,
}


# ----------------------------
# Metrics (using telemetry)
# ----------------------------

def compute_energy_joules(samples: List[TelemetrySample]) -> Optional[float]:
    pts = [(s.t_ns, s.gpu_power_w) for s in samples if s.gpu_power_w is not None]
    if len(pts) < 2:
        return None
    pts.sort(key=lambda x: x[0])
    energy_j = 0.0
    for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
        dt_s = (t1 - t0) / 1e9
        energy_j += 0.5 * (p0 + p1) * dt_s
    return float(energy_j)


def compute_avg_power(samples: List[TelemetrySample]) -> Optional[float]:
    vals = [s.gpu_power_w for s in samples if s.gpu_power_w is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def compute_max_temp(samples: List[TelemetrySample]) -> Optional[float]:
    vals = [s.gpu_temp_c for s in samples if s.gpu_temp_c is not None]
    if not vals:
        return None
    return float(max(vals))


def compute_avg_temp(samples: List[TelemetrySample]) -> Optional[float]:
    vals = [s.gpu_temp_c for s in samples if s.gpu_temp_c is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


# ----------------------------
# Baseline / Regression
# ----------------------------

def baseline_key(cfg: RunCfg) -> str:
    return f"{cfg.workload}|model={cfg.model}|batch={cfg.batch_size}|prec={cfg.precision}|img={cfg.image_size}|n={cfg.matmul_n}"


def load_registry(out_dir: Path) -> Dict[str, Any]:
    p = out_dir / "registry.json"
    if p.exists():
        return read_json(p)
    return {"runs": []}


def save_registry(out_dir: Path, registry: Dict[str, Any]) -> None:
    write_json(out_dir / "registry.json", registry)


def find_latest_pass_baseline(registry: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    runs = registry.get("runs", [])
    for r in reversed(runs):
        if r.get("baseline_key") == key and r.get("gate_status") == "PASS":
            return r
    return None


# ----------------------------
# Benchmark Session
# ----------------------------

class BenchmarkSession:
    def __init__(self, cfg: RunCfg):
        self.cfg = cfg
        self.stop_event = threading.Event()

        self.session_id = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%SZ')}_perfw_{cfg.workload}_{uuid.uuid4().hex[:6]}"
        self.out_root = Path(cfg.output_dir).expanduser().resolve()
        self.session_dir = self.out_root / self.session_id
        self.repeats_dir = self.session_dir / "repeats"

        safe_mkdir(self.repeats_dir)

        self.host = socket.gethostname()
        self.influx = InfluxWriter(cfg.influx)
        self.influx.start()

        if cfg.grafana.enabled and cfg.grafana.open_on_start:
            self._open_grafana(session_id=self.session_id, run_id=None)

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> Path:
        write_json(self.session_dir / "config.json", dataclasses.asdict(self.cfg))
        write_json(self.session_dir / "session_metadata.json", {
            "session_id": self.session_id,
            "created_utc": utc_now_iso(),
            "host": self.host,
            "os": platform.platform(),
            "notes": self.cfg.notes,
        })

        registry = load_registry(self.out_root)
        bkey = baseline_key(self.cfg)
        baseline_rec = None
        if self.cfg.regression.enabled and self.cfg.regression.baseline_policy == "latest_pass":
            baseline_rec = find_latest_pass_baseline(registry, bkey)

        results = []
        summaries = []

        for i in range(1, int(self.cfg.repeats) + 1):
            if self.stop_event.is_set():
                break
            rep_results, rep_summary, gate_status = self._run_repeat(i, baseline_rec)
            results.append(rep_results)
            summaries.append({**rep_summary, "gate_status": gate_status})

        thr = [float(r["throughput_items_s"]) for r in results if "throughput_items_s" in r]
        avg_thr, std_thr = mean_std(thr)

        session_summary = {
            "session_id": self.session_id,
            "workload": self.cfg.workload,
            "model": self.cfg.model,
            "created_utc": utc_now_iso(),
            "repeat_count": len(results),
            "throughput_mean_items_s": avg_thr,
            "throughput_std_items_s": std_thr,
            "repeat_summaries": summaries,
        }
        write_json(self.session_dir / "session_summary.json", session_summary)

        self.influx.stop()
        return self.session_dir

    def _run_repeat(self, repeat_index: int, baseline_rec: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
        run_id = f"{self.session_id}_r{repeat_index:02d}"
        run_dir = self.repeats_dir / f"r{repeat_index:02d}_{uuid.uuid4().hex[:8]}"
        safe_mkdir(run_dir)

        meta = {
            "run_id": run_id,
            "session_id": self.session_id,
            "repeat_index": repeat_index,
            "created_utc": utc_now_iso(),
            "host": self.host,
            "workload": self.cfg.workload,
            "model": self.cfg.model,
            "cfg": {
                "duration_s": self.cfg.duration_s,
                "warmup_s": self.cfg.warmup_s,
                "batch_size": self.cfg.batch_size,
                "precision": self.cfg.precision,
                "image_size": self.cfg.image_size,
                "matmul_n": self.cfg.matmul_n,
                "allow_tf32": self.cfg.allow_tf32,
                "notes": self.cfg.notes,
            }
        }
        write_json(run_dir / "run_metadata.json", meta)

        tags = {
            "host": self.host,
            "session_id": self.session_id,
            "run_id": run_id,
            "workload": self.cfg.workload,
            "model": self.cfg.model,
            "repeat": str(repeat_index),
            "precision": self.cfg.precision,
            "batch_size": str(self.cfg.batch_size),
            "image_size": str(self.cfg.image_size),
            "matmul_n": str(self.cfg.matmul_n),
        }

        telem_csv = run_dir / "telemetry.csv"
        telem = TelemetryCollector(self.cfg.sample_hz, telem_csv, self.influx if self.influx.enabled else None, tags)
        telem.start()

        wl_cls = WORKLOAD_CLASSES.get(self.cfg.workload)
        if wl_cls is None:
            raise ValueError(f"Unknown workload: {self.cfg.workload}. Available: {list(WORKLOAD_CLASSES.keys())}")
        wl = wl_cls()
        wl.setup(self.cfg)

        warm_end = time.time() + float(self.cfg.warmup_s)
        while time.time() < warm_end and not self.stop_event.is_set():
            wl.warmup(self.stop_event)

        results = wl.run_timed(float(self.cfg.duration_s), self.stop_event)
        write_json(run_dir / "results.json", results)

        telem.stop()

        avg_power = compute_avg_power(telem.samples)
        max_temp = compute_max_temp(telem.samples)
        avg_temp = compute_avg_temp(telem.samples)
        energy_j = compute_energy_joules(telem.samples)

        throughput = float(results.get("throughput_items_s", 0.0))
        run_duration = float(results.get("duration_s", 0.0))

        perf_w = (throughput / avg_power) if (avg_power and avg_power > 0) else None

        total_units = throughput * run_duration
        energy_per_unit = (energy_j / total_units) if (energy_j and total_units > 0) else None

        summary = {
            "run_id": run_id,
            "session_id": self.session_id,
            "repeat_index": repeat_index,
            "workload": self.cfg.workload,
            "model": self.cfg.model,
            "units": results.get("units", "items"),
            "run_duration_s": run_duration,
            "throughput_items_s": throughput,
            "avg_gpu_power_w": avg_power,
            "max_gpu_temp_c": max_temp,
            "avg_gpu_temp_c": avg_temp,
            "energy_j": energy_j,
            "perf_per_w": perf_w,
            "energy_per_unit_j": energy_per_unit,
            "baseline_key": baseline_key(self.cfg),
        }
        write_json(run_dir / "summary.json", summary)

        self._write_summary_to_influx(tags, summary)

        gate_status, incident = self._gate_and_incident(summary, baseline_rec)
        if incident:
            write_json(run_dir / "incident.json", incident)

        registry = load_registry(self.out_root)
        record = {
            "timestamp_utc": utc_now_iso(),
            "session_id": self.session_id,
            "run_id": run_id,
            "workload": self.cfg.workload,
            "model": self.cfg.model,
            "repeat_index": repeat_index,
            "baseline_key": summary["baseline_key"],
            "gate_status": gate_status,
            "metrics": {
                "throughput_items_s": throughput,
                "avg_gpu_power_w": avg_power,
                "max_gpu_temp_c": max_temp,
                "perf_per_w": perf_w,
                "energy_j": energy_j,
                "energy_per_unit_j": energy_per_unit,
            },
            "path": str(run_dir),
        }
        registry["runs"].append(record)
        save_registry(self.out_root, registry)

        if self.cfg.grafana.enabled and self.cfg.grafana.open_on_start:
            self._open_grafana(session_id=self.session_id, run_id=run_id)

        return results, summary, gate_status

    def _write_summary_to_influx(self, tags: Dict[str, str], summary: Dict[str, Any]) -> None:
        if not self.influx.enabled:
            return
        tag_str = ",".join([f"{k}={_lp_escape_tag(v)}" for k, v in tags.items()])

        fields = {}
        for k in [
            "throughput_items_s",
            "avg_gpu_power_w",
            "max_gpu_temp_c",
            "avg_gpu_temp_c",
            "energy_j",
            "perf_per_w",
            "energy_per_unit_j",
            "run_duration_s",
        ]:
            v = summary.get(k)
            if v is None:
                continue
            fields[k] = float(v)

        if not fields:
            return

        field_str = ",".join([f"{k}={v:.6f}" for k, v in fields.items()])
        lp = f"ai_benchmark,{tag_str} {field_str} {ns_now()}"
        self.influx.enqueue(lp)

    def _gate_and_incident(self, summary: Dict[str, Any], baseline_rec: Optional[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not self.cfg.regression.enabled:
            return "NA", None
        if not baseline_rec:
            return "PASS", {
                "type": "NO_BASELINE",
                "message": "No PASS baseline exists yet (latest_pass policy). This run becomes the initial baseline candidate.",
                "run_id": summary["run_id"],
                "baseline_key": summary["baseline_key"],
            }

        new_pw = summary.get("perf_per_w")
        old_pw = ((baseline_rec.get("metrics") or {}).get("perf_per_w"))

        if new_pw is None or old_pw is None:
            return "WARN", {
                "type": "INSUFFICIENT_DATA",
                "message": "Cannot compute perf/W gate because power telemetry is missing in baseline or current run.",
                "run_id": summary.get("run_id"),
                "baseline_run_id": baseline_rec.get("run_id"),
            }

        drop_pct = -pct_change(float(new_pw), float(old_pw))  # positive = regression
        status = "PASS"
        if drop_pct >= self.cfg.regression.fail_drop_pct:
            status = "FAIL"
        elif drop_pct >= self.cfg.regression.warn_drop_pct:
            status = "WARN"

        incident = {
            "type": "PERFW_REGRESSION",
            "gate_status": status,
            "run_id": summary.get("run_id"),
            "baseline_run_id": baseline_rec.get("run_id"),
            "baseline_key": summary.get("baseline_key"),
            "perf_per_w_new": new_pw,
            "perf_per_w_baseline": old_pw,
            "drop_pct": drop_pct,
            "triage_hint": "Check laptop power mode, thermals/throttling, driver changes, and background load.",
        }
        return status, incident

    def _open_grafana(self, session_id: str, run_id: Optional[str]) -> None:
        url = (self.cfg.grafana.dashboard_url or "").strip()
        if not url:
            return
        sep = "&" if "?" in url else "?"
        url2 = f"{url}{sep}var-session_id={session_id}"
        if run_id:
            url2 += f"&var-run_id={run_id}"
        try:
            webbrowser.open(url2)
        except requests.RequestException:
            pass

