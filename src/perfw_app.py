import dataclasses
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from perfw_core import RunCfg, BenchmarkSession, WORKLOAD_CLASSES


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perf/W SLO Runner (v1)")
        self.geometry("880x620")

        self.session: BenchmarkSession | None = None
        self.worker: threading.Thread | None = None

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        top = ttk.Frame(self)
        top.pack(fill="x", **pad)

        ttk.Button(top, text="Load Config", command=self.load_config).pack(side="left")
        ttk.Button(top, text="Save Config", command=self.save_config).pack(side="left", padx=6)

        self.btn_run = ttk.Button(top, text="RUN", command=self.run_clicked)
        self.btn_run.pack(side="left", padx=12)

        self.btn_stop = ttk.Button(top, text="STOP", command=self.stop_clicked, state="disabled")
        self.btn_stop.pack(side="left")

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, **pad)

        left = ttk.LabelFrame(frm, text="Benchmark Settings")
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        right = ttk.LabelFrame(frm, text="Influx / Grafana / Gates")
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(0, weight=1)

        # Vars
        self.var_out = tk.StringVar(value="runs")
        self.var_workload = tk.StringVar(value="resnet50")
        self.var_model = tk.StringVar(value="resnet50")
        self.var_duration = tk.DoubleVar(value=60.0)
        self.var_warmup = tk.DoubleVar(value=10.0)
        self.var_repeats = tk.IntVar(value=3)
        self.var_sample_hz = tk.DoubleVar(value=2.0)

        self.var_batch = tk.IntVar(value=32)
        self.var_prec = tk.StringVar(value="fp16")
        self.var_img = tk.IntVar(value=224)
        self.var_n = tk.IntVar(value=2048)
        self.var_tf32 = tk.BooleanVar(value=True)
        self.var_notes = tk.StringVar(value="")

        self.var_influx_enabled = tk.BooleanVar(value=True)
        self.var_influx_url = tk.StringVar(value="http://localhost:8181")
        self.var_influx_db = tk.StringVar(value="perfw")

        self.var_graf_enabled = tk.BooleanVar(value=True)
        self.var_graf_url = tk.StringVar(value="http://localhost:3000/d/perf-w-ai/perf-w-ai-ai-workloads-live?orgId=1")
        self.var_graf_open = tk.BooleanVar(value=True)

        self.var_gate_enabled = tk.BooleanVar(value=True)
        self.var_warn = tk.DoubleVar(value=5.0)
        self.var_fail = tk.DoubleVar(value=10.0)

        # Left column widgets
        r = 0
        ttk.Label(left, text="Output folder").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_out, width=34).grid(row=r, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(left, text="Browse", command=self.browse_out).grid(row=r, column=2, padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Workload").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        wl = ttk.Combobox(left, textvariable=self.var_workload, values=list(WORKLOAD_CLASSES.keys()), state="readonly")
        wl.grid(row=r, column=1, sticky="ew", padx=8, pady=4)
        wl.bind("<<ComboboxSelected>>", lambda _e: self._sync_model_default())

        r += 1
        ttk.Label(left, text="Model label").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_model).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Duration (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_duration).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Warmup (s)").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_warmup).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Repeats").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_repeats).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Telemetry Hz").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_sample_hz).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Batch size").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_batch).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Precision").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        prec = ttk.Combobox(left, textvariable=self.var_prec, values=["fp32", "fp16"], state="readonly")
        prec.grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="ResNet image size").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_img).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Matmul N").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_n).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        r += 1
        ttk.Checkbutton(left, text="Allow TF32 (matmul)", variable=self.var_tf32).grid(row=r, column=1, sticky="w", padx=8, pady=4)

        r += 1
        ttk.Label(left, text="Notes").grid(row=r, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(left, textvariable=self.var_notes).grid(row=r, column=1, sticky="ew", padx=8, pady=4)

        left.columnconfigure(1, weight=1)

        # Right column widgets
        rr = 0
        ttk.Checkbutton(right, text="Enable Influx writes", variable=self.var_influx_enabled).grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        rr += 1
        ttk.Label(right, text="Influx URL").grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(right, textvariable=self.var_influx_url).grid(row=rr, column=1, sticky="ew", padx=8, pady=4)

        rr += 1
        ttk.Label(right, text="Database").grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(right, textvariable=self.var_influx_db).grid(row=rr, column=1, sticky="ew", padx=8, pady=4)

        rr += 1
        ttk.Checkbutton(right, text="Open Grafana dashboard", variable=self.var_graf_enabled).grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        rr += 1
        ttk.Label(right, text="Grafana dashboard URL").grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(right, textvariable=self.var_graf_url).grid(row=rr, column=1, sticky="ew", padx=8, pady=4)

        rr += 1
        ttk.Checkbutton(right, text="Open on run start", variable=self.var_graf_open).grid(row=rr, column=0, sticky="w", padx=8, pady=4)

        rr += 1
        ttk.Checkbutton(right, text="Enable regression gates", variable=self.var_gate_enabled).grid(row=rr, column=0, sticky="w", padx=8, pady=6)

        rr += 1
        ttk.Label(right, text="WARN drop %").grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(right, textvariable=self.var_warn).grid(row=rr, column=1, sticky="ew", padx=8, pady=4)

        rr += 1
        ttk.Label(right, text="FAIL drop %").grid(row=rr, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(right, textvariable=self.var_fail).grid(row=rr, column=1, sticky="ew", padx=8, pady=4)

        right.columnconfigure(1, weight=1)

        # Log
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.txt = tk.Text(log_frame, height=10)
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)
        self.log("Ready. Set your config and click RUN. (Token must be set as INFLUXDB3_AUTH_TOKEN env var.)")

    def _sync_model_default(self):
        wl = self.var_workload.get()
        if not self.var_model.get().strip():
            self.var_model.set(wl)
        elif self.var_model.get() in ("resnet50", "matmul"):
            self.var_model.set(wl)

    def log(self, msg: str):
        self.txt.insert("end", msg + "\n")
        self.txt.see("end")

    def browse_out(self):
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.var_out.set(d)

    def current_cfg(self) -> RunCfg:
        d = {
            "output_dir": self.var_out.get(),
            "workload": self.var_workload.get(),
            "model": self.var_model.get().strip() or self.var_workload.get(),
            "duration_s": float(self.var_duration.get()),
            "warmup_s": float(self.var_warmup.get()),
            "repeats": int(self.var_repeats.get()),
            "sample_hz": float(self.var_sample_hz.get()),

            "batch_size": int(self.var_batch.get()),
            "precision": self.var_prec.get(),
            "image_size": int(self.var_img.get()),
            "matmul_n": int(self.var_n.get()),
            "allow_tf32": bool(self.var_tf32.get()),

            "notes": self.var_notes.get(),

            "influx": {
                "enabled": bool(self.var_influx_enabled.get()),
                "url": self.var_influx_url.get(),
                "database": self.var_influx_db.get(),
                "precision": "nanosecond",
                "accept_partial": True,
                "no_sync": True
            },
            "grafana": {
                "enabled": bool(self.var_graf_enabled.get()),
                "dashboard_url": self.var_graf_url.get(),
                "open_on_start": bool(self.var_graf_open.get())
            },
            "regression": {
                "enabled": bool(self.var_gate_enabled.get()),
                "warn_drop_pct": float(self.var_warn.get()),
                "fail_drop_pct": float(self.var_fail.get()),
                "baseline_policy": "latest_pass"
            }
        }
        return RunCfg.from_dict(d)

    def run_clicked(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running", "A run is already in progress.")
            return

        cfg = self.current_cfg()
        self.session = BenchmarkSession(cfg)

        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")

        self.log(f"RUN started: workload={cfg.workload}, model={cfg.model}, duration={cfg.duration_s}s, repeats={cfg.repeats}")
        self.log("Tip: If Influx is enabled, make sure INFLUXDB3_AUTH_TOKEN is set in your environment.")

        def job():
            try:
                out = self.session.run()
                self.log(f"✅ Finished. Artifacts in: {out}")
                self.log("If Grafana shows no data, increase dashboard time range or confirm datasource URL.")
            except Exception as e:
                self.log(f"❌ Error: {type(e).__name__}: {e}")
            finally:
                self.btn_run.config(state="normal")
                self.btn_stop.config(state="disabled")

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    def stop_clicked(self):
        if self.session:
            self.log("Stop requested…")
            self.session.request_stop()

    def load_config(self):
        p = filedialog.askopenfilename(title="Load config JSON", filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return

        self.var_out.set(d.get("output_dir", self.var_out.get()))
        self.var_workload.set(d.get("workload", self.var_workload.get()))
        self.var_model.set(d.get("model", self.var_model.get()))
        self.var_duration.set(d.get("duration_s", self.var_duration.get()))
        self.var_warmup.set(d.get("warmup_s", self.var_warmup.get()))
        self.var_repeats.set(d.get("repeats", self.var_repeats.get()))
        self.var_sample_hz.set(d.get("sample_hz", self.var_sample_hz.get()))
        self.var_batch.set(d.get("batch_size", self.var_batch.get()))
        self.var_prec.set(d.get("precision", self.var_prec.get()))
        self.var_img.set(d.get("image_size", self.var_img.get()))
        self.var_n.set(d.get("matmul_n", self.var_n.get()))
        self.var_tf32.set(d.get("allow_tf32", self.var_tf32.get()))
        self.var_notes.set(d.get("notes", self.var_notes.get()))

        influx = d.get("influx", {})
        self.var_influx_enabled.set(influx.get("enabled", self.var_influx_enabled.get()))
        self.var_influx_url.set(influx.get("url", self.var_influx_url.get()))
        self.var_influx_db.set(influx.get("database", self.var_influx_db.get()))

        graf = d.get("grafana", {})
        self.var_graf_enabled.set(graf.get("enabled", self.var_graf_enabled.get()))
        self.var_graf_url.set(graf.get("dashboard_url", self.var_graf_url.get()))
        self.var_graf_open.set(graf.get("open_on_start", self.var_graf_open.get()))

        reg = d.get("regression", {})
        self.var_gate_enabled.set(reg.get("enabled", self.var_gate_enabled.get()))
        self.var_warn.set(reg.get("warn_drop_pct", self.var_warn.get()))
        self.var_fail.set(reg.get("fail_drop_pct", self.var_fail.get()))

        self.log(f"Loaded config: {p}")

    def save_config(self):
        p = filedialog.asksaveasfilename(
            title="Save config JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if not p:
            return
        cfg = self.current_cfg()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(cfg), f, indent=2)
        self.log(f"Saved config: {p}")


if __name__ == "__main__":
    App().mainloop()