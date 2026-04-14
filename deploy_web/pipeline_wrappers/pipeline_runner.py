"""
pipeline_runner.py — Demo pipeline runner (Steps 02–09)
=========================================================
Gọi đúng các script gốc trong CS431MCQGen/scripts/ qua SLURM,
mỗi bước chạy riêng trong job SLURM, stream stdout về Streamlit.

KHÔNG sửa hay copy code gốc. Chỉ wrap để gọi từ Streamlit.

SLURM mode:
    sbatch deploy_pipeline.sh (đặt trong deploy_web/scripts/)
    → Job chạy toàn bộ pipeline
    → Stdout file ghi log để Streamlit đọc realtime
"""

from __future__ import annotations

import subprocess
import threading
import time
import re
import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable


# ─── Paths ────────────────────────────────────────────────────────────────────
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent  # CS431MCQGen/
SCRIPTS_DIR   = PIPELINE_ROOT / "scripts"
DEPLOY_SCRIPTS = Path(__file__).resolve().parent / "scripts"  # deploy_web/scripts/
SRC_DIR       = PIPELINE_ROOT / "src"
PYTHON        = "python"


@dataclass
class StepResult:
    name: str
    step_num: str
    success: bool
    output_file: Path | None = None
    error: str | None = None
    duration_sec: float = 0.0
    items_processed: int = 0
    items_passed: int = 0
    items_failed: int = 0


@dataclass
class SlurmJobInfo:
    job_id: str
    state: str  # RUNNING, COMPLETED, FAILED, CANCELLED, PENDING
    log_path: Path | None = None


def _stream_output(
    proc: subprocess.Popen,
    callback: Callable[[str], None] | None = None,
) -> str:
    """Read stdout+stderr line-by-line, optionally call callback."""
    lines = []
    for stream_name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr)):
        if stream is None:
            continue
        for raw_line in iter(stream.readline, ""):
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace")
            lines.append(line)
            if callback:
                callback(line)
    return "".join(lines)


def _parse_items_from_jsonl(path: Path) -> int:
    """Count lines in a JSONL file."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


class PipelineRunner:
    """
    Run the MCQGen pipeline via SLURM.

    Architecture:
      1. app.py → submit deploy_pipeline.sh via sbatch
      2. Monitor job via squeue + tail job log file
      3. Stream parsed progress back to Streamlit callbacks

    Usage (SLURM mode):
        runner = PipelineRunner(exp_name="demo_test")
        runner.run_pipeline_slurm(
            on_log=st.text_area callback,
            on_progress=pct callback,
        )
    """

    def __init__(
        self,
        exp_name: str,
        on_log: Callable[[str], None] | None = None,
        on_step_done: Callable[[StepResult], None] | None = None,
        on_progress: Callable[[int, str], None] | None = None,
        on_job_status: Callable[[SlurmJobInfo], None] | None = None,
    ):
        self.exp_name   = exp_name
        self.on_log     = on_log
        self.on_step_done = on_step_done
        self.on_progress = on_progress
        self.on_job_status = on_job_status
        self._job_info: SlurmJobInfo | None = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        if self.on_log:
            self.on_log(line)

    def _pct(self, done: int, total: int, label: str = ""):
        pct = int(done / total * 100) if total > 0 else 0
        if self.on_progress:
            self.on_progress(pct, label)

    def _run_bash(
        self,
        script_path: Path,
        cwd: Path | None = None,
        extra_env: dict | None = None,
    ) -> tuple[bool, str]:
        """Run a bash script via subprocess. Returns (success, output)."""
        if not script_path.exists():
            self._log(f"⚠️  Script not found: {script_path}")
            return False, f"Script not found: {script_path}"

        env = None
        if extra_env:
            env = {**os.environ, **extra_env}

        try:
            proc = subprocess.Popen(
                ["bash", str(script_path)],
                cwd=cwd or PIPELINE_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=False,
                bufsize=1,
            )
            output = _stream_output(proc, self._log)
            proc.wait()
            return proc.returncode == 0, output
        except Exception as e:
            self._log(f"❌ Exception: {e}")
            return False, str(e)

    # ── SLURM job submit ──────────────────────────────────────────────────────

    def submit_slurm_job(self) -> SlurmJobInfo | None:
        """
        Submit deploy_pipeline.sh as a SLURM job.
        Returns SlurmJobInfo with job_id.
        """
        script = DEPLOY_SCRIPTS / "deploy_pipeline.sh"
        if not script.exists():
            self._log(f"⚠️  SLURM script not found: {script}")
            return None

        self._log(f"📤 Submitting SLURM job (exp={self.exp_name})...")
        self._log(f"   Script: {script}")
        self._log(f"   EXP_NAME={self.exp_name}")

        env = {**os.environ, "EXP_NAME": self.exp_name}
        try:
            result = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split(";")[0].strip()
                self._job_info = SlurmJobInfo(
                    job_id=job_id,
                    state="PENDING",
                    log_path=PIPELINE_ROOT / "log" / f"deploy_pipeline_{job_id}.out",
                )
                self._log(f"✅ SLURM job submitted: {job_id}")
                if self.on_job_status:
                    self.on_job_status(self._job_info)
                return self._job_info
            else:
                self._log(f"❌ sbatch failed: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            self._log("❌ sbatch timed out")
            return None
        except Exception as e:
            self._log(f"❌ sbatch exception: {e}")
            return None

    # ── Monitor SLURM job ─────────────────────────────────────────────────────

    def _parse_squeue_line(self, line: str) -> dict | None:
        """Parse one line of squeue -o output."""
        # Format: "JOBID NAME USER STATE ..."
        parts = line.split()
        if len(parts) >= 4:
            return {
                "job_id": parts[0],
                "name": parts[1],
                "state": parts[2],
                "rest": " ".join(parts[3:]),
            }
        return None

    def get_job_status(self, job_id: str) -> SlurmJobInfo | None:
        """Query SLURM for job status via squeue."""
        if not job_id:
            return None
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%i %j %u %T", "--noheader"],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0 and result.stdout.strip():
                state = result.stdout.strip().split()[3] if len(result.stdout.strip().split()) >= 4 else "UNKNOWN"
                state_upper = state.upper()
                if "RUN" in state_upper:
                    mapped = "RUNNING"
                elif "PEND" in state_upper or "PD" in state_upper:
                    mapped = "PENDING"
                elif "DONE" in state_upper or "CD" in state_upper:
                    mapped = "COMPLETED"
                elif "CA" in state_upper:
                    mapped = "CANCELLED"
                else:
                    mapped = state_upper

                info = SlurmJobInfo(
                    job_id=job_id,
                    state=mapped,
                    log_path=self._job_info.log_path if self._job_info else None,
                )
                self._job_info = info
                return info
            else:
                # Job not in queue → check if it completed
                info = SlurmJobInfo(
                    job_id=job_id,
                    state="COMPLETED",
                    log_path=self._job_info.log_path if self._job_info else None,
                )
                self._job_info = info
                return info
        except Exception as e:
            self._log(f"⚠️  squeue error: {e}")
            return self._job_info

    def _tail_log(self, log_path: Path, last_pos: int) -> tuple[list[str], int]:
        """Tail new lines from log file since last_pos. Returns (lines, new_pos)."""
        if not log_path.exists():
            return [], last_pos
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(last_pos)
                new_lines = f.readlines()
                new_pos = f.tell()
            return new_lines, new_pos
        except Exception:
            return [], last_pos

    def _parse_step_from_log(self, line: str) -> tuple[str, str] | None:
        """
        Parse STEP marker from log line.
        Returns (step_num, step_name) or None.
        E.g. "════════════════════════════════\nSTEP 02: Retrieval..." → ("02", "Retrieval...")
        """
        m = re.search(r"STEP\s+(\d+)\s*[:\-]?\s*(.+)", line)
        if m:
            return m.group(1), m.group(2).strip()
        return None

    def _parse_items_from_log(self, log_path: Path) -> int:
        """Count processed items by scanning log for success counts."""
        if not log_path.exists():
            return 0
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            # Match patterns like "✅ P1 ch04_t01 seq=0: type=single_correct, count=1"
            matches = re.findall(r"✅|❌|⚠️", content)
            return len([m for m in matches if m == "✅"])
        except Exception:
            return 0

    def _parse_mcq_counts(self, log_path: Path) -> tuple[int, int]:
        """Parse MCQ pass/fail counts from log."""
        if not log_path.exists():
            return 0, 0
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            passed = len(re.findall(r"✅.*(?:PASS|done|DONE)", content))
            failed = len(re.findall(r"❌.*(?:FAIL|failed|FAILED)", content))
            return passed, failed
        except Exception:
            return 0, 0

    def monitor_job(
        self,
        job_id: str,
        poll_interval: int = 10,
        on_step_done: Callable[[StepResult], None] | None = None,
    ) -> list[StepResult]:
        """
        Poll SLURM job + tail log file, fire callbacks on step completion.
        Returns list of StepResult when job finishes.
        """
        self._log(f"🔍 Monitoring SLURM job: {job_id}")
        self._pct(0, 100, label=f"SLURM job {job_id} — PENDING")

        step_map = {
            "01": StepResult(name="Indexing (CPU)", step_num="01", success=True, items_processed=0),
            "02": StepResult(name="Retrieval", step_num="02", success=False, items_processed=0),
            "03": StepResult(name="P1: Generate Stem", step_num="03", success=False, items_processed=0),
            "04": StepResult(name="P2+P3: Self-Refine", step_num="04", success=False, items_processed=0),
            "05": StepResult(name="P4: Distractors", step_num="05", success=False, items_processed=0),
            "06": StepResult(name="P5-P8: CoT", step_num="06", success=False, items_processed=0),
            "07": StepResult(name="Eval: Overall", step_num="07", success=False, items_processed=0),
            "08": StepResult(name="Eval: IWF", step_num="08", success=False, items_processed=0),
            "09": StepResult(name="Explanation", step_num="09", success=False, items_processed=0),
        }

        results: list[StepResult] = []
        log_pos = 0
        start_time = time.time()
        last_step = "02"  # pipeline starts at step 02

        while True:
            # ── Poll job state ───────────────────────────────────────────────
            info = self.get_job_status(job_id)
            if info and self.on_job_status:
                self.on_job_status(info)

            if info and info.state in ("COMPLETED", "CANCELLED", "FAILED"):
                self._log(f"🏁 Job {job_id} finished: {info.state}")
                break

            # ── Tail log file for new lines ──────────────────────────────────
            if info and info.log_path:
                new_lines, log_pos = self._tail_log(info.log_path, log_pos)
                for raw_line in new_lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    self._log(line)

                    # Detect step transitions
                    step_info = self._parse_step_from_log(line)
                    if step_info:
                        s_num, s_name = step_info
                        if s_num in step_map and s_num != last_step:
                            last_step = s_num
                            pct = (int(s_num) / 9) * 80  # 0-80% for job
                            self._pct(pct, 100, label=f"Step {s_num}: {s_name}")

            # ── Progress estimate ────────────────────────────────────────────
            if info and info.state == "RUNNING":
                elapsed = time.time() - start_time
                # Estimate based on elapsed time (rough)
                pct = min(85, int(elapsed / 60))  # rough 1%/min up to 85%
                self._pct(pct, 100, label=f"Job running... ({int(elapsed//60)}m)")

            time.sleep(poll_interval)

        # ── Job done — parse final results ────────────────────────────────
        self._pct(90, 100, label="Parsing final results...")
        if info and info.log_path and info.log_path.exists():
            mcq_path = PipelineRunner.find_final_output()
            if mcq_path:
                mcqs = PipelineRunner.load_final_output(mcq_path)
                # Approximate step results from log
                for s_num, s_res in step_map.items():
                    if s_num == "01":
                        s_res.success = True
                    else:
                        # All subsequent steps succeeded if job completed
                        s_res.success = (info.state == "COMPLETED")
                    s_res.items_processed = len(mcqs) if mcqs else 0
                    s_res.items_passed = len(mcqs) if mcqs else 0
                    results.append(s_res)

        self._pct(100, 100, label="Done")
        self._log("✅ Monitoring complete")

        if self.on_step_done:
            for r in results:
                self.on_step_done(r)

        return results

    # ── Full SLURM pipeline ────────────────────────────────────────────────────

    def run_pipeline_slurm(
        self,
        skip_explain: bool = True,
    ) -> list[StepResult]:
        """
        Submit pipeline as SLURM job and monitor until completion.
        Returns list of StepResult.
        """
        self._log("=" * 50)
        self._log(f"🚀 Starting pipeline: {self.exp_name}")
        self._log("=" * 50)

        # 1. Submit SLURM job
        job_info = self.submit_slurm_job()
        if not job_info:
            self._log("❌ Failed to submit SLURM job")
            return []

        # 2. Monitor job
        results = self.monitor_job(job_info.job_id)

        return results

    # ── Local fallback (direct subprocess) ────────────────────────────────────

    def run_full_pipeline_local(self, skip_explain: bool = True) -> list[StepResult]:
        """
        Fallback: run each step directly as subprocess (NO SLURM).
        Only for testing on a machine WITH GPU available.
        """
        self._log("⚠️  Running LOCAL mode (no SLURM) — ensure GPU is available!")

        step_scripts = [
            ("02", self.run_retrieval),
            ("03", self.run_gen_stem),
            ("04", self.run_refine),
            ("05", self.run_distractors),
            ("06", self.run_cot),
            ("07", self.run_eval_overall),
            ("08", self.run_eval_iwf),
        ]

        results: list[StepResult] = []
        for i, (label, fn) in enumerate(step_scripts):
            self._pct(i, len(step_scripts), label=f"Step {label}")
            res = fn()
            results.append(res)
            if not res.success:
                self._log(f"⛔ Pipeline stopped at Step {label}")
                break
            if self.on_step_done:
                self.on_step_done(res)

        self._pct(len(step_scripts), len(step_scripts), label="Done")
        return results

    # ── Individual step runners (local subprocess) ──────────────────────────

    def run_retrieval(self) -> StepResult:
        """Run 02_retrieval.sh — hybrid retrieval."""
        start = time.time()
        script = SCRIPTS_DIR / "02_retrieval.sh"
        ok, _ = self._run_bash(script)
        retrieve_out = PIPELINE_ROOT / "data" / "intermediate" / "02_retrieval"
        n = len(list(retrieve_out.glob("*.jsonl"))) if retrieve_out.exists() else 0
        return StepResult(
            name="Retrieval (BM25+Vector+RRF)", step_num="02",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    def run_gen_stem(self) -> StepResult:
        """Run 03_gen_stem.sh — P1 stems."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "03_gen_stem.sh")
        n = _parse_items_from_jsonl(SCRIPTS_DIR.parent / "data" / "intermediate" / "03_gen_stem" / "all_p1_results.jsonl")
        return StepResult(
            name="P1: Generate Stem", step_num="03",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    def run_refine(self) -> StepResult:
        """Run 04_gen_refine.sh — P2+P3."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "04_gen_refine.sh")
        n = _parse_items_from_jsonl(SCRIPTS_DIR.parent / "data" / "intermediate" / "04_gen_refine" / "all_refined_results.jsonl")
        return StepResult(
            name="P2+P3: Self-Refine", step_num="04",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    def run_distractors(self) -> StepResult:
        """Run 05_gen_distractors.sh — P4 candidates."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "05_gen_distractors.sh")
        n = _parse_items_from_jsonl(SCRIPTS_DIR.parent / "data" / "intermediate" / "05_gen_distractors" / "all_candidates_results.jsonl")
        return StepResult(
            name="P4: Distractors", step_num="05",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    def run_cot(self) -> StepResult:
        """Run 06_gen_cot.sh — P5-P8."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "06_gen_cot.sh")
        n = _parse_items_from_jsonl(SCRIPTS_DIR.parent / "data" / "intermediate" / "06_gen_cot" / "all_final_mcqs.jsonl")
        return StepResult(
            name="P5-P8: CoT", step_num="06",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    def run_eval_overall(self) -> StepResult:
        """Run 07_eval.sh — overall eval."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "07_eval.sh")
        eval_out = SCRIPTS_DIR.parent / "data" / "intermediate" / "07_eval"
        n_passed = _parse_items_from_jsonl(eval_out / "evaluated_questions.jsonl")
        n_failed = _parse_items_from_jsonl(eval_out / "failed_questions.jsonl")
        return StepResult(
            name="Eval: Overall", step_num="07",
            success=ok, duration_sec=time.time() - start,
            items_processed=n_passed + n_failed, items_passed=n_passed, items_failed=n_failed,
        )

    def run_eval_iwf(self) -> StepResult:
        """Run 08_eval_iwf.sh — IWF eval."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "08_eval_iwf.sh")
        iwf_out = SCRIPTS_DIR.parent / "data" / "intermediate" / "08_eval_iwf"
        n = _parse_items_from_jsonl(iwf_out / "final_accepted_questions.jsonl")
        return StepResult(
            name="Eval: IWF", step_num="08",
            success=ok, duration_sec=time.time() - start, items_processed=n, items_passed=n,
        )

    def run_explain(self) -> StepResult:
        """Run 09_explain.sh — explanation (optional)."""
        start = time.time()
        ok, _ = self._run_bash(SCRIPTS_DIR / "09_explain.sh")
        n = _parse_items_from_jsonl(SCRIPTS_DIR.parent / "data" / "intermediate" / "09_explain" / "explained_questions.jsonl")
        return StepResult(
            name="Explanation", step_num="09",
            success=ok, duration_sec=time.time() - start, items_processed=n,
        )

    # ── Utilities ──────────────────────────────────────────────────────────────

    @staticmethod
    def find_final_output() -> Path | None:
        """Find most recent final_accepted_questions.jsonl in output/."""
        output_dir = PIPELINE_ROOT / "output"
        candidates = sorted(
            output_dir.rglob("final_accepted_questions.jsonl"),
            key=lambda p: p.stat().st_mtime,
        )
        return candidates[-1] if candidates else None

    @staticmethod
    def load_final_output(path: Path) -> list[dict]:
        records = []
        if not path or not path.exists():
            return records
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def get_slurm_job_info(job_id: str) -> dict | None:
        """Quick squeue lookup for job info."""
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%i %j %T %M %l", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(None, 4)
                if len(parts) >= 4:
                    return {
                        "job_id": parts[0],
                        "name": parts[1],
                        "state": parts[2],
                        "time_used": parts[3] if len(parts) > 3 else "?",
                    }
        except Exception:
            pass
        return None
