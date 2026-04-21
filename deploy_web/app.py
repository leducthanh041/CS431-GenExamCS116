"""
app.py — MCQGen Demo Web UI (v3)
=================================
Feature: step-by-step execution, per-step retry, topic list editor,
         experiment selector + persistent results.

Changes from v2:
  1. Topic list isolation: editor writes ONLY to output/<exp>/topic_list.json,
     never touches input/topic_list.json (the source-of-truth file).
  2. Smart step status: a step is marked DONE only when its output file is
     valid (non-empty + valid JSONL). Otherwise a Re-check button appears.
     Re-run is always available regardless of status.
  3. New experiment form: no auto-generated timestamp in the text field.
     User types their own name. Text field retains value (no on_change rerun).

Chạy:
    cd CS431MCQGen/deploy_web
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import streamlit as st
import subprocess
import time
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# ── Add paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR.parent / "src"))

from html_render.mcq_renderer import (
    render_mcq_card,
    stats_summary,
    render_stats_html,
)
from pipeline_wrappers.pipeline_runner import PipelineRunner, StepResult

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MCQGen",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PIPELINE_ROOT = BASE_DIR.parent
DEPLOY_SCRIPTS = BASE_DIR / "scripts"
LOG_DIR = PIPELINE_ROOT / "log"
TOPIC_LIST_FILE = PIPELINE_ROOT / "input" / "topic_list.json"

STEP_META = {
    2: {"label": "02 — Retrieval", "name": "Hybrid Retrieval (BM25+Vector+RRF)", "script": "02_retrieval.sh", "python": "src/gen/retrieval.py"},
    3: {"label": "03 — Gen Stem", "name": "P1: Generate Stem + Key", "script": "03_gen_stem.sh", "python": "src/gen/p1_gen_stem.py"},
    4: {"label": "04 — Refine", "name": "P2+P3: Self-Refine", "script": "04_gen_refine.sh", "python": "src/gen/p2_p3_refine.py"},
    5: {"label": "05 — Distractors", "name": "P4: Distractor Candidates", "script": "05_gen_distractors.sh", "python": "src/gen/p4_candidates.py"},
    6: {"label": "06 — CoT", "name": "P5-P8: CoT Distractor Selection", "script": "06_gen_cot.sh", "python": "src/gen/p5_p8_cot.py"},
    7: {"label": "07 — Eval", "name": "Eval: Overall (8 criteria)", "script": "07_eval.sh", "python": "src/eval/eval_overall.py"},
    8: {"label": "08 — IWF", "name": "Eval: IWF (Item Writing Flaws)", "script": "08_eval_iwf.sh", "python": "src/eval/eval_iwf.py"},
    9: {"label": "09 — Explain", "name": "Explanation Generation", "script": "09_explain.sh", "python": "src/gen/explain_mcq.py"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def exp_topic_list_path(exp_name: str) -> Path:
    """Path đến topic_list riêng của mỗi experiment."""
    return PIPELINE_ROOT / "output" / exp_name / "topic_list.json"

def load_topic_list(exp_name: str = "") -> list:
    """Load topic list: ưu tiên file riêng của exp, fallback về file gốc."""
    if exp_name:
        exp_path = exp_topic_list_path(exp_name)
        if exp_path.exists():
            with open(exp_path, encoding="utf-8") as f:
                return json.load(f)
    if not TOPIC_LIST_FILE.exists():
        return []
    with open(TOPIC_LIST_FILE, encoding="utf-8") as f:
        return json.load(f)

def save_topic_list(data: list, exp_name: str = ""):
    """Lưu topic list: vào file riêng của exp (không ghi đè file gốc)."""
    if exp_name:
        path = exp_topic_list_path(exp_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    # Nếu không có exp_name → không lưu (bảo vệ file gốc)

def list_experiments() -> list[dict]:
    """Scan output/ for all experiments with step info."""
    output_dir = PIPELINE_ROOT / "output"
    step_dirs = {
        2: "02_retrieval", 3: "03_gen_stem", 4: "04_gen_refine",
        5: "05_gen_distractors", 6: "06_gen_cot", 7: "07_eval",
        8: "08_eval_iwf", 9: "09_explain",
    }
    experiments = []
    if not output_dir.exists():
        return experiments
    for d in sorted(output_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        # Issue #2 fix: check VALID outputs, not just directory existence
        done_steps, step_counts = _get_valid_steps(d, step_dirs)
        # Final MCQ count — use the best available final file
        mcqs = []
        for _final_name in ["08_eval_iwf/final_accepted_questions.jsonl",
                             "08_eval_iwf/evaluated_questions.jsonl"]:
            fpath = d / _final_name
            if fpath.exists() and fpath.stat().st_size > 0:
                with open(fpath, encoding="utf-8") as f:
                    mcqs = [json.loads(l) for l in f if l.strip()]
                break
        mcq_count = len(mcqs)
        experiments.append({
            "name": d.name,
            "path": d,
            "done_steps": done_steps,
            "step_counts": step_counts,
            "mcq_count": mcq_count,
            "mcqs": mcqs,
            "mtime": datetime.fromtimestamp(d.stat().st_mtime),
        })
    return experiments


# ── Issue #2: per-step output validators ──────────────────────────────────────
STEP_OUTPUT_FILES = {
    # step_num: (output_dir_name, expected_jsonl_name_or_glob_pattern)
    # For step 2: many per-topic files (e.g. ch04_t01.jsonl) — use glob
    2: ("02_retrieval", "*.jsonl"),           # any .jsonl in the dir = valid
    3: ("03_gen_stem", "all_p1_results.jsonl"),
    4: ("04_gen_refine", "all_refined_results.jsonl"),
    5: ("05_gen_distractors", "all_candidates_results.jsonl"),
    6: ("06_gen_cot", "all_final_mcqs.jsonl"),
    7: ("07_eval", "evaluated_questions.jsonl"),
    8: ("08_eval_iwf", "final_accepted_questions.jsonl"),
    9: ("09_explain", "explanations.jsonl"),
}


def _validate_step_output(exp_dir: Path, step_num: int) -> tuple[bool, int]:
    """
    Check if step output is valid (non-empty, valid JSONL).
    Returns (is_valid, item_count).
    - is_valid=True: output exists, non-empty, all lines are parseable JSON
    - is_valid=False: missing / empty / partially corrupted

    Supports glob patterns (e.g. "*.jsonl") for steps with per-topic files.
    """
    if step_num not in STEP_OUTPUT_FILES:
        return True, 0  # unknown step, assume OK
    dir_name, pattern = STEP_OUTPUT_FILES[step_num]
    step_dir = exp_dir / dir_name

    if not step_dir.exists():
        return False, 0

    # Handle glob patterns (step 2: many per-topic .jsonl files)
    if "*" in pattern:
        files = sorted(step_dir.glob(pattern))
        if not files:
            return False, 0
        total_count = 0
        for fpath in files:
            if fpath.stat().st_size == 0:
                return False, 0  # one empty file → invalid
            try:
                with open(fpath, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        json.loads(line)
                        total_count += 1
            except (json.JSONDecodeError, OSError):
                return False, total_count
        return total_count > 0, total_count

    # Exact filename
    fpath = step_dir / pattern
    if not fpath.exists() or fpath.stat().st_size == 0:
        return False, 0

    count = 0
    try:
        with open(fpath, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                json.loads(line)  # raise if invalid
                count += 1
    except (json.JSONDecodeError, OSError):
        return False, count  # partially corrupted → invalid

    return count > 0, count


def _get_valid_steps(exp_dir: Path, step_dirs_map: dict) -> tuple[list[int], dict[int, int]]:
    """
    Scan experiment directory, return (valid_step_nums, step_counts).
    A step is VALID only if its output JSONL is non-empty and parseable.
    """
    valid_steps: list[int] = []
    counts: dict[int, int] = {}
    for step_num, dir_name in step_dirs_map.items():
        is_valid, cnt = _validate_step_output(exp_dir, step_num)
        counts[step_num] = cnt
        if is_valid:
            valid_steps.append(step_num)
    return valid_steps, counts

def load_explanations(exp_name: str) -> dict[str, dict]:
    """
    Load explanations.jsonl → dict keyed by question_id.
    Each value is the FULL record (top-level keys: explanation, sources, sources_used, etc.).
    Returns {} if file missing or invalid.
    """
    path = PIPELINE_ROOT / "output" / exp_name / "09_explain" / "explanations.jsonl"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    explanations: dict[str, dict] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                q_id = record.get("question_id", "")
                if q_id:
                    explanations[q_id] = record
    except (json.JSONDecodeError, OSError):
        pass
    return explanations


def merge_mcqs_with_explanations(mcqs: list[dict], explanations: dict[str, dict]) -> list[dict]:
    """
    Join MCQ list with explanation data by question_id.
    Injects explanation fields directly into each MCQ dict for the renderer.

    Actual structure in explanations.jsonl (full record):
      {
        "question_id": "cs116_04_0",
        "explanation": {
          "question_motivation": "...",
          "correct_answer_rationale": "...",
          "distractor_explanations": {"B": "...", ...},
          "knowledge_context": {...},
          "sources_used": [{"type":"slide","url":"",...}, {"type":"video","url":"...",...}]
        },
        "sources": [{"type":"slide",...}, {"type":"video",...}]   ← flat list
      }

    Renderer expects:  top-level keys: explanation (dict), sources (list),
                       slide_citations, video_citations
    """
    enriched = []
    for mcq in mcqs:
        q_id = mcq.get("question_id", "")
        record = explanations.get(q_id, {})
        if record:
            enriched_mcq = dict(mcq)
            # explanation field = the nested explanation dict
            exp_dict = record.get("explanation", {})
            enriched_mcq["explanation"] = exp_dict
            # sources_used + sources → collect all into flat sources list for renderer
            sources_used = exp_dict.get("sources_used", []) if isinstance(exp_dict, dict) else []
            sources_flat = record.get("sources", [])
            enriched_mcq["sources"] = sources_used + sources_flat
            # Split into slide / video citations for dedicated rendering
            if isinstance(exp_dict, dict):
                enriched_mcq["distractor_explanations"] = exp_dict.get("distractor_explanations", {})
                enriched_mcq["slide_citations"] = [
                    s for s in sources_used
                    if isinstance(s, dict) and s.get("type") == "slide"
                ]
                enriched_mcq["video_citations"] = [
                    s for s in sources_used
                    if isinstance(s, dict) and s.get("type") == "video"
                ]
            else:
                enriched_mcq["distractor_explanations"] = {}
                enriched_mcq["slide_citations"] = []
                enriched_mcq["video_citations"] = []
            enriched.append(enriched_mcq)
        else:
            enriched.append(mcq)
    return enriched

def get_exp_log(exp_name: str) -> str:
    """Find and return last ~50 lines of deploy_pipeline log for this exp."""
    log_files = sorted(LOG_DIR.glob(f"deploy_pipeline_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    for lf in log_files:
        # Check if this log belongs to this experiment
        try:
            with open(lf, encoding="utf-8", errors="replace") as f:
                content = f.read()
            if exp_name in content or "demo_" + exp_name in content or exp_name.replace("demo_", "") in content:
                lines = content.splitlines()
                return "\n".join(lines[-80:])
        except Exception:
            pass
    return ""

def submit_step_slurm(step_num: int, exp_name: str) -> tuple[bool, str, str]:
    """Submit a single step via SLURM sbatch. Returns (ok, msg, job_id)."""
    meta = STEP_META[step_num]
    script = DEPLOY_SCRIPTS / "deploy_pipeline.sh"
    if not script.exists():
        return False, f"Script not found: {script}", ""

    # Build a special SBATCH wrapper that runs only one step
    # Use srun inside the job instead of full pipeline script
    python_file = PIPELINE_ROOT / meta["python"]
    step_name = meta["label"].split(" — ")[0]

    # Create a temporary job script for single step
    tmp_script = LOG_DIR / f"_step_{step_num}_{exp_name}_$$.sh"
    with open(tmp_script, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=mcqgen_s{step_num}
#SBATCH --output={LOG_DIR}/step_{step_num}_{exp_name}_%j.out
#SBATCH --error={LOG_DIR}/step_{step_num}_{exp_name}_%j.err
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=64G
#SBATCH --gres=mps:l40:2 --time=12:00:00 

set -e
module clear -f 2>/dev/null || true
source /datastore/uittogether/tools/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
unset CUDA_VISIBLE_DEVICES

PROJECT_ROOT="{PIPELINE_ROOT}"
REQUIRED_VRAM=36000

CHECK_OUT=$("$PROJECT_ROOT/scripts/gpu_check.sh" $REQUIRED_VRAM $SLURM_JOB_ID) || true
EXIT_CODE=$?
if [ $EXIT_CODE -eq 10 ] || [ $EXIT_CODE -eq 11 ]; then echo "$CHECK_OUT"; exit 1; fi
BEST_GPU=$CHECK_OUT

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-s$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-s$SLURM_JOB_ID
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
export CUDA_VISIBLE_DEVICES=$BEST_GPU
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"
export EXP_NAME="{exp_name}"

# Copy experiment-specific topic_list.json to input/ before running
EXP_TOPIC_LIST="$PROJECT_ROOT/output/{exp_name}/topic_list.json"
if [ -f "$EXP_TOPIC_LIST" ]; then
    cp "$EXP_TOPIC_LIST" "$PROJECT_ROOT/input/topic_list.json"
    echo "📋 Using topic_list from experiment: $EXP_TOPIC_LIST"
fi

echo "STEP {step_num}: {meta['name']} | GPU $BEST_GPU | Exp: {exp_name}"
python -u "{python_file}"
rm -rf $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
echo "STEP {step_num} done"
""")

    os.chmod(tmp_script, 0o755)
    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(tmp_script)],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "EXP_NAME": exp_name},
        )
        os.remove(tmp_script)
        if result.returncode == 0:
            job_id = result.stdout.strip().split(";")[0].strip()
            return True, f"Job {job_id} submitted", job_id
        else:
            return False, f"sbatch failed: {result.stderr}", ""
    except subprocess.TimeoutExpired:
        return False, "sbatch timed out", ""
    except Exception as e:
        try:
            os.remove(tmp_script)
        except Exception:
            pass
        return False, str(e), ""

def cancel_slurm_job(job_id: str) -> tuple[bool, str]:
    """Cancel a running SLURM job. Returns (ok, message)."""
    if not job_id:
        return False, "No job ID provided"
    try:
        result = subprocess.run(
            ["scancel", job_id],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, f"Job {job_id} cancelled"
        else:
            return False, f"scancel failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "scancel timed out"
    except Exception as e:
        return False, str(e)


def submit_full_pipeline_slurm(exp_name: str) -> tuple[bool, str, str]:
    """Submit the full pipeline (02-09) as a single SLURM job. Returns (ok, msg, job_id)."""
    script = DEPLOY_SCRIPTS / "deploy_pipeline.sh"
    if not script.exists():
        return False, f"Script not found: {script}", ""

    try:
        result = subprocess.run(
            ["sbatch", "--parsable", str(script)],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "EXP_NAME": exp_name},
        )
        if result.returncode == 0:
            job_id = result.stdout.strip().split(";")[0].strip()
            return True, f"Full pipeline job {job_id} submitted", job_id
        else:
            return False, f"sbatch failed: {result.stderr}", ""
    except subprocess.TimeoutExpired:
        return False, "sbatch timed out", ""
    except Exception as e:
        return False, str(e), ""


def poll_full_pipeline_until_done(
    job_id: str,
    exp_name: str,
    log_container,
    progress_bar,
    status_text,
    step_status_container,
):
    """Poll full pipeline SLURM job, update UI, stop on cancel."""
    ALL_STEPS = [2, 3, 4, 5, 6, 7, 8, 9]
    STEP_WEIGHTS = {
        2: 12, 3: 25, 4: 38, 5: 50, 6: 62, 7: 75, 8: 88, 9: 95,
    }
    log_path = LOG_DIR / f"deploy_pipeline_{job_id}.out"
    log_pos = 0
    log_lines = []
    last_step = 0
    cancelled = False

    while True:
        # ── Check cancellation ───────────────────────────────────────────────
        # (We check st.session_state in the calling block; here we just poll state)
        # ── Job state ────────────────────────────────────────────────────────
        state = "UNKNOWN"
        try:
            r = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                state = r.stdout.strip().upper()
        except Exception:
            pass

        icon_map = {
            "RUNNING": "🟢", "PENDING": "🟡", "COMPLETED": "✅",
            "FAILED": "❌", "CANCELLED": "🚫",
        }
        icon = icon_map.get(state, "⚪")
        status_text.info(f"{icon} Full Pipeline — Job `{job_id}` — {state}")

        # ── Tail log ─────────────────────────────────────────────────────────
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(log_pos)
                    new_lines = f.readlines()
                    log_pos = f.tell()
                for raw_line in new_lines:
                    line = raw_line.strip()
                    if not line:
                        continue
                    log_lines.append(line)
                    # Detect step transitions: "STEP 02: ..."
                    import re as _re
                    m = _re.search(r"STEP\s+(\d+)\s*[:\-]?\s*(.+)", line)
                    if m:
                        s_num = int(m.group(1))
                        if s_num != last_step and s_num in ALL_STEPS:
                            last_step = s_num
                            step_status_container.info(
                                f"⚡ Step {s_num}: {m.group(2).strip()}"
                            )
            except Exception:
                pass

        if len(log_lines) > 500:
            log_lines = log_lines[-500:]

        # ── Progress ─────────────────────────────────────────────────────────
        if state == "RUNNING":
            pct = STEP_WEIGHTS.get(last_step, 5)
            progress_bar.progress(
                pct / 100,
                text=f"Full Pipeline — Step {last_step or '?'}: Running..."
            )
        elif state == "PENDING":
            progress_bar.progress(0.02, text="Full Pipeline — Waiting for GPU...")
        elif state in ("COMPLETED", "FAILED", "CANCELLED"):
            progress_bar.progress(1.0, text=f"Job {state}")
            # Bug fix: rerun immediately so grid refreshes right away
            time.sleep(1)
            st.rerun()
            return state, last_step

        # ── Update log display ────────────────────────────────────────────────
        log_container.text_area(
            "📋 Full Pipeline Log",
            value="\n".join(log_lines[-200:]),
            height=240,
            disabled=True,
            key=f"full_log_{job_id}_{log_pos}",
        )
        time.sleep(12)

    return state, last_step


def check_slurm_available() -> bool:
    try:
        r = subprocess.run(["sinfo", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def poll_job_until_done(job_id: str, step_num: int, exp_name: str,
                        log_container, progress_bar, status_text,
                        full_pipeline: bool = False):
    """
    Poll squeue + tail log until job completes.
    Returns final state string: COMPLETED / FAILED / CANCELLED.

    full_pipeline=True: do NOT call st.rerun() on completion
    → Just return the state, let the full pipeline block handle advancement.
    This is critical: st.rerun() inside a while loop spins forever
    because Streamlit blocks the while loop until the next user interaction.
    """
    step_pct = {
        2: 12, 3: 25, 4: 38, 5: 50, 6: 62, 7: 75, 8: 88, 9: 95,
    }
    log_path = LOG_DIR / f"step_{step_num}_{exp_name}_{job_id}.out"
    log_pos = 0
    log_lines = []
    iteration = 0

    while True:
        # Job state
        state = "UNKNOWN"
        try:
            r = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                state = r.stdout.strip().upper()
        except Exception:
            pass

        icon = {"RUNNING": "🟢", "PENDING": "🟡", "COMPLETED": "✅",
                "FAILED": "❌", "CANCELLED": "🚫"}.get(state, "⚪")
        status_text.info(f"{icon} Step {step_num} — {state}")

        # Tail log
        new_lines, log_pos = [], log_pos
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(log_pos)
                    new_lines = f.readlines()
                    log_pos = f.tell()
            except Exception:
                pass

        for raw_line in new_lines:
            line = raw_line.strip()
            if line:
                log_lines.append(line)
        if len(log_lines) > 500:
            log_lines = log_lines[-500:]

        # Progress
        pct = step_pct.get(step_num, 10)
        if state == "RUNNING":
            progress_bar.progress(pct / 100, text=f"Step {step_num}: {STEP_META[step_num]['name']} — Running")
        elif state == "PENDING":
            progress_bar.progress(0.05, text="Waiting for GPU allocation...")
        elif state in ("COMPLETED", "FAILED", "CANCELLED"):
            progress_bar.progress(1.0, text=f"Job {state}")
            # Clear step_running so grid shows fresh status
            st.session_state.step_running = 0
            if "_recheck_versions" not in st.session_state:
                st.session_state._recheck_versions = {}
            for sn in STEP_OUTPUT_FILES:
                st.session_state._recheck_versions.pop(sn, None)
            # KEY FIX: if in full pipeline mode, just return (don't st.rerun)
            # st.rerun() blocks the while loop until user interaction — causes infinite loops
            if full_pipeline:
                return state
            # Per-step mode: st.rerun so grid refreshes immediately
            time.sleep(1)
            st.rerun()
            return state

        # Update log
        iteration += 1
        log_container.text_area(
            "Log",
            value="\n".join(log_lines[-200:]),
            height=200,
            disabled=True,
            key=f"log_s{step_num}_{job_id}_{iteration}",
        )
        time.sleep(12)


# ═══════════════════════════════════════════════════════════════════════════════
# Session state defaults
# ═══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("current_exp", ""),
    ("topic_edit_mode", False),
    ("step_running", 0),
    ("job_id_map", {}),
    # Issue #3 fix: explicit flag so we never touch text_input value involuntarily
    ("_new_exp_typed", ""),
    ("_form_submitted", False),
    # New: Full pipeline sequential tracking
    ("_full_next_step", None),   # next step to run (2..9), None = idle
    ("_full_exp_name", ""),     # experiment name for the full pipeline run
    ("_trigger_full_refresh", False),  # flag: poll done → trigger st.rerun via button
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# Header — Experiment Selector + New Experiment
# ═══════════════════════════════════════════════════════════════════════════════

col_logo, col_exp, col_topic = st.columns([0.5, 2.5, 1])

with col_logo:
    st.markdown("### 📝 **MCQGen**")

# ── Issue #3 fix: stable new-exp form that NEVER refreshes the text field ──
experiments = list_experiments()
exp_options = [e["name"] for e in experiments]

# Resolve current experiment — available to ALL sections (Topic Editor, Pipeline, Results)
exp_name = st.session_state.current_exp
exp_obj = next((e for e in experiments if e["name"] == exp_name), None)
done_steps = exp_obj["done_steps"] if exp_obj else []
step_counts = exp_obj.get("step_counts", {}) if exp_obj else {}
is_running = st.session_state.step_running > 0

with col_exp:
    with st.form("new_exp_form", clear_on_submit=False):
        c_left, c_btn = st.columns([4, 1])
        with c_left:
            # text_input has NO on_change, NO auto-timestamp — only user types here
            typed = st.text_input(
                "Tên experiment mới",
                value=st.session_state._new_exp_typed,
                placeholder="Nhập tên experiment mới...",
                label_visibility="collapsed",
            )
            # Persist typed value so it survives reruns
            st.session_state._new_exp_typed = typed
        with c_btn:
            st.markdown("")  # vertical alignment spacer
            submitted = st.form_submit_button("✨ Tạo mới", use_container_width=True)

        if submitted:
            name = typed.strip()
            if name:
                if name in exp_options:
                    st.warning(f"⚠️ Experiment **`{name}`** đã tồn tại. Đã chọn.")
                st.session_state.current_exp = name
                st.session_state._new_exp_typed = ""  # reset after create
                st.rerun()
            else:
                st.warning("⚠️ Vui lòng nhập tên experiment.")

    # Existing experiment selector (only shown when no current exp is set,
    # or shown as a small dropdown below the form)
    if st.session_state.current_exp:
        cur = st.session_state.current_exp
        # Highlight current exp name
        st.caption(f"📂 Đang chọn: **`{cur}`**")
    else:
        if exp_options:
            selected = st.selectbox(
                "— hoặc chọn experiment có sẵn:",
                options=exp_options,
                label_visibility="collapsed",
            )
            if st.button("Chọn", key="sel_from_list"):
                st.session_state.current_exp = selected
                st.rerun()
        else:
            st.info("Chưa có experiment nào. Nhập tên bên trên để tạo mới.")

with col_topic:
    st.markdown("")  # spacer
    if st.button("📋 Chỉnh sửa Topic List", use_container_width=True):
        st.session_state.topic_edit_mode = not st.session_state.topic_edit_mode
        st.rerun()

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC LIST EDITOR (collapsible)
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.topic_edit_mode:
    with st.expander("📋 Topic List Editor", expanded=True):
        topic_data = load_topic_list(exp_name)
        topic_json = json.dumps(topic_data, ensure_ascii=False, indent=2)

        edited_json = st.text_area(
            "topic_list.json (JSON) — riêng cho experiment này",
            value=topic_json,
            height=400,
            label_visibility="collapsed",
        )

        col_save, col_validate, col_reset, _ = st.columns([1, 1, 1, 3])
        with col_save:
            if st.button("💾 Lưu vào Experiment", use_container_width=True):
                try:
                    parsed = json.loads(edited_json)
                    save_topic_list(parsed, exp_name)
                    st.success(f"✅ Đã lưu topic_list riêng cho `{exp_name}`")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON lỗi: {e}")

        with col_validate:
            if st.button("🔍 Kiểm tra JSON", use_container_width=True):
                try:
                    parsed = json.loads(edited_json)
                    total_topics = sum(len(ch.get("topics", [])) for ch in parsed)
                    st.success(f"✅ OK — {len(parsed)} chapters, {total_topics} topics")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON lỗi: {e}")

        with col_reset:
            if st.button("↩️ Khôi phục gốc", use_container_width=True):
                st.session_state.topic_edit_mode = False
                st.rerun()

        # Issue #1 fix: crystal-clear isolation message
        st.success(
            f"📁 **File riêng của experiment:** `output/{exp_name}/topic_list.json`\n\n"
            "🔒 **File gốc an toàn:** `input/topic_list.json` — hoàn toàn không bị ảnh hưởng.\n\n"
            "💡 Topic list này chỉ dùng cho **Step 02 (Retrieval)** trở đi của experiment `"
            f"{exp_name}`. Mỗi experiment có topic list riêng biệt."
        )
        if st.button("🔒 Đóng Editor", key="close_topic_editor"):
            st.session_state.topic_edit_mode = False
            st.rerun()

    st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

# exp_name already resolved at top of module (after experiment form)

# Track which steps have been manually re-checked this session
if "_recheck_versions" not in st.session_state:
    st.session_state._recheck_versions = {}

# ── Full Pipeline + Control Bar ───────────────────────────────────────────────
# Two modes: per-step (step_running) OR full sequential (_full_next_step)
is_full_running = bool(st.session_state.get("_full_next_step") and st.session_state.get("_full_exp_name") == exp_name)

if exp_name:
    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 4])
    with ctrl_col1:
        st.markdown("**Controls**")

    with ctrl_col2:
        if is_full_running:
            if st.button("Stop", key="stop_full", use_container_width=True):
                st.session_state._full_next_step = None
                st.session_state._full_exp_name  = ""
                st.rerun()
        else:
            if st.button("Run Full (02-09)", key="run_full", use_container_width=True):
                st.session_state._full_next_step = 2
                st.session_state._full_exp_name   = exp_name
                st.rerun()

    with ctrl_col3:
        if is_full_running:
            next_s = st.session_state._full_next_step
            st.info(f"Full pipeline: Step {next_s} - {STEP_META[next_s]['name']}")
        else:
            st.caption("Run individual steps or click 'Run Full' for 02-09 sequentially.")

    # ── Visible Refresh button (manual + auto-triggered after each step) ──────
    # IMPORTANT: clear the flag FIRST on this render so it only fires ONCE.
    # After JS reload: page renders fresh, flag=False → no infinite reload loop.
    _pending = st.session_state.get("_trigger_full_refresh", False)
    if _pending:
        st.session_state._trigger_full_refresh = False  # clear BEFORE JS fires

    if _pending:
        st.info("⏳ Pipeline step done — auto-refreshing...")

    if st.button("🔄 Refresh", key="manual_refresh_btn", use_container_width=False):
        st.rerun()

    # Auto-reload via JS only when a pipeline step just completed.
    # Flag is already cleared above → reload fires at most ONCE per step.
    if _pending:
        import streamlit.components.v1 as components
        components.html(
            "<script>setTimeout(function(){parent.location.reload();}, 500);</script>",
            height=0, scrolling=False,
        )

st.markdown("---")

# Full pipeline execution:
# - Submit step via submit_step_slurm()
# - Poll DIRECTLY with full_pipeline=True (poll returns, does NOT st.rerun)
# - On COMPLETED: advance _full_next_step, st.rerun() ONCE
# - On FAILED/CANCELLED: stop
# - Grid shows per-step status; full panel shows current step + log
#
# IMPORTANT: poll_job_until_done(..., full_pipeline=True) returns state
#            WITHOUT calling st.rerun(). This lets the full pipeline block
#            advance to next step after each completion.

if is_full_running:
    step_running = st.session_state._full_next_step

    # Show current step header + progress
    st.markdown(f"### Full Pipeline - Step {step_running}: {STEP_META[step_running]['name']}")
    progress_bar = st.progress(0.02, text="Submitting...")
    status_text  = st.empty()
    log_container = st.empty()

    ok, msg, job_id = submit_step_slurm(step_running, exp_name)

    if ok and job_id:
        # Poll until done — full_pipeline=True means poll returns without st.rerun
        state = poll_job_until_done(
            job_id=job_id, step_num=step_running, exp_name=exp_name,
            log_container=log_container, progress_bar=progress_bar, status_text=status_text,
            full_pipeline=True,
        )

        # Job is done
        if state == "COMPLETED":
            next_step = (step_running + 1) if step_running < 9 else None
            if next_step is None:
                st.session_state._full_next_step = None
                st.session_state._full_exp_name   = ""
                st.success("Full pipeline completed!")
            else:
                st.session_state._full_next_step = next_step
        else:
            st.session_state._full_next_step = None
            st.session_state._full_exp_name = ""
            st.error(f"Step {step_running} failed: {state}")

        # KEY: set refresh flag, then use button to trigger st.rerun()
        # st.rerun() inside while loop is the bug — using button click instead
        st.session_state._trigger_full_refresh = True
    else:
        st.session_state._full_next_step = None
        st.session_state._full_exp_name = ""
        st.error(f"Submit failed: {msg}")
        st.session_state._trigger_full_refresh = True

st.markdown("---")




# ── Per-step run panel (disabled when full pipeline is active) ─────────────────
# Full pipeline takes exclusive control; per-step panel does NOT run during full pipeline
is_step_running = st.session_state.step_running > 0 and not is_full_running

if is_step_running:
    step_running = st.session_state.step_running
    st.markdown(f"#### Step {step_running}: {STEP_META[step_running]['name']}")

    progress_bar = st.progress(0, text="Submitting...")
    status_text = st.empty()
    log_container = st.empty()

    ok, msg, job_id = submit_step_slurm(step_running, exp_name)

    if ok and job_id:
        state = poll_job_until_done(
            job_id=job_id, step_num=step_running, exp_name=exp_name,
            log_container=log_container, progress_bar=progress_bar, status_text=status_text,
            full_pipeline=False,
        )
        if state == "COMPLETED":
            st.session_state.step_running = 0
            st.success(f"Step {step_running} completed!")
        else:
            st.error(f"Step {step_running} failed: {state}")
            st.session_state.step_running = 0
        st.rerun()
    else:
        st.error(f"Submit failed: {msg}")
        st.session_state.step_running = 0
        st.rerun()

st.markdown("---")

# ── Step grid ──────────────────────────────────────────────────────────────────
# State logic (mutually exclusive per step):
#   is_valid           → green card ✅ + "🔁 Chạy lại"
#   dir exists+incomplete → amber card ⚠️ + "🔍 Kiểm tra lại"
#   not started        → gray card ⬜ + "▶️ Chạy"
#   is_running         → yellow card ⏳  (block per-step buttons while any step is running)

# When full pipeline runs: track _full_next_step; per-step buttons show "⏳ Full đang chạy"
ALL_STEPS = [2, 3, 4, 5, 6, 7, 8, 9]

# Resolve step status fresh each render (no caching of dir_exists/file_size)
def _step_status(step_num):
    step_dir = PIPELINE_ROOT / "output" / exp_name / STEP_OUTPUT_FILES[step_num][0]
    dir_exists = step_dir.exists()
    pattern = STEP_OUTPUT_FILES[step_num][1]
    if "*" in pattern:
        files = list(step_dir.glob(pattern)) if dir_exists else []
        file_exists = bool(files)
        file_size = sum(f.stat().st_size for f in files) if files else 0
    else:
        fpath = step_dir / pattern
        file_exists = fpath.exists()
        file_size = fpath.stat().st_size if file_exists else 0
    is_valid = step_num in done_steps
    item_count = step_counts.get(step_num, 0)
    cached = st.session_state._recheck_versions.get(step_num)
    if cached is not None:
        item_count = cached
    return dir_exists, file_exists, file_size, is_valid, item_count

# is_full_running already resolved above from session state

rows = [(2, 3), (4, 5), (6, 7), (8, 9)]

for row_steps in rows:
    cols = st.columns([1] + [2] * len(row_steps))
    with cols[0]:
        st.markdown("")

    for i, step_num in enumerate(row_steps):
        meta = STEP_META[step_num]
        dir_exists, file_exists, file_size, is_valid, item_count = _step_status(step_num)
        is_this_running = st.session_state.step_running == step_num

        with cols[i + 1]:
            # ── Card (always rendered once) ──────────────────────────────────────
            if is_this_running:
                bg, border, icon = "#fef9c3", "#eab308", "⏳"
                status = "Đang chạy..."
            elif is_valid:
                bg, border, icon = "#f0fdf4", "#22c55e", "✅"
                status = f"Done — {item_count} items" if item_count else "Done"
            elif dir_exists and file_size > 0:
                bg, border, icon = "#fffbeb", "#f59e0b", "⚠️"
                status = f"Incomplete — {item_count} items" if item_count else "⚠️ Incomplete"
            else:
                bg, border, icon = "#f9fafb", "#e5e7eb", "⬜"
                status = "Chưa chạy"

            st.markdown(f"""
            <div style="background:{bg};border:2px solid {border};border-radius:10px;
                        padding:10px;margin:4px 0">
              <div style="font-weight:700;font-size:13px;color:#374151">{icon} {meta["label"]}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:2px">{meta["name"]}</div>
              <div style="font-size:10px;color:#9ca3af;margin-top:1px">{status}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Button (one per step, mutually exclusive) ──────────────────────
            if is_this_running:
                pass  # running → no action button
            elif is_valid:
                if is_full_running:
                    st.button(f"⏳ Full đang chạy", key=f"b_{step_num}_locked",
                              disabled=True, use_container_width=True)
                else:
                    if st.button(f"🔁 Chạy lại", key=f"b_{step_num}_rerun",
                                 use_container_width=True):
                        st.session_state.step_running = step_num
                        st.rerun()
            elif dir_exists and file_size > 0:
                if is_full_running:
                    st.button(f"⏳ Full đang chạy", key=f"b_{step_num}_locked",
                              disabled=True, use_container_width=True)
                else:
                    if st.button(f"🔍 Kiểm tra lại", key=f"b_{step_num}_recheck",
                                 use_container_width=True):
                        is_v, cnt = _validate_step_output(
                            PIPELINE_ROOT / "output" / exp_name, step_num
                        )
                        st.session_state._recheck_versions[step_num] = cnt
                        if is_v:
                            st.success(f"✅ Step {step_num} hợp lệ: {cnt} items")
                        else:
                            st.warning(f"⚠️ Step {step_num} chưa hợp lệ: {cnt} items. Cần chạy lại.")
                        st.rerun()
            else:
                if is_full_running:
                    st.button(f"⏳ Full đang chạy", key=f"b_{step_num}_locked",
                              disabled=True, use_container_width=True)
                elif not is_running:
                    if st.button(f"▶️ Chạy", key=f"b_{step_num}_run",
                                 use_container_width=True):
                        st.session_state.step_running = step_num
                        st.rerun()

# ── Step run panel ─────────────────────────────────────────────────────────────
if is_running:
    step_running = st.session_state.step_running
    st.markdown("---")
    st.markdown(f"#### ⚡ Đang chạy Step {step_running} — `{STEP_META[step_running]['name']}`")

    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()
    log_container = st.empty()

    ok, msg, job_id = submit_step_slurm(step_running, exp_name)

    if ok and job_id:
        state = poll_job_until_done(
            job_id=job_id,
            step_num=step_running,
            exp_name=exp_name,
            log_container=log_container,
            progress_bar=progress_bar,
            status_text=status_text,
        )
        if state == "COMPLETED":
            st.session_state.step_running = 0
            st.success(f"✅ Step {step_running} hoàn tất!")
        else:
            st.error(f"❌ Job {job_id} — {state}")
            st.session_state.step_running = 0
        st.rerun()
    else:
        st.error(f"❌ Submit failed: {msg}")
        st.session_state.step_running = 0

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Kết quả — Preview")

if exp_obj and exp_obj["mcqs"]:
    # Bug fix: join explanations into MCQs so render_mcq_card shows them
    explanations = load_explanations(exp_name)
    mcqs = merge_mcqs_with_explanations(exp_obj["mcqs"], explanations)
    has_explanations = bool(explanations)
    stats = stats_summary(mcqs)
    st.markdown(render_stats_html(stats), unsafe_allow_html=True)

    # Filters
    c1, c2, c3 = st.columns(3)
    with c1:
        type_filter = st.selectbox("Loại", ["Tất cả", "single_correct", "multiple_correct"])
    with c2:
        diff_filter = st.selectbox("Độ khó", ["Tất cả", "G1", "G2", "G3"])
    with c3:
        show_eval = st.checkbox("Hiện chi tiết eval")

    filtered = mcqs
    if type_filter != "Tất cả":
        filtered = [m for m in filtered if m.get("question_type") == type_filter]
    if diff_filter != "Tất cả":
        filtered = [m for m in filtered
                    if m.get("difficulty_label", m.get("difficulty", "")) == diff_filter]

    st.markdown(f"**{len(filtered)} / {len(mcqs)}** câu hỏi")

    page_size = 5
    n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Trang", min_value=1, max_value=n_pages, value=1)
    start_i = (page - 1) * page_size

    for mcq in filtered[start_i:start_i + page_size]:
        if show_eval:
            with st.expander(f"📋 `{mcq.get('question_id', '?')}`"):
                st.json(mcq, expanded=False)
        else:
            st.markdown(render_mcq_card(mcq), unsafe_allow_html=True)

elif exp_obj:
    st.info(f"Chưa có kết quả cho experiment này. Chạy các steps để tạo dữ liệu.")
else:
    st.info(f"Experiment `{exp_name}` chưa có dữ liệu. Chạy pipeline steps để tạo.")

# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
with st.expander("📁 Lịch sử Experiments", expanded=False):
    if experiments:
        for exp in experiments[:15]:
            done = sorted(exp["done_steps"])
            mtime = exp["mtime"].strftime("%d/%m/%Y %H:%M")
            step_counts = exp.get("step_counts", {})
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    badge = "✅" if exp["mcq_count"] > 0 else "⬜"
                    # Show item counts for key steps
                    counts_str = ""
                    if step_counts.get(8):
                        counts_str = f" **(IWF: {step_counts[8]})**"
                    st.markdown(f"{badge} **`{exp['name']}`** — {mtime}{counts_str}")
                with c2:
                    st.markdown(f"Steps: {done if done else '—'}")
                with c3:
                    if st.button("Chọn", key=f"sel_{exp['name']}"):
                        st.session_state.current_exp = exp["name"]
                        st.rerun()
                if exp["mcqs"]:
                    exp_explanations = load_explanations(exp["name"])
                    exp_mcqs = merge_mcqs_with_explanations(exp["mcqs"], exp_explanations)
                    stats = stats_summary(exp_mcqs)
                    st.markdown(render_stats_html(stats), unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("Chưa có experiment nào.")
