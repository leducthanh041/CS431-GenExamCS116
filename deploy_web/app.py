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
    # step_num: (output_dir_name, expected_jsonl_name)
    2: ("02_retrieval", "all_retrieval_results.jsonl"),
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
    """
    if step_num not in STEP_OUTPUT_FILES:
        return True, 0  # unknown step, assume OK
    dir_name, file_name = STEP_OUTPUT_FILES[step_num]
    fpath = exp_dir / dir_name / file_name

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

def load_exp_mcqs(exp_name: str) -> tuple[list[dict], str]:
    """Load final MCQs for an experiment. Returns (mcqs, path_note)."""
    output_dir = PIPELINE_ROOT / "output" / exp_name
    for final_name in ["08_eval_iwf/final_accepted_questions.jsonl",
                       "08_eval_iwf/evaluated_questions.jsonl",
                       "08_eval_iwf/final_rejected_questions.jsonl"]:
        fpath = output_dir / final_name
        if fpath.exists():
            mcqs = []
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            mcqs.append(json.loads(line))
                        except Exception:
                            pass
            path_note = fpath.name
            return mcqs, path_note
    return [], ""

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

def check_slurm_available() -> bool:
    try:
        r = subprocess.run(["sinfo", "--version"], capture_output=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return False

def poll_job_until_done(job_id: str, step_num: int, exp_name: str,
                        log_container, progress_bar, status_text):
    """Poll squeue + tail log, update Streamlit widgets in place."""
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
                    new_lines = [l.strip() for l in f.readlines() if l.strip()]
                    log_pos = f.tell()
            except Exception:
                pass

        log_lines.extend(new_lines)
        if len(log_lines) > 500:
            log_lines = log_lines[-500:]

        # Progress
        pct = step_pct.get(step_num, 10)
        if state == "RUNNING":
            progress_bar.progress(pct / 100, text=f"Step {step_num}: {STEP_META[step_num]['name']} — {state}")
        elif state == "PENDING":
            progress_bar.progress(0.05, text="Waiting for GPU allocation...")
        elif state in ("COMPLETED", "FAILED", "CANCELLED"):
            progress_bar.progress(1.0, text=f"Job {state}")
            break

        # Update log
        iteration += 1
        log_container.text_area(
            "📋 Step Log",
            value="\n".join(log_lines[-200:]),
            height=240,
            disabled=True,
            key=f"log_s{step_num}_{job_id}_{iteration}",
        )
        time.sleep(12)

    return state


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

exp_name = st.session_state.current_exp
exp_obj = next((e for e in experiments if e["name"] == exp_name), None)
done_steps = exp_obj["done_steps"] if exp_obj else []
is_running = st.session_state.step_running > 0

st.markdown(f"### ⚡ Pipeline — `{exp_name}`")

# ── Step grid ──────────────────────────────────────────────────────────────────
rows = [(2, 3), (4, 5), (6, 7), (8, 9)]

for row_steps in rows:
    cols = st.columns([1] + [2] * len(row_steps))
    with cols[0]:
        st.markdown("")

    for i, step_num in enumerate(row_steps):
        meta = STEP_META[step_num]
        is_done = step_num in done_steps
        is_this_running = st.session_state.step_running == step_num

        with cols[i + 1]:
            bg = "#f0fdf4" if is_done else ("#fef9c3" if is_this_running else "#f9fafb")
            border = "#22c55e" if is_done else ("#eab308" if is_this_running else "#e5e7eb")
            icon = "✅" if is_done else ("⏳" if is_this_running else "⬜")

            st.markdown(f"""
            <div style="background:{bg};border:2px solid {border};border-radius:10px;
                        padding:10px;margin:4px 0">
              <div style="font-weight:700;font-size:13px;color:#374151">{icon} {meta["label"]}</div>
              <div style="font-size:11px;color:#6b7280;margin-top:2px">{meta["name"]}</div>
            </div>
            """, unsafe_allow_html=True)

            if not is_done and not is_running:
                if st.button(f"▶️ Chạy", key=f"run_{step_num}", use_container_width=True):
                    st.session_state.step_running = step_num
                    st.rerun()
            elif is_done:
                if st.button(f"🔁 Chạy lại", key=f"rerun_{step_num}", use_container_width=True):
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
    mcqs = exp_obj["mcqs"]
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
            with st.container():
                c1, c2, c3 = st.columns([3, 1, 1])
                with c1:
                    badge = "✅" if exp["mcq_count"] > 0 else "⬜"
                    st.markdown(f"{badge} **`{exp['name']}`** — {mtime}")
                with c2:
                    st.markdown(f"Steps: {done if done else '—'}")
                with c3:
                    if st.button("Chọn", key=f"sel_{exp['name']}"):
                        st.session_state.current_exp = exp["name"]
                        st.rerun()
                if exp["mcqs"]:
                    stats = stats_summary(exp["mcqs"])
                    st.markdown(render_stats_html(stats), unsafe_allow_html=True)
                st.markdown("---")
    else:
        st.info("Chưa có experiment nào.")
