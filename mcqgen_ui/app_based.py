"""
mcqgen_ui/app.py — MCQGen Simple Generator UI
=============================================
Streamlit web interface đơn giản:
  1. Chọn experiment có sẵn hoặc tạo mới
  2. Hiển thị trạng thái các bước đã chạy
  3. Chỉnh sửa topic list tạm thời (chỉ dùng trong session)
  4. Nhấn "Gen" → chạy retrieval + 03_09_pipeline.sh
  5. Hiển thị kết quả + download JSON

Chạy:
    cd CS431MCQGen/mcqgen_ui
    streamlit run app.py --server.port 8502 --server.address 0.0.0.0
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcq_renderer import render_mcq_card, stats_summary, render_stats_html

TOPIC_LIST_SOURCE = PROJECT_ROOT / "input" / "topic_list_based.json"
LOG_DIR = PROJECT_ROOT / "log"
OUTPUT_ROOT = PROJECT_ROOT / "output"

STEP_META = {
    2: {"label": "02 — Retrieval", "name": "Hybrid Retrieval (BM25+Vector+RRF)"},
    3: {"label": "03 — Gen Stem", "name": "P1: Generate Stem + Key"},
    4: {"label": "04 — Refine", "name": "P2+P3: Self-Refine"},
    5: {"label": "05 — Distractors", "name": "P4: Distractor Candidates"},
    6: {"label": "06 — CoT", "name": "P5-P8: CoT Distractor Selection"},
    7: {"label": "07 — Eval", "name": "Eval: Overall (8 criteria)"},
    8: {"label": "08 — IWF", "name": "Eval: IWF (Item Writing Flaws)"},
    9: {"label": "09 — Explain", "name": "Explanation Generation"},
}

STEP_OUTPUT_FILES = {
    2: ("02_retrieval", "*.jsonl"),
    3: ("03_gen_stem", "all_p1_results.jsonl"),
    4: ("04_gen_refine", "all_refined_results.jsonl"),
    5: ("05_gen_distractors", "all_candidates_results.jsonl"),
    6: ("06_gen_cot", "all_final_mcqs.jsonl"),
    7: ("07_eval", "evaluated_questions.jsonl"),
    8: ("08_eval_iwf", "final_accepted_questions.jsonl"),
    9: ("09_explain", "explanations.jsonl"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_topic_list_source() -> list[dict]:
    if not TOPIC_LIST_SOURCE.exists():
        return []
    with open(TOPIC_LIST_SOURCE, encoding="utf-8") as f:
        return json.load(f)


def load_explanations(exp_name: str) -> dict[str, dict]:
    path = OUTPUT_ROOT / exp_name / "09_explain" / "explanations.jsonl"
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
    enriched = []
    for mcq in mcqs:
        q_id = mcq.get("question_id", "")
        record = explanations.get(q_id, {})
        if record:
            exp_dict = record.get("explanation", {})
            mcq = dict(mcq)
            mcq["explanation"] = exp_dict
            sources_used = exp_dict.get("sources_used", []) if isinstance(exp_dict, dict) else []
            sources_flat = record.get("sources", [])
            mcq["sources"] = sources_used + sources_flat
        enriched.append(mcq)
    return enriched


def load_mcqs(exp_name: str) -> list[dict]:
    final_paths = [
        OUTPUT_ROOT / exp_name / "08_eval_iwf" / "final_accepted_questions.jsonl",
        OUTPUT_ROOT / exp_name / "08_eval_iwf" / "evaluated_questions.jsonl",
    ]
    mcqs = []
    for fp in final_paths:
        if fp.exists() and fp.stat().st_size > 0:
            with open(fp, encoding="utf-8") as f:
                mcqs = [json.loads(l) for l in f if l.strip()]
            break
    return mcqs


def validate_step_output(exp_dir: Path, step_num: int) -> tuple[bool, int]:
    if step_num not in STEP_OUTPUT_FILES:
        return True, 0
    dir_name, pattern = STEP_OUTPUT_FILES[step_num]
    step_dir = exp_dir / dir_name
    if not step_dir.exists():
        return False, 0

    if "*" in pattern:
        files = sorted(step_dir.glob(pattern))
        if not files:
            return False, 0
        total = 0
        for fp in files:
            if fp.stat().st_size == 0:
                return False, 0
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            json.loads(line)
                            total += 1
            except (json.JSONDecodeError, OSError):
                return False, total
        return total > 0, total

    fp = step_dir / pattern
    if not fp.exists() or fp.stat().st_size == 0:
        return False, 0
    count = 0
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    json.loads(line)
                    count += 1
    except (json.JSONDecodeError, OSError):
        return False, count
    return count > 0, count


def list_experiments() -> list[dict]:
    """Scan output/ for all experiments with step status."""
    experiments = []
    if not OUTPUT_ROOT.exists():
        return experiments
    for d in sorted(OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        done_steps = []
        step_counts = {}
        for sn in STEP_OUTPUT_FILES:
            is_valid, cnt = validate_step_output(d, sn)
            step_counts[sn] = cnt
            if is_valid:
                done_steps.append(sn)
        mcqs = load_mcqs(d.name)
        experiments.append({
            "name": d.name,
            "path": d,
            "done_steps": sorted(done_steps),
            "step_counts": step_counts,
            "mcq_count": len(mcqs),
            "mtime": datetime.fromtimestamp(d.stat().st_mtime),
        })
    return experiments


def poll_slurm_until_done(job_id: str, log_path: Path,
                           progress_bar, status_text, log_container) -> str:
    log_pos = 0
    log_lines = []
    iteration = 0
    unknown_count = 0  # đếm số lần liên tiếp state = UNKNOWN

    while True:
        iteration += 1
        state = "UNKNOWN"
        try:
            r = subprocess.run(
                ["squeue", "-j", job_id, "-o", "%T", "--noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                state = r.stdout.strip().upper()
                unknown_count = 0
            else:
                # Job không còn trong squeue → check sacct để lấy final state
                unknown_count += 1
                if unknown_count >= 2:
                    try:
                        sacct = subprocess.run(
                            ["sacct", "-j", job_id, "--format=State", "--noheader", "-P"],
                            capture_output=True, text=True, timeout=10,
                        )
                        if sacct.returncode == 0 and sacct.stdout.strip():
                            lines = [l.strip() for l in sacct.stdout.strip().split("\n") if l.strip()]
                            if lines:
                                state = lines[0].upper()
                                if "+" in state:
                                    state = state.split("+")[0]
                    except Exception:
                        pass
                    # Nếu sacct cũng không trả được → coi như COMPLETED
                    if state == "UNKNOWN" and unknown_count >= 3:
                        state = "COMPLETED"
        except Exception:
            pass

        icon = {"RUNNING": "🟢", "PENDING": "🟡", "COMPLETED": "✅",
                "FAILED": "❌", "CANCELLED": "🚫"}.get(state, "⚪")
        status_text.info(f"{icon} Job `{job_id}` — {state}")

        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(log_pos)
                    new_lines = f.readlines()
                    log_pos = f.tell()
                for raw in new_lines:
                    line = raw.strip()
                    if line:
                        log_lines.append(line)
            except Exception:
                pass

        if len(log_lines) > 500:
            log_lines = log_lines[-500:]

        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            progress_bar.progress(1.0, text=f"Job {state}")
            time.sleep(1)
            st.rerun()
            return state

        log_container.text_area(
            "📋 Log", value="\n".join(log_lines[-200:]),
            height=200, disabled=True, key=f"log_{job_id}_{iteration}",
        )
        time.sleep(12)

# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("exp_name", ""),
    ("topic_data", None),
    ("topic_edited", None),
    ("generation_phase", None),
    ("pipeline_job_id", None),
    ("generation_done", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MCQGen Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("### 📝 **MCQGen — Simple Generator**")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Experiment Selector (chọn có sẵn hoặc tạo mới)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("#### 1️⃣ Chọn hoặc tạo Experiment")

experiments = list_experiments()
exp_options = [e["name"] for e in experiments]

sel_col1, sel_col2, sel_col3 = st.columns([2, 2, 1])

with sel_col1:
    if exp_options:
        selected_exp = st.selectbox(
            "Chọn experiment có sẵn",
            options=["— Tạo mới —"] + exp_options,
            index=0 if not st.session_state.exp_name else
                  (exp_options.index(st.session_state.exp_name) + 1
                   if st.session_state.exp_name in exp_options else 0),
            label_visibility="collapsed",
        )
    else:
        selected_exp = "— Tạo mới —"
        st.info("Chưa có experiment nào. Nhập tên bên dưới để tạo mới.")

with sel_col2:
    new_exp_name = st.text_input(
        "Tên experiment mới",
        placeholder="ví dụ: exp_demo_v1",
        label_visibility="collapsed",
    )

with sel_col3:
    st.markdown("")
    if st.button("✨ Áp dụng", use_container_width=True):
        if selected_exp != "— Tạo mới —" and not new_exp_name.strip():
            # Chọn experiment có sẵn
            st.session_state.exp_name = selected_exp
            st.session_state.topic_data = load_topic_list_source()
            st.session_state.topic_edited = None
            st.session_state.generation_phase = None
            st.session_state.generation_done = False
            st.session_state.pipeline_job_id = None
            st.rerun()
        elif new_exp_name.strip():
            # Tạo mới
            name = new_exp_name.strip()
            if " " in name or "/" in name:
                st.warning("⚠️ Tên không được chứa dấu cách hoặc `/`.")
            else:
                st.session_state.exp_name = name
                st.session_state.topic_data = load_topic_list_source()
                st.session_state.topic_edited = None
                st.session_state.generation_phase = None
                st.session_state.generation_done = False
                st.session_state.pipeline_job_id = None
                st.rerun()
        else:
            st.warning("⚠️ Chọn experiment hoặc nhập tên mới.")

# ── Hiển thị trạng thái experiment đã chọn ─────────────────────────────────────
exp_name = st.session_state.exp_name
if exp_name:
    exp_dir = OUTPUT_ROOT / exp_name
    exp_obj = next((e for e in experiments if e["name"] == exp_name), None)

    if exp_obj:
        done = exp_obj["done_steps"]
        mcq_count = exp_obj["mcq_count"]
        mtime = exp_obj["mtime"].strftime("%d/%m/%Y %H:%M")
        st.success(f"📂 **`{exp_name}`** — Cập nhật: {mtime} — MCQs: {mcq_count}")
    else:
        st.success(f"📂 **`{exp_name}`** — Experiment mới")
        done = []

    # ── Step status grid ──────────────────────────────────────────────────────
    rows = [(2, 3, 4, 5), (6, 7, 8, 9)]
    for row_steps in rows:
        cols = st.columns(len(row_steps))
        for i, sn in enumerate(row_steps):
            meta = STEP_META[sn]
            is_valid, cnt = validate_step_output(exp_dir, sn)
            with cols[i]:
                if is_valid:
                    bg, border, icon = "#f0fdf4", "#22c55e", "✅"
                    status = f"Done — {cnt} items" if cnt else "Done"
                elif (exp_dir / STEP_OUTPUT_FILES[sn][0]).exists():
                    bg, border, icon = "#fffbeb", "#f59e0b", "⚠️"
                    status = "Incomplete"
                else:
                    bg, border, icon = "#f9fafb", "#e5e7eb", "⬜"
                    status = "Chưa chạy"
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {border};border-radius:10px;
                            padding:8px;margin:2px 0;text-align:center">
                  <div style="font-weight:700;font-size:12px;color:#374151">{icon} {meta["label"]}</div>
                  <div style="font-size:10px;color:#6b7280">{meta["name"]}</div>
                  <div style="font-size:9px;color:#9ca3af">{status}</div>
                </div>
                """, unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Topic List Editor
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.topic_data is None:
    st.info("Chọn hoặc tạo experiment bên trên để bắt đầu.")
else:
    with st.expander("#### 2️⃣ Chỉnh sửa Topic List (tạm thời)", expanded=False):
        current_data = st.session_state.topic_edited if st.session_state.topic_edited is not None else st.session_state.topic_data
        topic_json = json.dumps(current_data, ensure_ascii=False, indent=2)

        edited_json = st.text_area(
            "topic_list.json (JSON)",
            value=topic_json,
            height=350,
            label_visibility="collapsed",
        )

        tl_col1, tl_col2, tl_col3 = st.columns([1, 1, 1])
        with tl_col1:
            if st.button("💾 Áp dụng", use_container_width=True):
                try:
                    parsed = json.loads(edited_json)
                    total_topics = sum(len(ch.get("topics", [])) for ch in parsed)
                    st.session_state.topic_edited = parsed
                    st.session_state.topic_data = parsed
                    st.success(f"✅ {len(parsed)} chapters, {total_topics} topics")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON lỗi: {e}")
        with tl_col2:
            if st.button("🔍 Kiểm tra", use_container_width=True):
                try:
                    parsed = json.loads(edited_json)
                    total_topics = sum(len(ch.get("topics", [])) for ch in parsed)
                    st.success(f"✅ OK — {len(parsed)} chapters, {total_topics} topics")
                except json.JSONDecodeError as e:
                    st.error(f"❌ JSON lỗi: {e}")
        with tl_col3:
            if st.button("↩️ Khôi phục gốc", use_container_width=True):
                st.session_state.topic_edited = None
                st.session_state.topic_data = load_topic_list_source()
                st.rerun()

        total_topics = sum(len(ch.get("topics", [])) for ch in current_data)
        topic_ids = [t["topic_id"] for ch in current_data for t in ch.get("topics", [])]
        st.caption(f"📋 {len(current_data)} chapters · {total_topics} topics: `{', '.join(topic_ids[:10])}"
                   + ("..." if len(topic_ids) > 10 else ""))

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 3: Generate Button
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("#### 3️⃣ Sinh câu hỏi")

    gen_col1, gen_col2, gen_col3 = st.columns([1, 1, 4])

    with gen_col1:
        gen_pressed = st.button(
            "🚀 Generate MCQ", use_container_width=True,
            disabled=st.session_state.generation_phase is not None or not st.session_state.topic_data,
        )

    with gen_col2:
        if st.session_state.generation_phase == "pipeline" and st.session_state.pipeline_job_id:
            if st.button("⛔ Dừng Job", use_container_width=True):
                subprocess.run(["scancel", st.session_state.pipeline_job_id], capture_output=True)
                st.session_state.generation_phase = None
                st.session_state.pipeline_job_id = None
                st.rerun()

    with gen_col3:
        phase = st.session_state.generation_phase
        if phase == "pipeline":
            st.info(f"⏳ Đang chạy Pipeline 03-09... Job ID: `{st.session_state.pipeline_job_id}`")
        elif phase == "done":
            st.success("✅ Generation hoàn tất!")
        else:
            st.caption("Nhấn Generate để chạy retrieval → 03-09 pipeline")

    # ── Execute generation on button press ───────────────────────────────────
    if gen_pressed and st.session_state.generation_phase is None:
        exp_name = st.session_state.exp_name
        topic_data = st.session_state.topic_edited or st.session_state.topic_data

        if not exp_name:
            st.error("⚠️ Chưa có experiment name.")
        elif not topic_data:
            st.error("⚠️ Chưa có topic data.")
        else:
            # ── Save topic list ───────────────────────────────────────────────
            exp_path = OUTPUT_ROOT / exp_name
            exp_path.mkdir(parents=True, exist_ok=True)
            with open(exp_path / "topic_list.json", "w", encoding="utf-8") as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=2)
            input_topic_list = PROJECT_ROOT / "input" / "topic_list.json"
            with open(input_topic_list, "w", encoding="utf-8") as f:
                json.dump(topic_data, f, ensure_ascii=False, indent=2)

            # ── Update src/common.py EXP_NAME ──────────────────────────────────
            common_py = PROJECT_ROOT / "src" / "common.py"
            with open(common_py, encoding="utf-8") as f:
                content = f.read()
            import re as _re
            new_content = _re.sub(
                r'(class Config:\s+#[^\n]*\n\s+EXP_NAME\s*=\s*")[^"]*(")',
                rf'\g<1>{exp_name}\g<2>',
                content,
            )
            with open(common_py, "w", encoding="utf-8") as f:
                f.write(new_content)

            LOG_DIR.mkdir(parents=True, exist_ok=True)

            # ── Phase 1: Chạy retrieval trực tiếp (CPU) ─────────────────────
            st.info("🔄 Đang chạy Step 02: Hybrid Retrieval...")
            retrieval_env = {
                **subprocess.os.environ,
                "EXP_NAME": exp_name,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(PROJECT_ROOT),
            }
            retrieval_ok = False
            try:
                ret_result = subprocess.run(
                    ["python", "-u", str(PROJECT_ROOT / "src" / "gen" / "retrieval.py")],
                    capture_output=True, text=True,
                    cwd=str(PROJECT_ROOT),
                    env=retrieval_env,
                    timeout=600,
                )
                if ret_result.returncode == 0:
                    st.success("✅ Step 02 Retrieval hoàn tất!")
                    retrieval_ok = True
                else:
                    st.error(f"❌ Retrieval failed:\n```\n{ret_result.stderr[-500:]}\n```")
            except subprocess.TimeoutExpired:
                st.error("❌ Retrieval timeout (>10 phút)")
            except Exception as e:
                st.error(f"❌ Retrieval error: {e}")

            if not retrieval_ok:
                st.stop()

            # ── Phase 2: Submit 03_09_pipeline ──────────────────────────────
            pipeline_script = PROJECT_ROOT / "scripts" / "03_09_pipeline.sh"
            if not pipeline_script.exists():
                st.error(f"❌ Script không tìm thấy: `{pipeline_script}`")
            else:
                try:
                    log_out = LOG_DIR / f"03_09_{exp_name}_%j.out"
                    log_err = LOG_DIR / f"03_09_{exp_name}_%j.err"
                    result = subprocess.run(
                        ["sbatch", "--parsable",
                         "--output", str(log_out),
                         "--error", str(log_err),
                         str(pipeline_script)],
                        capture_output=True, text=True, timeout=30,
                        env={**subprocess.os.environ, "EXP_NAME": exp_name},
                    )
                    if result.returncode == 0:
                        job_id = result.stdout.strip().split(";")[0].strip()
                        st.session_state.generation_phase = "pipeline"
                        st.session_state.pipeline_job_id = job_id
                        st.success(f"✅ Pipeline submitted: Job `{job_id}`")
                        st.rerun()
                    else:
                        st.error(f"❌ sbatch pipeline failed: {result.stderr}")
                except subprocess.TimeoutExpired:
                    st.error("❌ sbatch pipeline timed out")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    # ── Poll pipeline job ────────────────────────────────────────────────────
    if st.session_state.generation_phase == "pipeline" and st.session_state.pipeline_job_id:
        job_id = st.session_state.pipeline_job_id
        exp_name = st.session_state.exp_name

        progress_bar = st.progress(0.5, text="Pipeline 03-09...")
        status_text = st.empty()
        log_container = st.empty()

        job_log_files = sorted(LOG_DIR.glob(f"03_09_{exp_name}_*.out"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        log_path = job_log_files[0] if job_log_files else LOG_DIR / f"03_09_{exp_name}_%j.out"

        state = poll_slurm_until_done(job_id, log_path, progress_bar, status_text, log_container)

        if state == "COMPLETED":
            st.session_state.generation_phase = "done"
            st.session_state.generation_done = True
            st.success("✅ Generation hoàn tất!")
        else:
            st.error(f"❌ Pipeline job {state}: `{job_id}`")
            st.session_state.generation_phase = None
        st.rerun()

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 4: Results + Download
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("#### 4️⃣ Kết quả")

    exp_name = st.session_state.exp_name
    if not exp_name:
        st.info("Chưa chọn experiment.")
    else:
        exp_dir = OUTPUT_ROOT / exp_name
        mcqs = load_mcqs(exp_name)

        if mcqs:
            explanations = load_explanations(exp_name)
            mcqs_enriched = merge_mcqs_with_explanations(mcqs, explanations)
            has_explanations = bool(explanations)

            stats = stats_summary(mcqs_enriched)
            st.markdown(render_stats_html(stats), unsafe_allow_html=True)

            # ── Download buttons ─────────────────────────────────────────────
            dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 4])
            with dl_col1:
                # MCQ answers JSON
                mcq_json_str = json.dumps(mcqs, ensure_ascii=False, indent=2)
                st.download_button(
                    label="📥 Download MCQs (JSON)",
                    data=mcq_json_str,
                    file_name=f"{exp_name}_mcqs.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with dl_col2:
                if has_explanations:
                    # Explanations JSON
                    explain_list = list(explanations.values())
                    explain_json_str = json.dumps(explain_list, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Download Explanations (JSON)",
                        data=explain_json_str,
                        file_name=f"{exp_name}_explanations.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                else:
                    st.caption("Chưa có explanations (Step 09)")
            with dl_col3:
                if has_explanations:
                    # Combined: MCQs + explanations merged
                    combined_json_str = json.dumps(mcqs_enriched, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="📥 Download All (MCQs + Explanations)",
                        data=combined_json_str,
                        file_name=f"{exp_name}_full.json",
                        mime="application/json",
                    )

            st.markdown("---")

            # ── Filters ──────────────────────────────────────────────────────
            f1, f2, f3 = st.columns(3)
            with f1:
                type_filter = st.selectbox("Loại", ["Tất cả", "single_correct", "multiple_correct"])
            with f2:
                diff_filter = st.selectbox("Độ khó", ["Tất cả", "G1", "G2", "G3"])
            with f3:
                show_eval = st.checkbox("Hiện chi tiết eval")

            filtered = mcqs_enriched
            if type_filter != "Tất cả":
                filtered = [m for m in filtered if m.get("question_type") == type_filter]
            if diff_filter != "Tất cả":
                filtered = [m for m in filtered
                           if m.get("difficulty_label", m.get("difficulty", "")) == diff_filter]

            st.markdown(f"**{len(filtered)} / {len(mcqs_enriched)}** câu hỏi")

            # ── Pagination ───────────────────────────────────────────────────
            page_size = 5
            n_pages = max(1, (len(filtered) + page_size - 1) // page_size)
            page = st.number_input("Trang", min_value=1, max_value=n_pages, value=1, key="result_page")
            start_i = (page - 1) * page_size

            for mcq in filtered[start_i:start_i + page_size]:
                if show_eval:
                    with st.expander(f"📋 `{mcq.get('question_id', '?')}`"):
                        st.json(mcq, expanded=False)
                else:
                    st.markdown(render_mcq_card(mcq), unsafe_allow_html=True)

                    # ── Inline explanation (nếu có) ──────────────────────────
                    exp_data = mcq.get("explanation", {})
                    if isinstance(exp_data, dict) and exp_data:
                        with st.expander(f"💡 Giải thích — `{mcq.get('question_id', '?')}`"):
                            # Motivation
                            motivation = exp_data.get("question_motivation", "")
                            if motivation:
                                st.markdown(f"**Lý do đặt câu hỏi:** {motivation}")

                            # Correct answer rationale
                            rationale = exp_data.get("correct_answer_rationale", "")
                            if rationale:
                                st.markdown(f"**Đáp án đúng:** {rationale}")

                            # Distractor explanations
                            dist_expl = exp_data.get("distractor_explanations", {})
                            if dist_expl:
                                st.markdown("**Giải thích đáp án sai:**")
                                for label, expl in sorted(dist_expl.items()):
                                    st.markdown(f"- **{label}:** {expl}")

                            # Sources
                            sources = mcq.get("sources", [])
                            if sources:
                                st.markdown("**Nguồn tham khảo:**")
                                for src in sources:
                                    if isinstance(src, dict):
                                        src_type = src.get("type", "")
                                        url = src.get("url", "")
                                        desc = src.get("description", src.get("title", ""))
                                        if url:
                                            st.markdown(f"- [{src_type}] [{desc}]({url})")
                                        elif desc:
                                            st.markdown(f"- [{src_type}] {desc}")
        else:
            st.info("Chưa có kết quả — chạy generation bên trên để tạo dữ liệu.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Section 5: Experiment History
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("---")
    with st.expander("📁 Lịch sử Experiments", expanded=False):
        if experiments:
            for exp in experiments[:15]:
                done = sorted(exp["done_steps"])
                mtime = exp["mtime"].strftime("%d/%m/%Y %H:%M")
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    badge = "✅" if exp["mcq_count"] > 0 else "⬜"
                    st.markdown(f"{badge} **`{exp['name']}`** — {mtime} — MCQs: {exp['mcq_count']}")
                with c2:
                    done_labels = [f"S{s}" for s in done]
                    st.caption(f"Steps: {', '.join(done_labels) if done_labels else '—'}")
                with c3:
                    if st.button("Chọn", key=f"sel_{exp['name']}"):
                        st.session_state.exp_name = exp["name"]
                        st.session_state.topic_data = load_topic_list_source()
                        st.session_state.topic_edited = None
                        st.session_state.generation_phase = None
                        st.session_state.generation_done = False
                        st.session_state.pipeline_job_id = None
                        st.rerun()
                st.markdown("---")
        else:
            st.info("Chưa có experiment nào.")