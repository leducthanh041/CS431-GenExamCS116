"""
web/app.py — Streamlit Web Interface for MCQGen Pipeline
========================================================
Cho phép người dùng:
  1. Nhập prompt tự do → hệ thống parse và xác định chapters/topics cần gen
  2. Điều chỉnh số câu hỏi, độ khó, tỉ lệ single/multiple
  3. Chạy pipeline và xem kết quả
  4. Xem explanation với trích dẫn (YouTube timestamp, slide, web)

Chạy:
  cd CS431MCQGen/web
  streamlit run app.py --server.port 8501

hoặc:
  streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

# ── Project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gen.prompt_parser import (
    parse_user_prompt, merge_with_base_config, parse_by_keywords
)
from src.gen.prompt_config import load_generation_config, compute_batch_context

# ── Adaptive engine imports ────────────────────────────────────────────────────
import src.adaptive.db as adaptive_db
from src.adaptive import (
    load_profile,
    init_profile,
    get_study_plan,
    detect_weak_topics,
    generate_adaptive_quiz,
    create_quiz_session,
    submit_answer,
    grade_quiz,
    end_session,
    get_session,
    get_topic_accuracy,
    get_chapter_accuracy,
    get_overall_accuracy,
    load_topic_list,
    get_pool_coverage,
    get_coverage_report,
    generate_on_demand,
    find_missing_for_topics,
    refresh_pool,
)


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CS431MCQGen — Sinh đề thi MCQ",
    page_icon="📝",
    layout="wide",
)


# ── Constants ───────────────────────────────────────────────────────────────

CHAPTER_INFO = {
    "ch02": ("Popular Libraries", "NumPy, Pandas, Matplotlib, Scikit-learn"),
    "ch03": ("ML Pipeline & EDA", "Machine Learning Pipeline, Exploratory Data Analysis"),
    "ch04": ("Tiền xử lý dữ liệu", "Missing Data, Outlier Detection, Feature Extraction/Transformation/Selection"),
    "ch05": ("Đánh giá mô hình", "Classification Metrics, Regression Metrics, Cross-validation"),
    "ch06": ("Unsupervised Learning", "Clustering, Dimensionality Reduction"),
    "ch07a": ("Supervised Learning - Regression", "Linear Regression, Regularization (Ridge/Lasso/Elastic Net)"),
    "ch07b": ("Supervised Learning - Classification", "Logistic Regression, Decision Trees, SVM"),
    "ch08": ("Deep Learning với CNN", "Neural Networks, CNN (Pooling, Filter, Transfer Learning)"),
    "ch09": ("Parameter Tuning", "Grid Search, Random Search, Bayesian Optimization"),
    "ch10": ("Ensemble Models", "Bagging, Boosting (AdaBoost/GBM/XGBoost), Random Forest"),
    "ch11": ("Model Deployment", "Model Serving (Flask/FastAPI/Docker), Model Monitoring & Drift Detection"),
}

DIFFICULTY_DISPLAY = {
    "G1": "G1 – Nhớ/Hiểu",
    "G2": "G2 – Áp dụng/Phân tích",
    "G3": "G3 – Đánh giá/Sáng tạo",
}


# ── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "config": None,
        "parsed_prompt": None,
        "results": None,
        "explanations": None,
        "pipeline_status": None,
        "logs": [],
        # ── Adaptive quiz state ────────────────────────────────────────────────
        "quiz_session": None,
        "quiz_questions": [],
        "quiz_current": 0,
        "quiz_user_id": "student_demo",
        "quiz_answers": {},
        "quiz_started": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Sidebar: Config ─────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Cấu hình")
        st.markdown("---")

        # Load base config
        try:
            base_cfg = load_generation_config()
            st.success("✅ Đã load generation_config.yaml")
        except Exception as e:
            st.error(f"❌ Lỗi load config: {e}")
            base_cfg = {}

        st.markdown("---")

        # Override settings
        st.subheader("Tuỳ chỉnh")
        target_min = st.number_input(
            "Số câu tối thiểu", min_value=1, max_value=100, value=25, step=5,
            help="Số câu hỏi tối thiểu cần gen"
        )
        target_max = st.number_input(
            "Số câu tối đa", min_value=1, max_value=100, value=35, step=5,
            help="Số câu hỏi tối đa cần gen (dự phòng judge reject)"
        )

        single_ratio = st.slider(
            "Tỉ lệ câu 1 đáp án", min_value=0.5, max_value=1.0, value=0.8, step=0.05,
            help="80% = 5-7/30 câu multiple answer. 100% = toàn bộ single answer."
        )

        st.markdown("---")
        st.subheader("📚 Chapters trọng điểm")
        default_chapters = base_cfg.get("generation", {}).get("focus_chapters", [])
        selected_chapters = st.multiselect(
            "Chọn chapters cần tập trung",
            options=list(CHAPTER_INFO.keys()),
            default=default_chapters,
            format_func=lambda x: f"{x}: {CHAPTER_INFO[x][0]}",
        )

        if selected_chapters:
            weights = {}
            st.markdown("**Trọng số:**")
            for ch in selected_chapters:
                w = st.slider(f"  {ch} — {CHAPTER_INFO[ch][0]}", 0.5, 3.0, 2.0, 0.5, key=f"w_{ch}")
                weights[ch] = w
        else:
            weights = {}

        return {
            "target_range": [target_min, target_max],
            "single_correct_ratio": single_ratio,
            "focus_chapters": selected_chapters,
            "topic_weights": weights,
        }, base_cfg


# ── Prompt parsing section ─────────────────────────────────────────────────

def render_prompt_input(base_cfg: dict):
    st.subheader("📝 Nhập yêu cầu bằng ngôn ngữ tự nhiên")

    with st.expander("💡 Ví dụ prompts", expanded=False):
        examples = [
            "Tôi muốn ôn tập chương 7b và chương 8 về classification và CNN",
            "Cho tôi 50 câu hỏi tập trung vào ensemble models và hyperparameter tuning",
            "Tôi cần 20 câu hỏi G3 (khó) về deep learning và neural networks",
            "Ôn tập toàn bộ môn, ưu tiên chương 4 và chương 10",
        ]
        for ex in examples:
            st.markdown(f"- *{ex}*")

    prompt_text = st.text_area(
        "Prompt của bạn",
        placeholder="Ví dụ: Tôi muốn ôn tập chương 7b và chương 8 về classification và CNN...",
        height=80,
        help="Nhập yêu cầu bằng tiếng Việt hoặc tiếng Anh. Hệ thống sẽ tự động parse.",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        parse_btn = st.button("🔍 Parse Prompt", type="secondary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", type="secondary", use_container_width=True)

    parsed_result = None
    if parse_btn and prompt_text:
        with st.spinner("Đang parse prompt..."):
            parsed_result = parse_user_prompt(prompt_text, use_llm=True)

        if parsed_result:
            st.session_state["parsed_prompt"] = parsed_result

            cols = st.columns(3)
            with cols[0]:
                fc = parsed_result.get("focus_chapters", [])
                st.success(f"📚 Chapters: {', '.join(fc) if fc else 'Tất cả'}")
            with cols[1]:
                tr = parsed_result.get("target_range", [25, 35])
                st.info(f"📊 Số câu: {tr[0]}-{tr[1]}")
            with cols[2]:
                method = parsed_result.get("parse_method", "keyword")
                st.info(f"🔧 Parse: {method}")

            # Show parsed weights
            tw = parsed_result.get("topic_weights", {})
            if tw:
                st.markdown("**Trọng số topics:**")
                for ch, w in sorted(tw.items()):
                    st.markdown(f"  - `{ch}`: ×{w}")
        else:
            st.warning("Không parse được prompt. Dùng keyword fallback.")

    if clear_btn:
        st.session_state["parsed_prompt"] = None
        st.rerun()

    return parsed_result


# ── Pipeline overview ───────────────────────────────────────────────────────

def render_pipeline_steps():
    steps = [
        ("01b", "Chunk Transcripts", "Whisper JSON → timestamped chunks"),
        ("01", "Indexing", "Slide + Transcript → ChromaDB"),
        ("02", "Hybrid Retrieval", "BM25 + Vector + RRF + Rerank"),
        ("03", "P1: Stem + Key", "Qwen2.5-14B → stems"),
        ("04", "P2+P3: Refine", "Self-refine stems"),
        ("05", "P4: Distractors", "Generate 6 candidates"),
        ("06", "P5-P8: CoT", "Evaluate → Remove → Select → Assemble"),
        ("07", "Eval Overall", "Gemma-3-12b: 8-criteria check"),
        ("08", "Eval IWF", "Gemma-3-12b: Item Writing Flaws"),
        ("09", "Explanation", "Gen explanations + web search"),
    ]

    st.subheader("🔄 Pipeline (9 bước)")
    for num, name, desc in steps:
        col1, col2, col3 = st.columns([1, 3, 4])
        with col1:
            st.markdown(f"**{num}**")
        with col2:
            st.markdown(name)
        with col3:
            st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────

def build_final_config(manual_cfg: dict, parsed: dict | None, base_cfg: dict) -> dict:
    """Merge manual settings + parsed prompt + base config into final config."""
    gen = base_cfg.get("generation", {}).copy()

    # Manual overrides
    gen["target_range"] = manual_cfg["target_range"]
    gen["single_correct_ratio"] = manual_cfg["single_correct_ratio"]
    if manual_cfg["focus_chapters"]:
        gen["focus_chapters"] = manual_cfg["focus_chapters"]
    if manual_cfg["topic_weights"]:
        gen["topic_weights"] = manual_cfg["topic_weights"]

    # Parsed prompt overrides
    if parsed:
        if parsed.get("focus_chapters"):
            gen["focus_chapters"] = parsed["focus_chapters"]
        if parsed.get("target_range"):
            gen["target_range"] = parsed["target_range"]
        if parsed.get("single_correct_ratio"):
            gen["single_correct_ratio"] = parsed["single_correct_ratio"]
        if parsed.get("topic_weights"):
            existing = gen.get("topic_weights", {})
            for ch, w in parsed["topic_weights"].items():
                existing[ch] = w
            gen["topic_weights"] = existing

    result = dict(base_cfg)
    result["generation"] = gen
    return result


def write_config_to_yaml(cfg: dict) -> Path:
    """Write final config to a temp YAML for pipeline to read."""
    import yaml
    config_path = PROJECT_ROOT / "configs" / "generation_config_active.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    return config_path


def run_pipeline(config_path: Path, steps: list[str]):
    """Run SLURM pipeline for specified steps."""
    script_path = PROJECT_ROOT / "scripts" / "00_pipeline.sh"

    cmd = [
        "sbatch",
        "--wait",
        "--output", str(PROJECT_ROOT / f"log/web_pipeline_%j.out"),
        "--error", str(PROJECT_ROOT / f"log/web_pipeline_%j.err"),
        str(script_path),
    ]

    with st.spinner("Đang chạy pipeline..."):
        result = subprocess.run(cmd, capture_output=True, text=True)

    return result


def render_pipeline_control(final_cfg: dict, parsed: dict | None):
    st.markdown("---")
    st.subheader("🚀 Chạy Pipeline")

    # Preview config
    with st.expander("📋 Preview cấu hình sẽ dùng", expanded=False):
        gen = final_cfg.get("generation", {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Target range", f"{gen.get('target_range', [25,35])}")
        with col2:
            st.metric("Single ratio", f"{gen.get('single_correct_ratio', 0.8):.0%}")
        with col3:
            fc = gen.get("focus_chapters", [])
            st.metric("Focus chapters", f"{len(fc)} chapters" if fc else "Tất cả")

        # Show chapter distribution
        from src.gen.prompt_config import distribute_questions
        total = gen.get("target_range", [25, 35])[1]
        dist = distribute_questions(total, config=final_cfg)
        st.markdown("**Phân bổ câu hỏi:**")
        for ch in sorted(dist.keys()):
            st.markdown(f"  `{ch}`: {dist[ch]} câu")

    # Batch context preview
    ctx = compute_batch_context(
        num_questions=gen.get("target_range", [25, 35])[1],
        config=final_cfg,
    )
    with st.expander("📊 Preview tỉ lệ single/multiple", expanded=False):
        st.markdown(
            f"- **Single correct:** {ctx['num_single']} ({ctx['ratio_single']:.0%}) "
            f"→ {ctx['num_multi']} multiple ({ctx['ratio_multi']:.0%})"
            f"\n- Trong multi: {ctx['num_two']} câu 2 đáp án, {ctx['num_three']} câu 3 đáp án"
        )

    # Run buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        full_run = st.button(
            "🚀 Chạy FULL pipeline (Steps 01-09)",
            type="primary",
            use_container_width=True,
            help="Chạy tất cả 9 bước từ indexing → explanation",
        )
    with col2:
        quick_run = st.button(
            "⚡ Chỉ chạy gen + eval (Steps 03-08)",
            type="secondary",
            use_container_width=True,
            help="Bỏ qua indexing và retrieval (đã có data)",
        )

    if full_run or quick_run:
        config_path = write_config_to_yaml(final_cfg)
        st.info(f"✅ Config ghi vào: {config_path}")

        # Update EXP_NAME in common.py to include timestamp
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        st.info(f"Pipeline sẽ chạy với experiment name: `exp_{ts}`. "
                 "Cập nhật EXP_NAME trong common.py trước khi chạy!")

        with st.expander("⚠️ Lưu ý trước khi chạy", expanded=True):
            st.markdown("""
            **Yêu cầu:**
            1. Đảm bảo `EXP_NAME` trong `src/common.py` đã được cập nhật
            2. Đảm bảo đã chạy `01b_transcribe.sh` và `01_index.sh` cho 6 video mới
            3. GPU cần trống đủ (~40GB VRAM)
            4. Mạng cần truy cập được DuckDuckGo để web search (explanation)
            """)

        # Actually run the pipeline
        with st.spinner("Đang submit SLURM job... (có thể mất 2-6 giờ)"):
            result = run_pipeline(config_path, steps=["01-09"] if full_run else ["03-08"])

        if result.returncode == 0:
            st.success(f"✅ Pipeline submitted thành công!")
            st.code(result.stdout)
        else:
            st.error(f"❌ Pipeline error: {result.stderr}")


# ── Results viewer ────────────────────────────────────────────────────────────

def render_results():
    st.markdown("---")
    st.subheader("📋 Kết quả")

    output_base = PROJECT_ROOT / "output"
    exp_dirs = sorted(output_base.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not exp_dirs:
        st.info("Chưa có kết quả. Chạy pipeline trước.")
        return

    selected_exp = st.selectbox(
        "Chọn experiment",
        options=[d.name for d in exp_dirs],
        index=0,
    )

    exp_dir = output_base / selected_exp

    # Accepted questions
    accepted_file = exp_dir / "08_eval_iwf" / "final_accepted_questions.jsonl"
    if accepted_file.exists():
        questions = []
        with open(accepted_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

        st.success(f"✅ {len(questions)} câu hỏi đã accept")

        # Stats
        g1 = sum(1 for q in questions if q.get("difficulty_label") == "G1")
        g2 = sum(1 for q in questions if q.get("difficulty_label") == "G2")
        g3 = sum(1 for q in questions if q.get("difficulty_label") == "G3")
        multi = sum(1 for q in questions if q.get("question_type") == "multiple_correct")

        cols = st.columns(4)
        cols[0].metric("Tổng", len(questions))
        cols[1].metric("G1/G2/G3", f"{g1}/{g2}/{g3}")
        cols[2].metric("Multiple", multi)
        cols[3].metric("Single", len(questions) - multi)

        # Preview
        st.markdown("**Preview 3 câu đầu:**")
        for q in questions[:3]:
            with st.expander(f"📌 {q.get('question_id', '?')}: {q.get('question_text','')[:80]}..."):
                st.markdown(f"**Type:** {q.get('question_type')}")
                st.markdown(f"**Difficulty:** {q.get('difficulty_label')}")
                opts = q.get("options", {})
                for letter, text in opts.items():
                    st.markdown(f"  **{letter}.** {text}")
                correct = q.get("correct_answers", [])
                st.markdown(f"**Đáp án đúng:** {', '.join(correct)}")
    else:
        st.info(f"Chưa có file accepted questions: {accepted_file}")


# ── On-demand config writer ──────────────────────────────────────────────────

def write_on_demand_config_from_missing(missing_topics):
    """Write on_demand_generation.yaml from MissingTopic list."""
    import yaml
    root = PROJECT_ROOT

    topic_ids = [m.topic_id for m in missing_topics]
    topic_weights = {m.topic_id: 2.0 for m in missing_topics}
    total = sum(m.needed_count for m in missing_topics)

    cfg = {
        "_generated_by": "adaptive_on_demand",
        "_generated_at": adaptive_db.now_iso(),
        "generation": {
            "target_range": [max(1, total - 2), total + 5],
            "single_correct_ratio": 0.6,
            "focus_topics": topic_ids,
            "topic_weights": topic_weights,
            "focus_chapters": list(set(m.chapter_id for m in missing_topics)),
        },
        "_missing_topics": [
            {"topic_id": m.topic_id, "topic_name": m.topic_name,
             "chapter_id": m.chapter_id, "current": m.current_count,
             "needed": m.needed_count, "difficulty": m.needed_difficulty}
            for m in missing_topics
        ],
    }

    config_path = root / "configs" / "on_demand_generation.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    return config_path


# ── Pool Coverage Panel ────────────────────────────────────────────────────────

def render_pool_coverage():
    st.subheader("📦 MCQ Pool Coverage")

    with st.spinner("Đang phân tích pool..."):
        report = get_coverage_report()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Tổng câu", report["pool_total"])
    cov_pct = report["coverage_pct"]
    col2.metric("📚 Topics covered", f"{report['topics_covered']}/{report['total_topics']}")
    col3.metric("🏆 Coverage", f"{cov_pct:.0f}%")
    col4.metric("📖 Chapters", f"{report['chapters_covered']}/{report['total_chapters']}")

    # Difficulty distribution
    st.markdown("### 📊 Phân bổ độ khó")
    diff_dist = report.get("difficulty_distribution", {})
    if diff_dist:
        g1 = diff_dist.get("G1", {})
        g2 = diff_dist.get("G2", {})
        g3 = diff_dist.get("G3", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 G1 – Nhớ/Hiểu", g1.get("count", 0), f"{g1.get('ratio',0)*100:.0f}%")
        c2.metric("🟡 G2 – Áp dụng/Phân tích", g2.get("count", 0), f"{g2.get('ratio',0)*100:.0f}%")
        c3.metric("🔴 G3 – Đánh giá/Sáng tạo", g3.get("count", 0), f"{g3.get('ratio',0)*100:.0f}%")

    # Type distribution
    type_dist = report.get("type_distribution", {})
    if type_dist:
        single = type_dist.get("single_correct", 0)
        multi = type_dist.get("multiple_correct", 0)
        s1, s2 = st.columns(2)
        s1.metric("✅ Single-correct", single)
        s2.metric("☑️ Multiple-correct", multi)

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        st.markdown("### 🔍 Recommendations")
        for rec in recs:
            if "⚠️" in rec or "🔴" in rec:
                st.error(rec)
            elif "🟡" in rec:
                st.warning(rec)
            else:
                st.info(rec)

    # Missing topics
    missing = report.get("missing_topics", [])
    if missing:
        st.markdown(f"### ❌ Topics chưa có câu hỏi ({len(missing)} topics)")
        topic_list = load_topic_list()
        topic_names = {}
        for ch in topic_list:
            for t in ch.get("topics", []):
                topic_names[t.get("topic_id", "")] = {
                    "name": t.get("topic_name", ""),
                    "chapter": ch.get("chapter_name", ""),
                }
        rows = [missing[i:i+3] for i in range(0, len(missing), 3)]
        for row in rows:
            cols = st.columns(3)
            for col, tid in zip(cols, row):
                info = topic_names.get(tid, {})
                col.markdown(f"❌ `{tid}`\n{info.get('name', '')}")
    else:
        st.success("✅ Tất cả topics đã có câu hỏi trong pool!")

    # Per-topic coverage table
    st.markdown("### 📋 Chi tiết theo topic")
    stats = get_pool_coverage()
    topic_list = load_topic_list()
    rows_data = []
    for ch in topic_list:
        for t in ch.get("topics", []):
            tid = t.get("topic_id", "")
            count = stats.by_topic.get(tid, 0)
            rows_data.append({
                "topic_id": tid,
                "topic_name": t.get("topic_name", ""),
                "chapter_id": ch.get("chapter_id", ""),
                "chapter_name": ch.get("chapter_name", ""),
                "count": count,
                "status": "✅" if count > 0 else "❌",
            })
    if rows_data:
        import pandas as pd
        df = pd.DataFrame(rows_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # On-demand generation
    st.markdown("---")
    st.subheader("⚡ On-Demand Generation")
    user_id_od = st.text_input(
        "👤 User ID (để lấy weak topics)",
        value=st.session_state.get("quiz_user_id", "student_demo"),
        key="od_uid",
    )

    weak_ids = []
    if user_id_od:
        weak_list = detect_weak_topics(user_id_od)
        weak_ids = [w.topic_id for w in weak_list]

    topics_to_generate = st.multiselect(
        "🎯 Chọn topics cần generate thêm câu hỏi",
        options=[r["topic_id"] for r in rows_data],
        default=weak_ids[:5] if weak_ids else [],
        format_func=lambda x: next((f"{r['topic_name']} ({r['chapter_id']})" for r in rows_data if r["topic_id"] == x), x),
    )

    num_per_topic = st.slider("Số câu/topic", 3, 10, 5, step=1)

    col_gen, col_check = st.columns([1, 1])
    with col_gen:
        if st.button("⚡ Generate On-Demand", type="primary", use_container_width=True):
            if topics_to_generate:
                with st.spinner("�ang phan tich va ghi config..."):
                    available, missing_topics = generate_on_demand(
                        topic_ids=topics_to_generate,
                        num_per_topic=num_per_topic,
                        auto_trigger=False,
                    )
                    cfg_path = write_on_demand_config_from_missing(missing_topics)
                    st.success(f"✅ Config ghi vào: `configs/on_demand_generation.yaml`")
                    st.info(
                        "⚠️ Vào tab **🎯 Tạo đề** và chạy **⚡ Chỉ chạy gen + eval (Steps 03-08)** "
                        "để sinh câu hỏi cho các topic đã chọn."
                    )
                    if available:
                        st.info(f"📦 Hien co {len(available)} cau trong pool cho topics đa chon.")
            else:
                st.warning("Vui lòng chọn ít nhất 1 topic.")

    with col_check:
        if st.button("🔄 Refresh Pool", type="secondary", use_container_width=True):
            count = refresh_pool()
            st.info(f"📦 Pool hiện có {count} câu hỏi.")


# ── Quiz Mode ─────────────────────────────────────────────────────────────────

def render_quiz_mode():
    st.subheader("🎮 Chế độ Quiz Adaptive")

    col_uid, col_mode, col_ch = st.columns([2, 2, 2])
    with col_uid:
        user_id = st.text_input(
            "👤 User ID",
            value=st.session_state.get("quiz_user_id", "student_demo"),
            placeholder="Nhập student ID của bạn",
        )
        st.session_state["quiz_user_id"] = user_id

    with col_mode:
        quiz_mode = st.selectbox(
            "📐 Chế độ quiz",
            options=["adaptive", "mixed", "focus_weak"],
            format_func=lambda x: {
                "adaptive": "🔄 Adaptive (70% yếu + 30% mạnh)",
                "mixed": "🎲 Mixed (ngẫu nhiên tất cả)",
                "focus_weak": "🎯 Focus Weak (chỉ topic yếu)",
            }.get(x, x),
            index=0,
        )

    with col_ch:
        topic_filter = st.selectbox(
            "📌 Tập trung topic",
            options=["Tất cả", "Chương 2", "Chương 3", "Chương 4", "Chương 5",
                    "Chương 6", "Chương 7a", "Chương 7b", "Chương 8",
                    "Chương 9", "Chương 10", "Chương 11"],
            format_func=lambda x: x,
            index=0,
        )

    num_questions = st.slider("📊 Số câu hỏi", 5, 30, 10, step=5)

    if st.session_state.get("quiz_started"):
        plan = get_study_plan(user_id)
        if plan and plan.notes:
            for note in plan.notes:
                if "⚠️" in note:
                    st.warning(note)
                elif "✅" in note:
                    st.success(note)
                else:
                    st.info(note)

    if not st.session_state.get("quiz_started"):
        weak_topics = detect_weak_topics(user_id)
        plan = get_study_plan(user_id)

        if weak_topics:
            st.markdown("**📌 Topics yếu cần ôn tập:**")
            for w in weak_topics[:5]:
                priority_emoji = {"critical": "🔴", "medium": "🟡", "low": "🟢"}.get(w.priority, "⚪")
                st.markdown(
                    f"  {priority_emoji} `{w.topic_name}` — acc={w.accuracy:.0%}, "
                    f"suggested: {DIFFICULTY_DISPLAY.get(w.suggested_difficulty, w.suggested_difficulty)}"
                )
        else:
            overall_acc = get_overall_accuracy(user_id)
            st.success(f"✅ Chưa có topic yếu! Overall accuracy: {overall_acc:.0%}. "
                       "Khuyến khích ôn tập đa dạng (chế độ Mixed).")

        col_start, col_clear = st.columns([1, 1])
        with col_start:
            if st.button("🚀 Bắt đầu Quiz", type="primary", use_container_width=True):
                with st.spinner("Đang tạo quiz..."):
                    chapter_to_topics = {
                        "Chương 2": ["ch02_t01", "ch02_t02", "ch02_t03", "ch02_t04"],
                        "Chương 3": ["ch03_t01", "ch03_t02"],
                        "Chương 4": ["ch04_t01", "ch04_t02", "ch04_t03", "ch04_t04", "ch04_t05"],
                        "Chương 5": ["ch05_t01", "ch05_t02", "ch05_t03"],
                        "Chương 6": ["ch06_t01", "ch06_t02"],
                        "Chương 7a": ["ch07a_t01", "ch07a_t02"],
                        "Chương 7b": ["ch07b_t01", "ch07b_t02", "ch07b_t03"],
                        "Chương 8": ["ch08_t01", "ch08_t02"],
                        "Chương 9": ["ch09_t01", "ch09_t02", "ch09_t03"],
                        "Chương 10": ["ch10_t01", "ch10_t02", "ch10_t03"],
                        "Chương 11": ["ch11_t01", "ch11_t02"],
                    }
                    focus_topics = None
                    if topic_filter != "Tất cả" and topic_filter in chapter_to_topics:
                        focus_topics = chapter_to_topics[topic_filter]

                    questions, plan = generate_adaptive_quiz(
                        user_id=user_id,
                        num_questions=num_questions,
                        focus_topics=focus_topics,
                        mode=quiz_mode,
                    )
                    if questions:
                        session = create_quiz_session(
                            user_id=user_id,
                            questions=questions,
                            mode=quiz_mode,
                        )
                        st.session_state["quiz_session"] = session.session_id
                        st.session_state["quiz_questions"] = questions
                        st.session_state["quiz_current"] = 0
                        st.session_state["quiz_answers"] = {}
                        st.session_state["quiz_started"] = True
                        st.session_state["quiz_start_time"] = time.time()
                        st.rerun()
                    else:
                        if plan and plan.notes:
                            for note in plan.notes:
                                st.warning(note)
                        else:
                            st.warning("Không có câu hỏi phù hợp trong pool.")

        with col_clear:
            if st.button("🔄 Reset Quiz", type="secondary", use_container_width=True):
                st.session_state["quiz_started"] = False
                st.session_state["quiz_session"] = None
                st.session_state["quiz_answers"] = {}
                st.rerun()
        return

    # Quiz in progress
    session_id = st.session_state.get("quiz_session", "")
    questions = st.session_state.get("quiz_questions", [])
    current = st.session_state.get("quiz_current", 0)
    answers = st.session_state.get("quiz_answers", {})

    if current >= len(questions):
        st.markdown("---")
        st.success("🎉 Quiz hoàn tất!")
        summary = end_session(session_id)
        col1, col2, col3 = st.columns(3)
        col1.metric("Tổng câu", summary.total_questions)
        col2.metric("Đúng", summary.correct_count)
        col3.metric("Accuracy", f"{summary.accuracy:.0%}")
        duration = summary.duration_seconds
        mins = int(duration // 60)
        secs = int(duration % 60)
        st.info(f"⏱️ Thời gian: {mins} phút {secs} giây")
        if summary.weak_topics_detected:
            st.markdown(f"⚠️ Weak topics mới: `{', '.join(summary.weak_topics_detected)}`")
        if st.button("🔄 Làm lại Quiz mới"):
            st.session_state["quiz_started"] = False
            st.session_state["quiz_session"] = None
            st.session_state["quiz_answers"] = {}
            st.rerun()
        return

    total_q = len(questions)
    elapsed = time.time() - st.session_state.get("quiz_start_time", time.time())
    progress = (current + 1) / total_q
    st.progress(progress, text=f"Câu {current + 1}/{total_q} — {elapsed:.0f}s")

    q = questions[current]
    qid = q.get("question_id", f"q_{current}")
    diff_label = q.get("difficulty_label", "G2")
    diff_display = DIFFICULTY_DISPLAY.get(diff_label, diff_label)
    topic_name = q.get("topic_name", q.get("_meta", {}).get("topic_name", ""))
    q_type = q.get("question_type", "single_correct")
    multi_tag = " [Multiple]" if q_type == "multiple_correct" else ""

    st.markdown(f"**Câu {current + 1}{multi_tag}** — {diff_display}")
    if topic_name:
        st.caption(f"📚 {topic_name}")
    st.markdown(q.get("question_text", ""))

    opts = q.get("options", {})
    selected = st.radio(
        "Chọn đáp án:",
        options=list(opts.keys()),
        format_func=lambda x: f"{x}. {opts[x]}",
        key=f"q_{qid}",
    )

    if qid in answers:
        result = answers[qid]
        correct_answers = result.get("correct_answers", [])
        is_correct = result.get("is_correct", False)
        explanation = result.get("explanation", "")

        with st.expander("📝 Xem kết quả", expanded=True):
            if is_correct:
                st.success("✅ Chính xác!")
            else:
                st.error(f"❌ Sai. Đáp án đúng: {', '.join(correct_answers)}")
            if explanation:
                st.markdown(explanation)

        if current < total_q - 1:
            if st.button("➡️ Câu tiếp theo", use_container_width=True):
                st.session_state["quiz_current"] = current + 1
                st.rerun()
    else:
        if st.button("✅ Nộp đáp án", type="primary", use_container_width=True):
            start_time = st.session_state.get("quiz_start_time", time.time())
            time_spent = int(time.time() - start_time)

            result = submit_answer(
                session_id=session_id,
                question_id=qid,
                user_answer=[selected],
                time_spent_seconds=time_spent,
            )
            answers[qid] = {
                "is_correct": result.is_correct,
                "correct_answers": result.correct_answers,
                "user_answer": result.user_answer,
                "explanation": result.explanation,
            }
            st.session_state["quiz_answers"] = answers
            st.session_state["quiz_start_time"] = time.time()
            st.rerun()


# ── Progress Dashboard ──────────────────────────────────────────────────────────

def render_progress_dashboard():
    st.subheader("📊 Tiến trình học tập")

    user_id = st.text_input(
        "👤 Nhập User ID",
        value=st.session_state.get("quiz_user_id", "student_demo"),
        key="progress_uid",
    )

    profile = load_profile(user_id)
    if profile is None:
        init_profile(user_id)
        profile = load_profile(user_id)

    overall = profile.overall_stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Overall Accuracy", f"{(overall.overall_accuracy if overall else 0)*100:.1f}%")
    col2.metric("📝 Total Attempts", overall.total_attempts if overall else 0)
    col3.metric("🔴 Weak Topics", overall.weak_topics_count if overall else 0)
    col4.metric("🟢 Strong Topics", overall.strong_topics_count if overall else 0)

    weak_list = detect_weak_topics(user_id)
    if weak_list:
        st.markdown("### ⚠️ Topics cần cải thiện")
        priority_colors = {"critical": "🔴", "medium": "🟡", "low": "🟢"}
        for w in weak_list:
            emoji = priority_colors.get(w.priority, "⚪")
            with st.container():
                col_p, col_acc, col_diff = st.columns([3, 1, 1])
                with col_p:
                    st.markdown(f"{emoji} **{w.topic_name}** (`{w.chapter_id}`)")
                with col_acc:
                    st.markdown(f"acc: {w.accuracy:.0%}")
                with col_diff:
                    st.markdown(f"→ {w.suggested_difficulty}")
    else:
        st.success("🎉 Chưa có topic yếu! Tiếp tục ôn tập đều đặn.")

    st.markdown("### 📚 Progress theo Chapter")
    topic_list = load_topic_list()
    for ch in topic_list:
        chapter_id = ch.get("chapter_id", "")
        chapter_name = ch.get("chapter_name", "")
        cs = profile.chapter_stats.get(chapter_id)
        acc = get_chapter_accuracy(user_id, chapter_id)
        topics_in_ch = ch.get("topics", [])

        if cs:
            attempted = cs.topics_attempted
            mastered = cs.topics_mastered
            weak = cs.topics_weak
        else:
            attempted = sum(1 for t in topics_in_ch if t.get("topic_id", "") in profile.topic_stats)
            mastered = weak = 0

        with st.expander(f"📘 {chapter_id}: {chapter_name} — {acc*100:.0f}% ({attempted}/{len(topics_in_ch)} topics)"):
            for topic in topics_in_ch:
                tid = topic.get("topic_id", "")
                ts = profile.topic_stats.get(tid)
                if ts and ts.attempts > 0:
                    bar_color = "🟢" if ts.accuracy >= 0.85 else "🔴" if ts.is_weak else "🟡"
                    st.markdown(
                        f"  {bar_color} `{tid}` — acc={ts.accuracy:.0%} ×{ts.attempts} attempts "
                        f"→ {DIFFICULTY_DISPLAY.get(ts.current_difficulty, ts.current_difficulty)}"
                    )
                else:
                    st.markdown(f"  ⬜ `{tid}` — chưa ôn tập")

    if st.button("📋 Xem kế hoạch học tập", type="primary", use_container_width=True):
        plan = get_study_plan(user_id)
        if plan:
            st.markdown(f"**📅 Generated at:** {plan.generated_at}")
            st.markdown(f"**📚 Total questions needed:** {plan.total_questions_needed}")
            if plan.priority_topics:
                st.markdown("**🎯 Priority topics:**")
                for w in plan.priority_topics:
                    st.markdown(f"  - `{w.topic_name}` (acc={w.accuracy:.0%}, priority={w.priority})")
            if plan.notes:
                st.markdown("**📝 Notes:**")
                for note in plan.notes:
                    st.markdown(f"  {note}")

    output_base = PROJECT_ROOT / "output"
    exp_dirs = sorted(output_base.glob("exp_*"), key=lambda p: p.stat().st_mtime, reverse=True)

    if not exp_dirs:
        st.info("Chưa có kết quả. Chạy pipeline trước.")
        return

    selected_exp = st.selectbox(
        "Chọn experiment",
        options=[d.name for d in exp_dirs],
        index=0,
    )

    exp_dir = output_base / selected_exp

    # Accepted questions
    accepted_file = exp_dir / "08_eval_iwf" / "final_accepted_questions.jsonl"
    if accepted_file.exists():
        questions = []
        with open(accepted_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))

        st.success(f"✅ {len(questions)} câu hỏi đã accept")

        # Stats
        g1 = sum(1 for q in questions if q.get("difficulty_label") == "G1")
        g2 = sum(1 for q in questions if q.get("difficulty_label") == "G2")
        g3 = sum(1 for q in questions if q.get("difficulty_label") == "G3")
        multi = sum(1 for q in questions if q.get("question_type") == "multiple_correct")

        cols = st.columns(4)
        cols[0].metric("Tổng", len(questions))
        cols[1].metric("G1/G2/G3", f"{g1}/{g2}/{g3}")
        cols[2].metric("Multiple", multi)
        cols[3].metric("Single", len(questions) - multi)

        # Preview
        st.markdown("**Preview 3 câu đầu:**")
        for q in questions[:3]:
            with st.expander(f"📌 {q.get('question_id', '?')}: {q.get('question_text','')[:80]}..."):
                st.markdown(f"**Type:** {q.get('question_type')}")
                st.markdown(f"**Difficulty:** {q.get('difficulty_label')}")
                opts = q.get("options", {})
                for letter, text in opts.items():
                    st.markdown(f"  **{letter}.** {text}")
                correct = q.get("correct_answers", [])
                st.markdown(f"**Đáp án đúng:** {', '.join(correct)}")
    else:
        st.info(f"Chưa có file accepted questions: {accepted_file}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()

    st.title("📝 CS431MCQGen — Sinh đề thi MCQ tự động + Adaptive Learning")
    st.markdown(
        "<small>Pipeline: RAG → P1 Stem+Key → P2+P3 Refine → P4-P8 Distractor CoT → Eval → Explanation "
        "| Adaptive Loop: Track → Weakness → Difficulty → Gen → Loop</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Layout
    with st.sidebar:
        manual_cfg, base_cfg = render_sidebar()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Tạo đề",
        "📋 Kết quả",
        "📦 Pool Coverage",
        "🎮 Quiz Mode",
        "📊 Progress",
        "📚 Tài liệu",
    ])

    with tab1:
        parsed = render_prompt_input(base_cfg)
        render_pipeline_steps()
        final_cfg = build_final_config(manual_cfg, parsed, base_cfg)
        render_pipeline_control(final_cfg, parsed)

    with tab2:
        render_results()

    with tab3:
        render_pool_coverage()

    with tab4:
        render_quiz_mode()

    with tab5:
        render_progress_dashboard()

    with tab6:
        st.markdown("""
        ## 📚 Hướng dẫn sử dụng

        ### Tab 🎯 Tạo đề — Sinh câu hỏi MCQ mới
        1. **Nhập yêu cầu** bằng prompt tự do (tiếng Việt hoặc tiếng Anh)
        2. **Tuỳ chỉnh** số câu, độ khó, tỉ lệ single/multiple
        3. **Chạy Pipeline** — Full (01b-10) hoặc Quick (03-08)

        ### Tab 📦 Pool Coverage — Phân tích pool câu hỏi
        1. Xem **số câu**, coverage %, phân bổ độ khó
        2. Xem **topics nào chưa có** câu hỏi
        3. Bấm **On-Demand Generation** để sinh thêm cho topics yếu

        ### Tab 🎮 Quiz Mode — Học tập Adaptive
        1. Nhập **User ID** để lưu tiến trình
        2. Chọn **chế độ**: Adaptive / Mixed / Focus Weak
        3. Làm bài — hệ thống tự động track và điều chỉnh độ khó

        ### Tab 📊 Progress — Xem tiến trình
        1. Nhập **User ID** để xem profile học tập
        2. Xem **overall accuracy**, weak/strong topics
        3. Xem progress theo **từng chapter**

        ### Bloom's Taxonomy → Difficulty Mapping
        | Bloom Level | Grade | Description |
        |---|---|---|
        | L1: Nhớ, L2: Hiểu | **G1** | Nhớ/Hiểu — recall, recognize |
        | L3: Áp dụng, L4: Phân tích | **G2** | Áp dụng/Phân tích — apply, analyze |
        | L5: Đánh giá, L6: Sáng tạo | **G3** | Đánh giá/Sáng tạo — evaluate, create |

        ### Adaptive Loop
        ```
        RAG Retrieval → MCQGen P1-P8 → Student Quiz →
        Tracking → Weakness Detection → Difficulty Controller →
        Adaptive Generation → Loop
        ```
        """)
        st.markdown("### 📖 Chapters trong CS116")
        for ch, (name, topics) in CHAPTER_INFO.items():
            st.markdown(f"- **{ch}**: {name} — {topics}")


if __name__ == "__main__":
    main()
