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


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CS431MCQGen — Sinh đề thi MCQ",
    page_icon="📝",
    layout="wide",
)


# ── Constants ───────────────────────────────────────────────────────────────

CHAPTER_INFO = {
    "ch02": ("Popular Libraries", "NumPy, Pandas, Matplotlib, Scikit-learn"),
    "ch03": ("Pipeline & EDA", "Pipeline, Exploratory Data Analysis"),
    "ch04": ("Tiền xử lý dữ liệu", "Missing Data, Outlier Detection, Feature Extraction/Transformation/Selection"),
    "ch05": ("Đánh giá mô hình", "Classification Metrics, Regression Metrics, Cross-validation"),
    "ch06": ("Unsupervised Learning", "Clustering, Dimensionality Reduction"),
    "ch07a": ("Supervised Learning - Regression", "Linear Regression, Regularization"),
    "ch07b": ("Supervised Learning - Classification", "Logistic Regression, Decision Trees, SVM"),
    "ch08": ("Deep Learning với CNN", "Neural Networks, CNN"),
    "ch09": ("Parameter Tuning", "Grid Search, Random Search, Bayesian Optimization"),
    "ch10": ("Ensemble Models", "Bagging, Boosting, Random Forest"),
    "ch11": ("Model Deployment", "Model Serving, API, Monitoring"),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    init_state()

    st.title("📝 CS431MCQGen — Sinh đề thi MCQ tự động")
    st.markdown(
        "<small>Pipeline: RAG → P1 Stem+Key → P2+P3 Refine → P4-P8 Distractor CoT → Eval → Explanation</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Layout
    with st.sidebar:
        manual_cfg, base_cfg = render_sidebar()

    tab1, tab2, tab3 = st.tabs(["🎯 Tạo đề", "📋 Kết quả", "📚 Tài liệu"])

    with tab1:
        parsed = render_prompt_input(base_cfg)
        render_pipeline_steps()
        final_cfg = build_final_config(manual_cfg, parsed, base_cfg)
        render_pipeline_control(final_cfg, parsed)

    with tab2:
        render_results()

    with tab3:
        st.markdown("""
        ## 📚 Hướng dẫn sử dụng

        ### 1. Nhập yêu cầu
        Dùng câu prompt tự do bằng tiếng Việt hoặc tiếng Anh.
        Hệ thống sẽ tự parse để xác định:
        - Chapters cần tập trung
        - Số câu hỏi
        - Độ khó

        ### 2. Tuỳ chỉnh cấu hình
        Dùng sidebar để điều chỉnh:
        - Số câu hỏi (target range)
        - Tỉ lệ single/multiple answer
        - Chapters trọng điểm và trọng số

        ### 3. Chạy Pipeline
        - **Full pipeline (01-09):** Indexing → Retrieval → Gen → Eval → Explanation
        - **Quick (03-08):** Chỉ chạy generation + eval (đã có data)

        ### 4. Xem kết quả
        Tab "Kết quả" hiển thị:
        - Số câu hỏi accept
        - Phân bổ G1/G2/G3
        - Tỉ lệ single/multiple
        - Preview câu hỏi

        ### Chapters
        """)
        for ch, (name, topics) in CHAPTER_INFO.items():
            st.markdown(f"- **{ch}**: {name} — {topics}")


if __name__ == "__main__":
    main()
