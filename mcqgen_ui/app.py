"""
mcqgen_ui/app.py — MCQGen Generator + Quiz + Improve
======================================================
Sections:
  1. Experiment selector + auto-fill from Improve trigger
  2. Topic list editor (with weak topics suggestion)
  3. Generate MCQ (retrieval direct + pipeline SLURM)
  4. Results viewer + download JSON
  5. Quiz mode (take quiz → finish → show answers + explanations)
  6. Quiz history + weak topics + Improve buttons
  7. Experiment history
"""

from __future__ import annotations
import json, subprocess, sys, time, uuid
from pathlib import Path
from datetime import datetime
import streamlit as st

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mcq_renderer import render_mcq_card, stats_summary, render_stats_html

TOPIC_LIST_SOURCE = PROJECT_ROOT / "input" / "topic_list_based.json"
LOG_DIR      = PROJECT_ROOT / "log"
OUTPUT_ROOT  = PROJECT_ROOT / "output"
QUIZ_DIR     = PROJECT_ROOT / "data" / "quiz_history"

STEP_META = {
    2: {"label":"02 — Retrieval","name":"Hybrid Retrieval"},
    3: {"label":"03 — Gen Stem","name":"P1: Generate Stem + Key"},
    4: {"label":"04 — Refine","name":"P2+P3: Self-Refine"},
    5: {"label":"05 — Distractors","name":"P4: Distractor Candidates"},
    6: {"label":"06 — CoT","name":"P5-P8: CoT Selection"},
    7: {"label":"07 — Eval","name":"Eval: 8 criteria"},
    8: {"label":"08 — IWF","name":"Eval: IWF"},
    9: {"label":"09 — Explain","name":"Explanation Generation"},
}
STEP_OUTPUT_FILES = {
    2:("02_retrieval","*.jsonl"), 3:("03_gen_stem","all_p1_results.jsonl"),
    4:("04_gen_refine","all_refined_results.jsonl"), 5:("05_gen_distractors","all_candidates_results.jsonl"),
    6:("06_gen_cot","all_final_mcqs.jsonl"), 7:("07_eval","evaluated_questions.jsonl"),
    8:("08_eval_iwf","final_accepted_questions.jsonl"), 9:("09_explain","explanations.jsonl"),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_topic_list_source():
    if not TOPIC_LIST_SOURCE.exists(): return []
    with open(TOPIC_LIST_SOURCE, encoding="utf-8") as f: return json.load(f)

def load_explanations(en):
    p = OUTPUT_ROOT / en / "09_explain" / "explanations.jsonl"
    if not p.exists() or p.stat().st_size == 0: return {}
    out = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                r = json.loads(line); qid = r.get("question_id","")
                if qid: out[qid] = r
    except: pass
    return out

def merge_mcqs_with_explanations(mcqs, explanations):
    enriched = []
    for mcq in mcqs:
        qid = mcq.get("question_id",""); rec = explanations.get(qid,{})
        if rec:
            mcq = dict(mcq); ed = rec.get("explanation",{}); mcq["explanation"] = ed
            su = ed.get("sources_used",[]) if isinstance(ed,dict) else []; sf = rec.get("sources",[])
            mcq["sources"] = su + sf
        enriched.append(mcq)
    return enriched

def load_mcqs(en):
    for name in ["final_accepted_questions.jsonl","evaluated_questions.jsonl"]:
        fp = OUTPUT_ROOT / en / "08_eval_iwf" / name
        if fp.exists() and fp.stat().st_size > 0:
            with open(fp, encoding="utf-8") as f: return [json.loads(l) for l in f if l.strip()]
    return []

def validate_step_output(exp_dir, sn):
    if sn not in STEP_OUTPUT_FILES: return True, 0
    dn, pat = STEP_OUTPUT_FILES[sn]; sd = exp_dir / dn
    if not sd.exists(): return False, 0
    if "*" in pat:
        files = sorted(sd.glob(pat))
        if not files: return False, 0
        t = 0
        for fp in files:
            if fp.stat().st_size == 0: return False, 0
            try:
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.strip(): json.loads(line); t += 1
            except: return False, t
        return t > 0, t
    fp = sd / pat
    if not fp.exists() or fp.stat().st_size == 0: return False, 0
    c = 0
    try:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip(): json.loads(line); c += 1
    except: return False, c
    return c > 0, c

def list_experiments():
    exps = []
    if not OUTPUT_ROOT.exists(): return exps
    for d in sorted(OUTPUT_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir(): continue
        ds = []; sc = {}
        for sn in STEP_OUTPUT_FILES:
            v, c = validate_step_output(d, sn); sc[sn] = c
            if v: ds.append(sn)
        mcqs = load_mcqs(d.name)
        exps.append({"name":d.name,"path":d,"done_steps":sorted(ds),"step_counts":sc,
                      "mcq_count":len(mcqs),"mtime":datetime.fromtimestamp(d.stat().st_mtime)})
    return exps

def poll_slurm_until_done(job_id, log_path, pb, stxt, lc):
    """Poll SLURM job. Returns final state string. Does NOT call st.rerun()."""
    lpos = 0; ll = []; it = 0; uc = 0
    while True:
        it += 1; state = "UNKNOWN"
        try:
            r = subprocess.run(["squeue","-j",job_id,"-o","%T","--noheader"],
                               capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                state = r.stdout.strip().upper(); uc = 0
            else:
                uc += 1
                if uc >= 2:
                    try:
                        sa = subprocess.run(["sacct","-j",job_id,"--format=State","--noheader","-P"],
                                            capture_output=True, text=True, timeout=10)
                        if sa.returncode == 0 and sa.stdout.strip():
                            ls = [l.strip() for l in sa.stdout.strip().split("\n") if l.strip()]
                            if ls: state = ls[0].split("+")[0].upper()
                    except: pass
                    if state == "UNKNOWN" and uc >= 4:
                        state = "COMPLETED"
        except: pass

        ic = {"RUNNING":"🟢","PENDING":"🟡","COMPLETED":"✅",
              "FAILED":"❌","CANCELLED":"🚫"}.get(state,"⚪")
        stxt.info(f"{ic} Job `{job_id}` — {state}")

        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    f.seek(lpos); nw = f.readlines(); lpos = f.tell()
                for raw in nw:
                    l = raw.strip()
                    if l: ll.append(l)
            except: pass
        if len(ll) > 500: ll = ll[-500:]

        if state in ("COMPLETED","FAILED","CANCELLED"):
            pb.progress(1.0, text=f"Job {state}")
            return state

        lc.text_area("📋 Log", value="\n".join(ll[-200:]),
                     height=200, disabled=True, key=f"log_{job_id}_{it}")
        time.sleep(12)

# ── Quiz history helpers ─────────────────────────────────────────────────────

def save_quiz_session(en, sess):
    d = QUIZ_DIR / en; d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{sess['session_id']}.json", "w", encoding="utf-8") as f:
        json.dump(sess, f, ensure_ascii=False, indent=2)

def load_quiz_sessions(en):
    d = QUIZ_DIR / en
    if not d.exists(): return []
    out = []
    for fp in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(fp, encoding="utf-8") as f: out.append(json.load(f))
        except: pass
    return out

def compute_weak_topics(sessions):
    ts = {}
    for sess in sessions:
        for r in sess.get("results",[]):
            tid = r.get("topic_id","") or r.get("_meta",{}).get("topic_id","")
            tn = r.get("topic_name","") or r.get("_meta",{}).get("topic_name","")
            if not tid: continue
            if tid not in ts: ts[tid] = {"topic_name":tn,"correct":0,"total":0}
            ts[tid]["total"] += 1
            if r.get("is_correct"): ts[tid]["correct"] += 1
    for v in ts.values(): v["accuracy"] = v["correct"]/v["total"] if v["total"]>0 else 0.0
    return ts

# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════

for k, d in [
    ("exp_name",""), ("topic_data",None), ("topic_edited",None),
    ("generation_phase",None), ("pipeline_job_id",None), ("generation_done",False),
    ("quiz_active",False), ("quiz_questions",[]), ("quiz_answers",{}),
    ("quiz_finished",False), ("quiz_session_id",""), ("_quiz_results",None),
    ("_improve_topics",None),
]:
    if k not in st.session_state: st.session_state[k] = d

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="MCQGen", page_icon="📝", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("### 📝 **MCQGen — Generator + Quiz**")

# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣ Experiment Selector
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("#### 1️⃣ Chọn hoặc tạo Experiment")
experiments = list_experiments()
exp_options = [e["name"] for e in experiments]

c1, c2, c3 = st.columns([2, 2, 1])
with c1:
    cur_idx = 0
    if st.session_state.exp_name and st.session_state.exp_name in exp_options:
        cur_idx = exp_options.index(st.session_state.exp_name) + 1
    sel = st.selectbox("Exp", ["— Tạo mới —"] + exp_options,
                       index=cur_idx, label_visibility="collapsed")
with c2:
    new_name = st.text_input("Tên mới", placeholder="exp_demo_v1",
                             label_visibility="collapsed")
with c3:
    st.markdown("")
    if st.button("✨ Áp dụng", use_container_width=True):
        if sel != "— Tạo mới —" and not new_name.strip():
            st.session_state.exp_name = sel
        elif new_name.strip():
            n = new_name.strip()
            if " " in n or "/" in n:
                st.warning("⚠️ Tên không hợp lệ.")
            else:
                st.session_state.exp_name = n
        st.session_state.topic_data = load_topic_list_source()
        st.session_state.topic_edited = None
        st.session_state.generation_phase = None
        st.session_state.pipeline_job_id = None
        st.session_state.quiz_active = False
        st.session_state.quiz_finished = False
        st.session_state._quiz_results = None
        st.rerun()

exp_name = st.session_state.exp_name

# ── Auto-fill from Improve trigger ───────────────────────────────────────────
if st.session_state._improve_topics is not None:
    improve = st.session_state._improve_topics
    st.session_state._improve_topics = None

    base = improve.get("source_exp", exp_name or "exp")
    new_exp = f"{base}_improve_{datetime.now().strftime('%H%M%S')}"
    st.session_state.exp_name = new_exp
    exp_name = new_exp

    full_topics = load_topic_list_source()
    weak_ids = set(improve["topic_ids"])
    filtered = []
    for ch in full_topics:
        ch_topics = [t for t in ch.get("topics", []) if t["topic_id"] in weak_ids]
        if ch_topics:
            filtered.append({**ch, "topics": ch_topics})

    st.session_state.topic_data = filtered
    st.session_state.topic_edited = filtered
    st.session_state.generation_phase = None
    st.session_state.generation_done = False
    st.session_state.pipeline_job_id = None
    st.session_state.quiz_active = False
    st.session_state.quiz_finished = False
    st.session_state._quiz_results = None

    topic_names = improve.get("topic_names", [])
    st.success(f"🎯 **Improve mode** — Focus: {', '.join(topic_names)}")
    st.info(f"📂 Experiment mới: `{new_exp}` — Topic list đã filter. Nhấn **Generate**.")

# ── Show experiment info + step grid ──────────────────────────────────────────
if exp_name:
    exp_dir = OUTPUT_ROOT / exp_name
    exp_obj = next((e for e in experiments if e["name"] == exp_name), None)
    if exp_obj:
        st.success(f"📂 **`{exp_name}`** — MCQs: {exp_obj['mcq_count']} — "
                   f"{exp_obj['mtime'].strftime('%d/%m %H:%M')}")
    else:
        st.success(f"📂 **`{exp_name}`** — Mới")

    for row in [(2, 3, 4, 5), (6, 7, 8, 9)]:
        cols = st.columns(len(row))
        for i, sn in enumerate(row):
            m = STEP_META[sn]
            v, cnt = validate_step_output(exp_dir, sn)
            with cols[i]:
                if v:
                    bg, bd, ic, sta = "#f0fdf4", "#22c55e", "✅", f"Done — {cnt}"
                elif (exp_dir / STEP_OUTPUT_FILES[sn][0]).exists():
                    bg, bd, ic, sta = "#fffbeb", "#f59e0b", "⚠️", "Incomplete"
                else:
                    bg, bd, ic, sta = "#f9fafb", "#e5e7eb", "⬜", "Chưa chạy"
                st.markdown(
                    f'<div style="background:{bg};border:2px solid {bd};'
                    f'border-radius:10px;padding:8px;text-align:center">'
                    f'<div style="font-weight:700;font-size:12px">{ic} {m["label"]}</div>'
                    f'<div style="font-size:10px;color:#6b7280">{sta}</div></div>',
                    unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣ Topic List Editor
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.topic_data is None:
    st.info("Chọn experiment bên trên.")
else:
    # Weak topics warning
    if exp_name:
        past = load_quiz_sessions(exp_name)
        if past:
            wk = compute_weak_topics(past)
            wl_warn = {t: i for t, i in wk.items()
                       if i["accuracy"] < 0.5 and i["total"] >= 2}
            if wl_warn:
                wn = [f"**{v['topic_name']}** ({v['accuracy']*100:.0f}%)"
                      for v in sorted(wl_warn.values(), key=lambda x: x["accuracy"])]
                st.warning(f"🎯 Topics yếu: {', '.join(wn)}. Nên focus sinh thêm câu.")

    with st.expander("2️⃣ Topic List Editor", expanded=False):
        cur = st.session_state.topic_edited or st.session_state.topic_data
        ed = st.text_area("JSON", value=json.dumps(cur, ensure_ascii=False, indent=2),
                          height=300, label_visibility="collapsed")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            if st.button("💾 Áp dụng", key="ts"):
                try:
                    p = json.loads(ed)
                    st.session_state.topic_edited = p
                    st.session_state.topic_data = p
                    st.success("✅")
                except json.JSONDecodeError as e:
                    st.error(f"❌ {e}")
        with cc2:
            if st.button("🔍 Check", key="tc"):
                try:
                    p = json.loads(ed)
                    st.success(f"✅ {sum(len(c.get('topics',[]))for c in p)} topics")
                except json.JSONDecodeError as e:
                    st.error(f"❌ {e}")
        with cc3:
            if st.button("↩️ Reset", key="tr"):
                st.session_state.topic_edited = None
                st.session_state.topic_data = load_topic_list_source()
                st.rerun()

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # 3️⃣ Generate
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("#### 3️⃣ Sinh câu hỏi")
    g1, g2, g3 = st.columns([1, 1, 4])
    with g1:
        gp = st.button("🚀 Generate", use_container_width=True,
                        disabled=st.session_state.generation_phase is not None
                                or not st.session_state.topic_data)
    with g2:
        if st.session_state.generation_phase == "pipeline" and st.session_state.pipeline_job_id:
            if st.button("⛔ Stop", use_container_width=True):
                subprocess.run(["scancel", st.session_state.pipeline_job_id],
                               capture_output=True)
                st.session_state.generation_phase = None
                st.session_state.pipeline_job_id = None
                st.rerun()
    with g3:
        ph = st.session_state.generation_phase
        if ph == "pipeline":
            st.info(f"⏳ Pipeline... Job: `{st.session_state.pipeline_job_id}`")
        elif ph == "done":
            st.success("✅ Done!")
        else:
            st.caption("Generate → retrieval → pipeline 03-09")

    # ── Generate button pressed ──────────────────────────────────────────────
    if gp and st.session_state.generation_phase is None:
        en = st.session_state.exp_name
        td = st.session_state.topic_edited or st.session_state.topic_data
        if not en or not td:
            st.error("⚠️ Thiếu experiment/topic.")
        else:
            ep = OUTPUT_ROOT / en
            ep.mkdir(parents=True, exist_ok=True)
            with open(ep / "topic_list.json", "w", encoding="utf-8") as f:
                json.dump(td, f, ensure_ascii=False, indent=2)
            with open(PROJECT_ROOT / "input" / "topic_list.json", "w", encoding="utf-8") as f:
                json.dump(td, f, ensure_ascii=False, indent=2)

            cpy = PROJECT_ROOT / "src" / "common.py"
            with open(cpy, encoding="utf-8") as f:
                ct = f.read()
            import re as _re
            ct = _re.sub(
                r'(class Config:\s+#[^\n]*\n\s+EXP_NAME\s*=\s*")[^"]*(")',
                rf'\g<1>{en}\g<2>', ct)
            with open(cpy, "w", encoding="utf-8") as f:
                f.write(ct)

            LOG_DIR.mkdir(parents=True, exist_ok=True)

            # Phase 1: Retrieval (CPU, synchronous)
            st.info("🔄 Retrieval...")
            rv = {
                **subprocess.os.environ,
                "EXP_NAME": en, "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "",
                "PYTHONPATH": str(PROJECT_ROOT),
            }
            retrieval_ok = False
            try:
                rr = subprocess.run(
                    ["python", "-u", str(PROJECT_ROOT / "src" / "gen" / "retrieval.py")],
                    capture_output=True, text=True, cwd=str(PROJECT_ROOT),
                    env=rv, timeout=600)
                if rr.returncode == 0:
                    st.success("✅ Retrieval done!")
                    retrieval_ok = True
                else:
                    st.error(f"❌\n```\n{rr.stderr[-500:]}\n```")
            except subprocess.TimeoutExpired:
                st.error("❌ Timeout")
            except Exception as e:
                st.error(f"❌ {e}")

            if not retrieval_ok:
                st.stop()

            # Phase 2: Pipeline 03-09 (SLURM)
            ps = PROJECT_ROOT / "scripts" / "03_09_pipeline.sh"
            if not ps.exists():
                st.error(f"❌ Not found: {ps}")
            else:
                try:
                    res = subprocess.run(
                        ["sbatch", "--parsable",
                         "--output", str(LOG_DIR / f"03_09_{en}_%j.out"),
                         "--error", str(LOG_DIR / f"03_09_{en}_%j.err"),
                         str(ps)],
                        capture_output=True, text=True, timeout=30,
                        env={**subprocess.os.environ, "EXP_NAME": en})
                    if res.returncode == 0:
                        jid = res.stdout.strip().split(";")[0].strip()
                        st.session_state.generation_phase = "pipeline"
                        st.session_state.pipeline_job_id = jid
                        st.success(f"✅ Pipeline: `{jid}`")
                        st.rerun()
                    else:
                        st.error(f"❌ {res.stderr}")
                except Exception as e:
                    st.error(f"❌ {e}")

    # ── Poll pipeline job ────────────────────────────────────────────────────
    if (st.session_state.generation_phase == "pipeline"
            and st.session_state.pipeline_job_id):
        jid = st.session_state.pipeline_job_id
        en = st.session_state.exp_name
        pb = st.progress(0.5, text="Pipeline 03-09...")
        stxt = st.empty()
        lc = st.empty()
        lfs = sorted(LOG_DIR.glob(f"03_09_{en}_*.out"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        lp = lfs[0] if lfs else LOG_DIR / f"03_09_{en}_%j.out"

        state = poll_slurm_until_done(jid, lp, pb, stxt, lc)

        # Update state BEFORE rerun
        if state == "COMPLETED":
            st.session_state.generation_phase = "done"
            st.session_state.generation_done = True
        else:
            st.error(f"❌ {state}")
            st.session_state.generation_phase = None
        st.session_state.pipeline_job_id = None  # prevent re-poll
        time.sleep(1)
        st.rerun()

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # 4️⃣ Results + Download
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("#### 4️⃣ Kết quả")
    en = st.session_state.exp_name
    mcqs = load_mcqs(en) if en else []

    if mcqs:
        expl = load_explanations(en)
        me = merge_mcqs_with_explanations(mcqs, expl)
        st.markdown(render_stats_html(stats_summary(me)), unsafe_allow_html=True)

        d1, d2, d3 = st.columns([1, 1, 4])
        with d1:
            st.download_button("📥 MCQs", json.dumps(mcqs, ensure_ascii=False, indent=2),
                               f"{en}_mcqs.json", "application/json", use_container_width=True)
        with d2:
            if expl:
                st.download_button("📥 Explain",
                                   json.dumps(list(expl.values()), ensure_ascii=False, indent=2),
                                   f"{en}_expl.json", "application/json", use_container_width=True)
        with d3:
            if expl:
                st.download_button("📥 All",
                                   json.dumps(me, ensure_ascii=False, indent=2),
                                   f"{en}_full.json", "application/json")
        st.markdown("---")

        # ═══════════════════════════════════════════════════════════════════════
        # 5️⃣ Quiz Mode
        # ═══════════════════════════════════════════════════════════════════════

        st.markdown("#### 5️⃣ Làm bài")

        if not st.session_state.quiz_active and not st.session_state.quiz_finished:
            # ── Start quiz ───────────────────────────────────────────────────
            if st.button("📝 Bắt đầu làm bài", use_container_width=False):
                st.session_state.quiz_active = True
                st.session_state.quiz_questions = me
                st.session_state.quiz_answers = {}
                st.session_state.quiz_finished = False
                st.session_state.quiz_session_id = f"quiz_{uuid.uuid4().hex[:8]}"
                st.session_state._quiz_results = None
                st.rerun()

        elif st.session_state.quiz_active and not st.session_state.quiz_finished:
            # ── Taking quiz ──────────────────────────────────────────────────
            qs = st.session_state.quiz_questions
            ans = st.session_state.quiz_answers
            st.info(f"📝 {len(qs)} câu — chọn đáp án rồi nhấn **Kết thúc**.")

            for idx, q in enumerate(qs):
                qid = q.get("question_id", f"q_{idx}")
                opts = q.get("options", {})
                qtype = q.get("question_type", "single_correct")
                diff = q.get("difficulty_label", "")

                st.markdown(f"**Câu {idx+1}** ({diff})")
                st.markdown(q.get("question_text", ""))

                labels = sorted(opts.keys())
                if qtype == "multiple_correct":
                    sel = []
                    for lb in labels:
                        if st.checkbox(f"{lb}. {opts[lb]}",
                                       key=f"q_{idx}_{qid}_{lb}",
                                       value=lb in ans.get(qid, [])):
                            sel.append(lb)
                    ans[qid] = sel
                else:
                    display = [f"{lb}. {opts[lb]}" for lb in labels]
                    prev = ans.get(qid, [None])[0]
                    pi = labels.index(prev) if prev in labels else None
                    ch = st.radio("Đáp án:", display, key=f"q_{idx}_{qid}",
                                  index=pi, label_visibility="collapsed")
                    if ch:
                        ans[qid] = [ch.split(".")[0]]
                st.markdown("---")

            st.session_state.quiz_answers = ans
            answered = sum(1 for v in ans.values() if v)
            st.markdown(f"**Đã trả lời: {answered}/{len(qs)}**")

            if st.button("🏁 Kết thúc làm bài", type="primary", use_container_width=True):
                results = []
                for q in qs:
                    qid = q.get("question_id", "")
                    ua = set(a.upper() for a in ans.get(qid, []))
                    ca = set(c.upper() for c in q.get("correct_answers", []))
                    is_correct = ua == ca
                    tid = (q.get("_meta", {}).get("topic_id", "")
                           or q.get("topic_id", ""))
                    tn = (q.get("_meta", {}).get("topic_name", "")
                          or q.get("topic_name", ""))
                    results.append({
                        "question_id": qid, "topic_id": tid, "topic_name": tn,
                        "user_answer": sorted(ua), "correct_answers": sorted(ca),
                        "is_correct": is_correct,
                        "difficulty": q.get("difficulty_label", ""),
                    })
                cc = sum(1 for r in results if r["is_correct"])
                sd = {
                    "session_id": st.session_state.quiz_session_id,
                    "exp_name": en,
                    "timestamp": datetime.now().isoformat(),
                    "total": len(results), "correct": cc,
                    "accuracy": cc / len(results) if results else 0,
                    "results": results,
                }
                save_quiz_session(en, sd)
                st.session_state.quiz_active = False
                st.session_state.quiz_finished = True
                st.session_state._quiz_results = sd
                st.rerun()

        elif st.session_state.quiz_finished:
            # ── Show quiz results with explanations ──────────────────────────
            sd = st.session_state._quiz_results
            if sd is None:
                ss = load_quiz_sessions(en)
                sd = ss[0] if ss else {}

            if sd:
                total = sd.get("total", 0)
                correct = sd.get("correct", 0)
                acc = sd.get("accuracy", 0)
                color = "#22c55e" if acc >= 0.7 else "#f59e0b" if acc >= 0.5 else "#ef4444"
                msg = ("Xuất sắc!" if acc >= 0.8
                       else "Khá tốt!" if acc >= 0.6
                       else "Cần ôn thêm!")
                st.markdown(
                    f'<div style="background:{color}20;border:2px solid {color};'
                    f'border-radius:12px;padding:16px;text-align:center;margin:8px 0">'
                    f'<div style="font-size:28px;font-weight:700;color:{color}">'
                    f'{correct}/{total} ({acc*100:.0f}%)</div>'
                    f'<div style="font-size:14px;color:#6b7280">{msg}</div></div>',
                    unsafe_allow_html=True)

                # Load explanations fresh from file
                fresh_expl = load_explanations(en)
                qmap = {}
                if st.session_state.quiz_questions:
                    qmap = {q.get("question_id", ""): q
                            for q in st.session_state.quiz_questions}

                for r in sd.get("results", []):
                    qid = r["question_id"]
                    q = qmap.get(qid, {})
                    ic = "✅" if r["is_correct"] else "❌"
                    st.markdown(f"**{ic} {qid}** — {r.get('topic_name','')} "
                                f"({r.get('difficulty','')})")
                    if q:
                        st.markdown(q.get("question_text", ""))

                    # Options with marking
                    opts = q.get("options", {}) if q else {}
                    cs = set(r.get("correct_answers", []))
                    us = set(r.get("user_answer", []))

                    if opts:
                        for lb in sorted(opts.keys()):
                            tx = opts[lb]
                            if lb in cs and lb in us:
                                st.markdown(f"&nbsp;&nbsp;✅ **{lb}. {tx}** ← Bạn chọn (đúng)")
                            elif lb in cs:
                                st.markdown(f"&nbsp;&nbsp;🟢 **{lb}. {tx}** ← Đáp án đúng")
                            elif lb in us:
                                st.markdown(f"&nbsp;&nbsp;❌ ~~{lb}. {tx}~~ ← Bạn chọn (sai)")
                            else:
                                st.markdown(f"&nbsp;&nbsp;⬜ {lb}. {tx}")
                    else:
                        st.markdown(f"&nbsp;&nbsp;Bạn: **{', '.join(sorted(us))}** "
                                    f"— Đúng: **{', '.join(sorted(cs))}**")

                    # Explanation: fresh from file, fallback to session state
                    expl_rec = fresh_expl.get(qid, {})
                    expl_data = expl_rec.get("explanation", {})
                    if not expl_data and q:
                        expl_data = q.get("explanation", {})

                    if expl_data:
                        with st.expander(f"💡 Giải thích — {qid}"):
                            if isinstance(expl_data, str):
                                st.markdown(expl_data)
                            elif isinstance(expl_data, dict):
                                mot = expl_data.get("question_motivation", "")
                                if mot:
                                    st.markdown("**📌 Lý do đặt câu hỏi:**")
                                    st.markdown(mot)

                                rat = expl_data.get("correct_answer_rationale", "")
                                if rat:
                                    st.markdown(f"**✅ Đáp án đúng "
                                                f"({', '.join(sorted(cs))}):**")
                                    st.markdown(rat)

                                di = expl_data.get("distractor_explanations", {})
                                if di:
                                    st.markdown("**❌ Giải thích đáp án sai:**")
                                    for lb in sorted(di.keys()):
                                        opt_tx = opts.get(lb, "") if opts else ""
                                        pf = (f"**{lb}. {opt_tx}:**"
                                              if opt_tx else f"**{lb}:**")
                                        st.markdown(f"- {pf} {di[lb]}")

                                kc = expl_data.get("knowledge_context", {})
                                if isinstance(kc, dict) and kc:
                                    kcon = kc.get("key_concepts", [])
                                    if kcon:
                                        st.markdown(f"**📚 Khái niệm:** "
                                                    f"{', '.join(kcon)}")
                                    prereqs = kc.get("prerequisites", [])
                                    if prereqs:
                                        st.markdown(f"**🔗 Tiền đề:** "
                                                    f"{', '.join(prereqs)}")

                                sources = (expl_data.get("sources_used", [])
                                           or expl_rec.get("sources", []))
                                if sources:
                                    st.markdown("**📖 Nguồn:**")
                                    for src in sources:
                                        if isinstance(src, dict):
                                            stype = src.get("type", "")
                                            url = src.get("url", "")
                                            desc = src.get("description",
                                                           src.get("title", ""))
                                            page = src.get("page", "")
                                            ts_s = src.get("timestamp_start", "")
                                            if stype == "slide" and page:
                                                st.markdown(
                                                    f"- 📄 {desc} (trang {page})")
                                            elif url:
                                                label = desc or url
                                                if ts_s:
                                                    label += f" ({ts_s})"
                                                st.markdown(f"- 🔗 [{label}]({url})")
                                            elif desc:
                                                st.markdown(f"- {desc}")
                    st.markdown("---")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("🔄 Làm lại", use_container_width=True):
                        st.session_state.quiz_finished = False
                        st.session_state.quiz_active = False
                        st.session_state._quiz_results = None
                        st.rerun()
                with b2:
                    st.download_button(
                        "📥 Kết quả quiz",
                        json.dumps(sd, ensure_ascii=False, indent=2),
                        f"{en}_quiz_{sd.get('session_id','')}.json",
                        "application/json", use_container_width=True)

    else:
        st.info("Chưa có kết quả — chạy generation bên trên.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # 6️⃣ Quiz History + Weak Topics + Improve
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("#### 6️⃣ Lịch sử làm bài")
    if exp_name:
        sessions = load_quiz_sessions(exp_name)
        if sessions:
            wk = compute_weak_topics(sessions)
            wl = [(t, i) for t, i in wk.items()
                  if i["accuracy"] < 0.5 and i["total"] >= 2]
            if wl:
                wl.sort(key=lambda x: x[1]["accuracy"])
                st.markdown("**🎯 Topics yếu:**")
                for tid, info in wl:
                    wc1, wc2 = st.columns([4, 1])
                    with wc1:
                        pct = info["accuracy"] * 100
                        st.markdown(f"- **{info['topic_name']}** — "
                                    f"{info['correct']}/{info['total']} ({pct:.0f}%)")
                    with wc2:
                        if st.button("🔧 Improve", key=f"imp_{tid}",
                                     use_container_width=True):
                            st.session_state._improve_topics = {
                                "topic_ids": [tid],
                                "topic_names": [info["topic_name"]],
                                "source_exp": exp_name,
                            }
                            st.rerun()

                if len(wl) > 1:
                    if st.button(f"🔧 Improve tất cả {len(wl)} topics yếu",
                                 use_container_width=False):
                        st.session_state._improve_topics = {
                            "topic_ids": [tid for tid, _ in wl],
                            "topic_names": [info["topic_name"] for _, info in wl],
                            "source_exp": exp_name,
                        }
                        st.rerun()
                st.markdown("")

            # Session list
            for sess in sessions[:10]:
                sid = sess.get("session_id", "?")
                ts = sess.get("timestamp", "")[:16]
                t = sess.get("total", 0)
                c = sess.get("correct", 0)
                a = sess.get("accuracy", 0)
                ic = "✅" if a >= 0.7 else "⚠️" if a >= 0.5 else "❌"
                with st.expander(f"{ic} `{sid}` — {ts} — {c}/{t} ({a*100:.0f}%)"):
                    for r in sess.get("results", []):
                        ri = "✅" if r["is_correct"] else "❌"
                        st.markdown(
                            f"  {ri} `{r['question_id']}` — "
                            f"{r.get('topic_name','')} — "
                            f"Bạn: {r['user_answer']} / "
                            f"Đúng: {r['correct_answers']}")
        else:
            st.info("Chưa có lịch sử.")

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════════════════════
    # 7️⃣ Experiment History
    # ═══════════════════════════════════════════════════════════════════════════

    with st.expander("📁 Experiments", expanded=False):
        if experiments:
            for exp in experiments[:15]:
                ec1, ec2, ec3 = st.columns([3, 2, 1])
                with ec1:
                    badge = "✅" if exp["mcq_count"] > 0 else "⬜"
                    st.markdown(
                        f"{badge} **`{exp['name']}`** — "
                        f"{exp['mtime'].strftime('%d/%m %H:%M')} — "
                        f"{exp['mcq_count']} MCQs")
                with ec2:
                    st.caption(
                        f"Steps: {', '.join(f'S{s}' for s in exp['done_steps']) or '—'}")
                with ec3:
                    if st.button("Chọn", key=f"sel_{exp['name']}"):
                        st.session_state.exp_name = exp["name"]
                        st.session_state.topic_data = load_topic_list_source()
                        st.session_state.topic_edited = None
                        st.session_state.generation_phase = None
                        st.session_state.pipeline_job_id = None
                        st.session_state.quiz_active = False
                        st.session_state.quiz_finished = False
                        st.session_state._quiz_results = None
                        st.rerun()
                st.markdown("---")
        else:
            st.info("Chưa có.")