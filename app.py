"""SignalSentry Streamlit dashboard."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Local / NemoClaw live mode is controlled by .env (USE_MOCK_MODEL=false).
# Streamlit Community Cloud cannot reach inference.local — default mock until BYOK.
from src.models.llm import (
    PUBLIC_MODEL_CHOICES,
    force_mock_if_hosted_demo,
    is_hosted_demo_environment,
    load_model_config,
    resolve_model_config,
)

force_mock_if_hosted_demo()
os.environ.setdefault("MODEL_BASE_URL", "https://inference.local/v1")
os.environ.setdefault("MODEL_API_KEY", "nemoclaw-local-placeholder")
os.environ.setdefault("MODEL_NAME", "nvidia/nemotron-mini")
os.environ.setdefault("SEED", "42")

from src.agents.investigators import ask_assistant, investigate_churn
from src.generation.generate_all import META_PATH, generate_all
from src.models.schemas import CandidateAlert, Severity
from src.paths import GENERATED_DIR, GROUND_TRUTH_DIR, OUTPUTS_DIR
from src.presentation import build_churn_briefing, build_media_briefing
from src.privacy import (
    list_readable_files,
    read_investigation_log,
    synthetic_data_confirmation,
)
from src.run_analysis import run_analysis

st.set_page_config(
    page_title="SignalSentry",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_PAGES = (
    "Home",
    "Account risks",
    "Campaign issues",
    "How it works",
    "Privacy",
)


def _ensure_nav_state() -> None:
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Home"
    if st.session_state.nav_page not in NAV_PAGES:
        st.session_state.nav_page = "Home"
    # Keep radio widget in sync when brand / ← Home jumps pages.
    if "nav_radio" not in st.session_state:
        st.session_state.nav_radio = st.session_state.nav_page
    elif st.session_state.get("_force_nav_sync"):
        st.session_state.nav_radio = st.session_state.nav_page
        st.session_state._force_nav_sync = False


def _go_home() -> None:
    st.session_state.nav_page = "Home"
    st.session_state.nav_radio = "Home"
    st.session_state._force_nav_sync = True


def _page_header(title: str, subtitle: str = "") -> None:
    """Title row with an always-visible Home control."""
    left, right = st.columns([5, 1])
    with left:
        st.title(title)
        if subtitle:
            st.caption(subtitle)
    with right:
        st.write("")
        if title != "Home" and st.button("← Home", use_container_width=True, key=f"home_btn_{title}"):
            _go_home()
            st.rerun()



def _read_generation_meta() -> dict:
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def ensure_demo_data() -> None:
    """Generate synthetic data + analysis if outputs are missing (Streamlit Cloud cold start)."""
    alerts_path = OUTPUTS_DIR / "churn_alerts.json"
    churn_path = GENERATED_DIR / "churn_metrics.parquet"
    if alerts_path.exists() and churn_path.exists():
        return
    with st.spinner("Preparing synthetic demo data (first run only)..."):
        # Cold start always uses mock so Cloud boots without an API key.
        generate_all(seed=int(os.getenv("SEED", "42")))
        run_analysis(max_investigations=25, config=load_model_config())


def regenerate_demo_world(
    *,
    seed: int,
    n_accounts: int,
    n_campaigns: int,
    max_investigations: int,
) -> None:
    """Build a new synthetic world and re-run detect → investigate → evaluate."""
    generate_all(seed=seed, n_accounts=n_accounts, n_campaigns=n_campaigns)
    run_analysis(max_investigations=max_investigations, config=active_model_config())
    _load_metrics.clear()


def active_model_config():
    """Session-aware config: BYOK live path, else hosted mock / local .env."""
    return resolve_model_config(
        visitor_api_key=st.session_state.get("byok_api_key") or None,
        visitor_model=st.session_state.get("byok_model") or None,
    )


def byok_sidebar() -> None:
    """Portfolio policy: your key is never required; visitors may bring their own."""
    if "byok_model" not in st.session_state:
        st.session_state.byok_model = PUBLIC_MODEL_CHOICES[0]

    with st.sidebar.expander("Live LangChain (optional)", expanded=False):
        st.caption(
            "Default is mock (no API spend). Paste an `nvapi-…` key from build.nvidia.com, "
            "pick a model, then ask from any page."
        )
        key = st.text_input(
            "NVIDIA API key",
            type="password",
            value=st.session_state.get("byok_api_key", ""),
            help="Create a key at build.nvidia.com.",
            key="byok_input",
        )
        model = st.selectbox(
            "Model",
            options=list(PUBLIC_MODEL_CHOICES),
            index=list(PUBLIC_MODEL_CHOICES).index(st.session_state.byok_model)
            if st.session_state.byok_model in PUBLIC_MODEL_CHOICES
            else 0,
            help="If you see NotFoundError, switch models — availability varies by account.",
        )
        cols = st.columns(2)
        if cols[0].button("Use key", use_container_width=True):
            cleaned = key.strip()
            if cleaned and not cleaned.replace(" ", "").startswith("nvapi-") and "nvapi-" not in cleaned:
                st.warning("Key usually starts with `nvapi-`. Paste the full key from build.nvidia.com.")
            st.session_state.byok_api_key = cleaned
            st.session_state.byok_model = model
            st.rerun()
        if cols[1].button("Clear key", use_container_width=True):
            st.session_state.byok_api_key = ""
            os.environ.pop("SIGNAL_SENTRY_BYOK_ACTIVE", None)
            st.rerun()
        # Keep model selection even before Use key (for next apply)
        st.session_state.byok_model = model
        cfg = active_model_config()
        if cfg.use_mock:
            st.caption("Active: mock")
        elif cfg.is_nemoclaw_route:
            st.caption(f"Active: NemoClaw · {cfg.model_name}")
        else:
            st.caption(f"Active: BYOK · {cfg.model_name}")


def _flagged_issues_context(data: dict) -> list[dict]:
    """Compact issue list for Ask (local answers + live LLM context)."""
    rows: list[dict] = []
    inv_churn = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data.get("churn_investigations") or []
        if "alert" in row
    }
    inv_media = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data.get("media_investigations") or []
        if "alert" in row
    }
    for a in data.get("churn_alerts") or []:
        key = a["entity_id"] + "|" + a["alert_type"]
        brief = build_churn_briefing(a, inv_churn.get(key, {}))
        rows.append(
            {
                "kind": "Account",
                "domain": "churn",
                "entity_id": a.get("entity_id"),
                "severity": a.get("severity"),
                "headline": brief.get("headline"),
                "insight": brief.get("insight"),
                "recommended_action": brief.get("recommended_action"),
            }
        )
    for a in data.get("media_alerts") or []:
        key = a["entity_id"] + "|" + a["alert_type"]
        brief = build_media_briefing(a, inv_media.get(key, {}))
        rows.append(
            {
                "kind": "Campaign",
                "domain": "media",
                "entity_id": a.get("entity_id"),
                "severity": a.get("severity"),
                "headline": brief.get("headline"),
                "insight": brief.get("insight"),
                "recommended_action": brief.get("recommended_action"),
            }
        )
    sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    rows.sort(key=lambda r: sev_rank.get(str(r.get("severity", "")).lower(), 9))
    return rows


def ask_sidebar(data) -> None:
    """Always-visible Ask box in the sidebar (no hunting on long pages)."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Ask me")
    cfg = active_model_config()
    st.sidebar.caption(
        "Works offline on demo alerts. Add a valid `nvapi-…` key above for live LangChain."
        if cfg.use_mock
        else f"Live · `{cfg.model_name}` (falls back to demo data if auth fails)"
    )
    history_key = "ask_history_sidebar"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    question = st.sidebar.text_area(
        "Your question",
        placeholder="e.g. Tell me about my critical flagged issues",
        height=80,
        key="ask_sidebar_question",
        label_visibility="collapsed",
    )
    if st.sidebar.button("Ask me", type="primary", use_container_width=True, key="ask_sidebar_btn"):
        q = (question or "").strip()
        if not q:
            st.sidebar.warning("Type a question first.")
        else:
            with st.spinner("Thinking…"):
                answer, payload = ask_assistant(
                    q,
                    config=cfg,
                    context={
                        "page": "sidebar",
                        "churn_alert_count": len(data.get("churn_alerts") or []),
                        "media_alert_count": len(data.get("media_alerts") or []),
                        "path_label": cfg.path_label,
                        "hosted": is_hosted_demo_environment(),
                        "flagged_issues": _flagged_issues_context(data),
                    },
                )
            st.session_state[history_key].append({"q": q, "a": answer, "trace": payload.get("langchain_trace")})
            st.rerun()

    for turn in reversed(st.session_state[history_key][-3:]):
        st.sidebar.markdown(f"**You:** {turn['q']}")
        st.sidebar.markdown(turn["a"])
        st.sidebar.markdown("---")


def _inline_ask_panel(data, *, page_key: str, hint: str = "Ask about this page or the backend…") -> None:
    """Visible Ask section on each content page (text box + button — not chat_input)."""
    st.markdown("---")
    st.subheader("Ask me")
    cfg = active_model_config()
    mode = "offline demo answers" if cfg.use_mock else f"live · {cfg.model_name}"
    st.caption(f"Path: `{cfg.path_label}` · {mode}")

    history_key = f"ask_history_{page_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    cols = st.columns([4, 1])
    with cols[0]:
        question = st.text_input(
            "Question",
            placeholder=hint,
            key=f"ask_text_{page_key}",
            label_visibility="collapsed",
        )
    with cols[1]:
        asked = st.button("Ask me", type="primary", use_container_width=True, key=f"ask_btn_{page_key}")

    if asked:
        q = (question or "").strip()
        if not q:
            st.warning("Type a question first.")
        else:
            with st.spinner("Thinking…"):
                answer, payload = ask_assistant(
                    q,
                    config=cfg,
                    context={
                        "page": page_key,
                        "churn_alert_count": len(data.get("churn_alerts") or []),
                        "media_alert_count": len(data.get("media_alerts") or []),
                        "path_label": cfg.path_label,
                        "hosted": is_hosted_demo_environment(),
                        "flagged_issues": _flagged_issues_context(data),
                    },
                )
            st.session_state[history_key].append(
                {"role": "user", "content": q}
            )
            st.session_state[history_key].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "trace": payload.get("langchain_trace"),
                }
            )
            st.rerun()

    for msg in st.session_state[history_key][-6:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("trace"):
                with st.expander("LangChain trace"):
                    _render_trace(msg["trace"])



def _render_trace(trace: dict | None) -> None:
    if not trace:
        st.caption("No LangChain trace for this run.")
        return
    st.markdown(
        f"**Run** `{trace.get('run_id')}` · `{trace.get('kind')}` · path `{trace.get('path_label')}`"
    )
    for step in trace.get("steps") or []:
        dur = step.get("duration_ms")
        dur_s = f" · {dur} ms" if dur is not None else ""
        st.markdown(f"- `{step.get('status')}` **{step.get('name')}** — {step.get('detail')}{dur_s}")
    with st.expander("Full trace JSON"):
        st.json(trace)


def synthetic_data_sidebar_body() -> None:
    """Demo-data controls (rendered inside the More options expander)."""
    meta = _read_generation_meta()
    if "demo_seed" not in st.session_state:
        st.session_state.demo_seed = int(meta.get("seed") or os.getenv("SEED", "42"))

    st.markdown("**Demo data**")
    if meta:
        st.caption(
            f"Seed `{meta.get('seed')}` · {meta.get('n_accounts')} accounts · "
            f"{meta.get('n_campaigns')} campaigns"
        )
    seed = st.number_input(
        "World seed",
        min_value=0,
        max_value=2_147_483_647,
        value=int(st.session_state.demo_seed),
        step=1,
    )
    st.session_state.demo_seed = int(seed)
    if st.button("Random seed", use_container_width=True):
        st.session_state.demo_seed = secrets.randbelow(1_000_000)
        st.rerun()
    n_accounts = st.number_input(
        "Accounts", min_value=20, max_value=200, value=int(meta.get("n_accounts") or 100)
    )
    n_campaigns = st.number_input(
        "Campaigns", min_value=15, max_value=80, value=int(meta.get("n_campaigns") or 40)
    )
    max_inv = st.slider("Investigations to run", min_value=5, max_value=40, value=25)
    if st.button("Regenerate world", type="primary", use_container_width=True):
        with st.spinner(
            f"Generating world (seed={st.session_state.demo_seed}) and re-running analysis..."
        ):
            regenerate_demo_world(
                seed=int(st.session_state.demo_seed),
                n_accounts=int(n_accounts),
                n_campaigns=int(n_campaigns),
                max_investigations=int(max_inv),
            )
        st.success(f"Ready (seed {st.session_state.demo_seed}).")
        st.rerun()


# Visual direction: cool slate + teal (avoid purple/cream AI clichés)
st.markdown(
    """
    <style>
      :root {
        --ss-bg: #0f1720;
        --ss-panel: #16202c;
        --ss-accent: #2dd4bf;
        --ss-text: #e8eef5;
        --ss-muted: #93a4b5;
      }
      .stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b3a3a 0%, #0f1720 45%, #0b1219 100%); color: var(--ss-text); }
      h1, h2, h3 { font-family: "IBM Plex Sans", "Segoe UI", sans-serif !important; letter-spacing: -0.02em; }
      div[data-testid="stMetric"] {
        background: rgba(22, 32, 44, 0.85);
        border: 1px solid rgba(45, 212, 191, 0.25);
        padding: 0.75rem 1rem;
        border-radius: 0.4rem;
      }
      .ss-banner {
        border-left: 3px solid #2dd4bf;
        padding: 0.6rem 0.9rem;
        background: rgba(22,32,44,0.7);
        margin-bottom: 1rem;
        color: #c9d6e3;
      }
      .ss-brief {
        background: rgba(22, 32, 44, 0.92);
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 0.5rem;
        padding: 1rem 1.1rem;
        margin-bottom: 0.85rem;
      }
      .ss-brief h4 {
        margin: 0 0 0.35rem 0;
        color: #2dd4bf;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }
      .ss-brief p, .ss-brief li {
        color: #e8eef5;
        font-size: 0.95rem;
        line-height: 1.45;
      }
      .ss-brief ul { margin: 0.25rem 0 0 1.1rem; padding: 0; }
      .ss-kicker {
        color: #93a4b5;
        font-size: 0.8rem;
        margin-bottom: 0.35rem;
      }
      .ss-story {
        background: linear-gradient(145deg, rgba(22,32,44,0.95), rgba(27,58,58,0.35));
        border: 1px solid rgba(45, 212, 191, 0.35);
        border-radius: 0.6rem;
        padding: 1.15rem 1.25rem 0.4rem 1.25rem;
        margin-bottom: 1rem;
      }
      .ss-story h2 {
        margin: 0.15rem 0 0.75rem 0 !important;
        font-size: 1.45rem !important;
        color: #e8eef5 !important;
      }
      .ss-chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 0.35rem;
        background: rgba(45, 212, 191, 0.15);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.35);
      }
      .ss-chip-warn {
        background: rgba(251, 146, 60, 0.15);
        color: #fdba74;
        border-color: rgba(251, 146, 60, 0.35);
      }
      .ss-chip-crit {
        background: rgba(248, 113, 113, 0.15);
        color: #fca5a5;
        border-color: rgba(248, 113, 113, 0.35);
      }
      div[data-testid="stSidebar"] button[kind="secondary"] {
        text-align: left;
      }
      .ss-brand-hint {
        color: #93a4b5;
        font-size: 0.75rem;
        margin: -0.4rem 0 0.8rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _load_outputs():
    return {
        "churn_alerts": _load_json(OUTPUTS_DIR / "churn_alerts.json", []),
        "media_alerts": _load_json(OUTPUTS_DIR / "media_alerts.json", []),
        "churn_investigations": _load_json(OUTPUTS_DIR / "churn_investigations.json", []),
        "media_investigations": _load_json(OUTPUTS_DIR / "media_investigations.json", []),
        "evaluation": _load_json(OUTPUTS_DIR / "evaluation.json", {}),
        "summary": _load_json(OUTPUTS_DIR / "run_summary.json", {}),
    }


@st.cache_data(show_spinner=False)
def _load_metrics():
    churn = pd.read_parquet(GENERATED_DIR / "churn_metrics.parquet") if (GENERATED_DIR / "churn_metrics.parquet").exists() else pd.DataFrame()
    media = pd.read_parquet(GENERATED_DIR / "media_metrics.parquet") if (GENERATED_DIR / "media_metrics.parquet").exists() else pd.DataFrame()
    churn_gt = pd.read_parquet(GROUND_TRUTH_DIR / "churn_labels.parquet") if (GROUND_TRUTH_DIR / "churn_labels.parquet").exists() else pd.DataFrame()
    media_gt = pd.read_parquet(GROUND_TRUTH_DIR / "media_labels.parquet") if (GROUND_TRUTH_DIR / "media_labels.parquet").exists() else pd.DataFrame()
    return churn, media, churn_gt, media_gt


def overview_page(data, show_eval: bool):
    _page_header("Home", "Flagged issues first — open a card or click a chart for detail.")
    st.markdown(
        '<div class="ss-banner">Early warning for churn and paid-media issues — '
        "plain English first, numbers underneath. Recommendations need a human.</div>",
        unsafe_allow_html=True,
    )

    churn_alerts = data["churn_alerts"]
    media_alerts = data["media_alerts"]
    active = len(churn_alerts) + len(media_alerts)

    c1, c2, c3 = st.columns(3)
    c1.metric("Things to review", active)
    c2.metric("Account risks", len(churn_alerts))
    c3.metric("Campaign issues", len(media_alerts))

    inv_churn = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data["churn_investigations"]
        if "alert" in row
    }
    inv_media = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row.get("investigation", {})
        for row in data["media_investigations"]
        if "alert" in row
    }

    issue_rows: list[dict] = []
    for a in churn_alerts:
        key = a["entity_id"] + "|" + a["alert_type"]
        brief = build_churn_briefing(a, inv_churn.get(key, {}))
        inv = inv_churn.get(key, {})
        issue_rows.append(
            {
                "domain": "churn",
                "kind": "Account",
                "entity_id": a.get("entity_id"),
                "alert_type": a.get("alert_type"),
                "severity": a.get("severity"),
                "headline": brief["headline"],
                "insight": brief.get("insight"),
                "opportunity": brief.get("opportunity"),
                "recommended_action": brief.get("recommended_action"),
                "next_steps": " · ".join(brief.get("next_steps") or []),
                "expected_impact": " · ".join(brief.get("expected_impact") or []),
                "risk_score": inv.get("risk_score"),
                "confidence": inv.get("confidence"),
                "current_value": a.get("current_value"),
                "expected_value": a.get("expected_value"),
                "window": brief.get("window"),
                "metrics": ", ".join(a.get("metrics_involved") or []),
                "_alert": a,
                "_brief": brief,
                "_sort": {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    str(a.get("severity")).lower(), 9
                ),
            }
        )
    for a in media_alerts:
        key = a["entity_id"] + "|" + a["alert_type"]
        brief = build_media_briefing(a, inv_media.get(key, {}))
        inv = inv_media.get(key, {})
        issue_rows.append(
            {
                "domain": "media",
                "kind": "Campaign",
                "entity_id": a.get("entity_id"),
                "alert_type": a.get("alert_type"),
                "severity": a.get("severity"),
                "headline": brief["headline"],
                "insight": brief.get("insight"),
                "opportunity": brief.get("opportunity"),
                "recommended_action": brief.get("recommended_action"),
                "next_steps": " · ".join(brief.get("next_steps") or []),
                "expected_impact": " · ".join(brief.get("expected_impact") or []),
                "risk_score": None,
                "confidence": inv.get("confidence"),
                "current_value": a.get("current_value"),
                "expected_value": a.get("expected_value"),
                "window": brief.get("window"),
                "metrics": ", ".join(a.get("metrics_involved") or []),
                "_alert": a,
                "_brief": brief,
                "_sort": {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    str(a.get("severity")).lower(), 9
                ),
            }
        )
    issue_rows.sort(key=lambda r: (r["_sort"], -(r.get("risk_score") or 0)))

    if not issue_rows:
        st.info("No alerts yet — use Regenerate in the sidebar Advanced section if needed.")
        return

    issues_df = pd.DataFrame(
        [{k: v for k, v in row.items() if not k.startswith("_")} for row in issue_rows]
    )

    st.subheader("Needs attention")
    st.caption("Expand a card for the full brief, or click a chart bar below to filter the same list.")

    # Quick picker for one deep dive
    labels = [
        f"{r['kind']} · {r['entity_id']} — {r['headline']} ({r['severity']})" for r in issue_rows
    ]
    pick = st.selectbox("Inspect a flagged issue", ["(browse cards below)"] + labels, index=0)
    if pick != "(browse cards below)":
        chosen = issue_rows[labels.index(pick)]
        _render_story_summary(chosen["_brief"])
        with st.expander("Detector source for this issue", expanded=True):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "entity_id": chosen["entity_id"],
                            "domain": chosen["domain"],
                            "alert_type": chosen["alert_type"],
                            "severity": chosen["severity"],
                            "current_value": chosen["current_value"],
                            "expected_value": chosen["expected_value"],
                            "metrics": chosen["metrics"],
                            "window": chosen["window"],
                            "risk_score": chosen.get("risk_score"),
                            "confidence": chosen.get("confidence"),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.json(chosen["_alert"])

    # Flagged cards — open for detail
    for row in issue_rows[:8]:
        title = f"{row['kind']} · {row['entity_id']} · {row['severity']} — {row['headline']}"
        with st.expander(title, expanded=False):
            _render_story_summary(row["_brief"])
            st.markdown("**Source snapshot**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "entity_id": row["entity_id"],
                            "alert_type": row["alert_type"],
                            "severity": row["severity"],
                            "current_value": row["current_value"],
                            "expected_value": row["expected_value"],
                            "metrics": row["metrics"],
                            "window": row["window"],
                            "risk_score": row.get("risk_score"),
                            "confidence": row.get("confidence"),
                        }
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("Raw detector payload"):
                st.json(row["_alert"])

    st.subheader("Charts & distribution")
    st.caption("Click a bar to see the flagged issues in that bucket.")

    # Severity chart as explicit grouped bars (cleaner click → filter than histogram bins)
    sev_counts = (
        issues_df.groupby(["severity", "domain"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    if not sev_counts.empty:
        fig = px.bar(
            sev_counts,
            x="severity",
            y="count",
            color="domain",
            barmode="group",
            title="Alerts by severity",
            color_discrete_sequence=["#2dd4bf", "#38bdf8"],
            custom_data=["domain", "severity"],
        )
        event = st.plotly_chart(
            _style_chart(fig),
            use_container_width=True,
            on_select="rerun",
            selection_mode=("points", "box"),
            key="overview_severity_chart",
        )
        points = _selection_points(event)
        if points:
            matched = []
            for pt in points:
                domain = None
                severity = pt.get("x")
                custom = pt.get("customdata")
                if isinstance(custom, (list, tuple)) and len(custom) >= 2:
                    domain, severity = custom[0], custom[1]
                else:
                    domain = pt.get("legendgroup") or pt.get("curve_number")
                    # Map curve name if legendgroup is domain label
                    if domain not in {"churn", "media"}:
                        domain = pt.get("legendgroup")
                subset = issues_df.copy()
                if severity is not None:
                    subset = subset[subset["severity"].astype(str) == str(severity)]
                if domain in {"churn", "media"}:
                    subset = subset[subset["domain"] == domain]
                matched.append(subset)
            if matched:
                detail = pd.concat(matched).drop_duplicates(
                    subset=["domain", "entity_id", "alert_type"]
                )
                st.markdown("**Issues in selected severity bar**")
                st.dataframe(
                    detail[
                        [
                            "kind",
                            "entity_id",
                            "severity",
                            "headline",
                            "recommended_action",
                            "insight",
                            "next_steps",
                            "expected_impact",
                            "current_value",
                            "expected_value",
                            "window",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    # Risk score — bar by rounded score so clicks map cleanly
    risk_df = issues_df[issues_df["risk_score"].notna()].copy()
    if not risk_df.empty:
        risk_df["risk_bucket"] = risk_df["risk_score"].round(0).astype(int)
        risk_counts = risk_df.groupby("risk_bucket", as_index=False).size().rename(
            columns={"size": "count"}
        )
        fig2 = px.bar(
            risk_counts,
            x="risk_bucket",
            y="count",
            title="Churn risk score distribution",
            color_discrete_sequence=["#2dd4bf"],
            custom_data=["risk_bucket"],
        )
        fig2.update_layout(xaxis_title="risk_score")
        event2 = st.plotly_chart(
            _style_chart(fig2),
            use_container_width=True,
            on_select="rerun",
            selection_mode=("points", "box"),
            key="overview_risk_chart",
        )
        points2 = _selection_points(event2)
        if points2:
            buckets = set()
            for pt in points2:
                custom = pt.get("customdata")
                if isinstance(custom, (list, tuple)) and custom:
                    buckets.add(int(custom[0]))
                elif pt.get("x") is not None:
                    try:
                        buckets.add(int(round(float(pt["x"]))))
                    except (TypeError, ValueError):
                        pass
            if buckets:
                detail = risk_df[risk_df["risk_bucket"].isin(buckets)]
                st.markdown(f"**Accounts with risk score in {sorted(buckets)}**")
                st.dataframe(
                    detail[
                        [
                            "entity_id",
                            "severity",
                            "risk_score",
                            "headline",
                            "recommended_action",
                            "insight",
                            "next_steps",
                            "expected_impact",
                            "window",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    if show_eval and data["evaluation"]:
        with st.expander("Evaluation vs ground truth", expanded=False):
            st.json(data["evaluation"])

    _inline_ask_panel(
        data,
        page_key="overview",
        hint="e.g. What should I look at first on Overview?",
    )


def _severity_chip(severity: str) -> str:
    sev = str(severity or "unknown").lower()
    cls = "ss-chip"
    if sev in {"high", "medium"}:
        cls = "ss-chip ss-chip-warn"
    if sev == "critical":
        cls = "ss-chip ss-chip-crit"
    return f'<span class="{cls}">{sev}</span>'


def _brief_block(title: str, body: str) -> None:
    st.markdown(
        f'<div class="ss-brief"><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def _brief_list(title: str, items: list) -> None:
    if not items:
        return
    lis = "".join(f"<li>{item}</li>" for item in items)
    st.markdown(
        f'<div class="ss-brief"><h4>{title}</h4><ul>{lis}</ul></div>',
        unsafe_allow_html=True,
    )


def _render_story_summary(brief: dict) -> None:
    """Plain-English story first — what a CSM would say in a meeting."""
    chips = (
        f'{_severity_chip(brief.get("severity"))}'
        f'<span class="ss-chip">{brief.get("entity_id")}</span>'
        f'<span class="ss-chip">{brief.get("window")}</span>'
    )
    st.markdown(
        f'<div class="ss-story">{chips}<h2>{brief.get("headline")}</h2></div>',
        unsafe_allow_html=True,
    )
    _brief_block("What's going on", brief.get("insight") or "")
    if brief.get("opportunity"):
        _brief_block("Why it matters", brief.get("opportunity") or "")
    _brief_block("Recommended action", brief.get("recommended_action") or "")
    cols = st.columns(2)
    with cols[0]:
        _brief_list("Next steps", brief.get("next_steps") or [])
    with cols[1]:
        _brief_list("Expected impact", brief.get("expected_impact") or [])
    if brief.get("immediate_review"):
        st.error("Flag for immediate human review — advisory only; nothing is changed automatically.")


def _render_technical_section(
    *,
    brief: dict,
    alert: dict,
    chart_factory,
    extra_expanders: list | None = None,
) -> None:
    """Mechanics below the fold: chart math, detector payload, hypotheses."""
    st.markdown("---")
    with st.expander("Numbers & chart — click a point for source", expanded=True):
        st.caption("Detector math and the trend behind the story.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Current", f"{float(alert.get('current_value') or 0):.2f}")
        c2.metric("Expected", f"{float(alert.get('expected_value') or 0):.2f}")
        c3.metric("Change", brief.get("what_changed") or "—")
        chart_factory()

    with st.expander("How the detector decided this", expanded=False):
        st.write(
            {
                "alert_type": alert.get("alert_type"),
                "metrics_involved": alert.get("metrics_involved"),
                "supporting_calculations": alert.get("supporting_calculations"),
                "window": brief.get("window"),
            }
        )
        st.json(alert)

    with st.expander("Hypotheses, metrics & confidence", expanded=False):
        _brief_list("Diagnosis (hypotheses)", brief.get("diagnosis") or [])
        _brief_list("Success metrics", brief.get("success_metrics") or [])
        _brief_list("Guardrails", brief.get("guardrails") or [])
        _brief_list("Evidence", brief.get("evidence") or [])
        meta = []
        if brief.get("confidence") is not None:
            meta.append(f"Confidence {float(brief['confidence']):.0%}")
        if brief.get("risk_score") is not None:
            meta.append(f"Risk score {brief['risk_score']}")
        if meta:
            st.caption(" · ".join(meta))
        if brief.get("data_limitations"):
            st.markdown("**Data confidence**")
            for lim in brief["data_limitations"]:
                st.write(f"- {lim}")

    for title, render_fn in extra_expanders or []:
        with st.expander(title, expanded=False):
            render_fn()


def _pick_alert(alerts: list[dict], inv_map: dict, builder, *, label: str) -> tuple[dict, dict, dict]:
    """One human-readable dropdown + optional severity filter."""
    severities = sorted({str(a.get("severity", "unknown")) for a in alerts})
    filt = st.selectbox(
        "Show severity",
        options=["All"] + severities,
        index=0,
        help="Narrow the list when there are many alerts.",
    )
    filtered = alerts if filt == "All" else [a for a in alerts if str(a.get("severity")) == filt]
    if not filtered:
        filtered = alerts

    options = []
    for a in filtered:
        key = a["entity_id"] + "|" + a["alert_type"]
        inv = inv_map.get(key, {}).get("investigation", {})
        brief = builder(a, inv)
        sev = a.get("severity", "?")
        options.append(
            {
                "key": key,
                "label": f"{brief['headline']} · {a['entity_id']} ({sev})",
                "alert": a,
                "inv": inv,
                "brief": brief,
                "sort_sev": {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(str(sev).lower(), 9),
                "sort_risk": -(inv.get("risk_score") or 0),
            }
        )
    options.sort(key=lambda o: (o["sort_sev"], o["sort_risk"], o["label"]))
    labels = [o["label"] for o in options]
    chosen = st.selectbox(label, labels, index=0)
    row = options[labels.index(chosen)]
    return row["alert"], row["inv"], row["brief"]


def _style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef5",
        legend_title_text="",
        margin=dict(l=10, r=10, t=40, b=10),
        height=360,
        clickmode="event+select",
        dragmode="select",
    )
    return fig


def _add_anomaly_window(fig, start, end, label: str = "Flagged window"):
    if not start or not end:
        return fig
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor="rgba(45, 212, 191, 0.12)",
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
        annotation_font_color="#93a4b5",
    )
    return fig


def _selection_points(event) -> list[dict]:
    if event is None:
        return []
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection") or event.get("select")
    if selection is None:
        return []
    points = getattr(selection, "points", None)
    if points is None and isinstance(selection, dict):
        points = selection.get("points")
    return list(points or [])


def _match_source_rows(
    source_df: pd.DataFrame, points: list[dict], *, date_col: str = "date"
) -> pd.DataFrame:
    """Map Plotly selection points back to underlying dataframe rows."""
    if source_df.empty or not points:
        return pd.DataFrame()

    work = source_df.copy().reset_index(drop=True)
    if date_col in work.columns and date_col.lower() in {"date", "day", "week", "timestamp"}:
        work["_date_key"] = pd.to_datetime(work[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    matched_idx: set[int] = set()
    for pt in points:
        custom = pt.get("customdata")
        if isinstance(custom, (list, tuple)) and custom:
            try:
                matched_idx.add(int(custom[0]))
                continue
            except (TypeError, ValueError):
                pass

        x_val = pt.get("x")
        if x_val is not None and "_date_key" in work.columns:
            try:
                x_key = pd.to_datetime(x_val).strftime("%Y-%m-%d")
            except (TypeError, ValueError):
                x_key = str(x_val)[:10]
            hits = work.index[work["_date_key"] == x_key].tolist()
            if hits:
                matched_idx.update(int(i) for i in hits)
                continue

        if x_val is not None and date_col in work.columns:
            hits = work.index[work[date_col].astype(str) == str(x_val)].tolist()
            if hits:
                matched_idx.update(int(i) for i in hits)
                continue

        if x_val is not None:
            for col in work.columns:
                if work[col].dtype == object or str(work[col].dtype).startswith("string"):
                    hits = work.index[work[col].astype(str) == str(x_val)].tolist()
                    if hits:
                        matched_idx.update(int(i) for i in hits)
                        break

    if not matched_idx:
        return pd.DataFrame()
    out = work.loc[sorted(matched_idx)].drop(columns=["_date_key"], errors="ignore")
    return out.reset_index(drop=True)


def _attach_row_index_customdata(fig, n_rows: int) -> None:
    """Stamp each trace point with its dataframe row index for click→source mapping."""
    for trace in fig.data:
        if getattr(trace, "x", None) is None:
            continue
        n = len(trace.x)
        trace.customdata = [[i] for i in range(min(n, n_rows))]
        if not getattr(trace, "hovertemplate", None):
            trace.hovertemplate = (
                "%{x}<br>%{y}<br>row=%{customdata[0]}<extra>%{fullData.name}</extra>"
            )


def _plotly_with_source(
    fig,
    source_df: pd.DataFrame,
    *,
    key: str,
    date_col: str = "date",
    source_label: str = "Source rows for selection",
) -> None:
    """Render a Plotly chart; clicking/selecting points reveals underlying source data."""
    st.caption("Click a point (or box/lasso-select) to inspect the source data behind it.")
    _attach_row_index_customdata(fig, len(source_df))
    event = st.plotly_chart(
        _style_chart(fig),
        use_container_width=True,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
        key=key,
    )
    points = _selection_points(event)
    if not points:
        return

    st.markdown(f"**{source_label}**")
    clicked = [
        {
            "x": pt.get("x"),
            "y": pt.get("y"),
            "series": pt.get("legendgroup") or pt.get("curve_number"),
        }
        for pt in points[:12]
    ]
    st.dataframe(pd.DataFrame(clicked), use_container_width=True, hide_index=True)

    rows = _match_source_rows(source_df, points, date_col=date_col)
    if rows.empty:
        st.info("Could not map that click to a source row — try another point.")
        with st.expander("Raw selection payload"):
            st.json(points)
        return

    st.dataframe(rows, use_container_width=True, hide_index=True)
    with st.expander("Raw selection payload"):
        st.json(points)


def churn_page(data, churn_df, churn_gt, show_eval: bool):
    _page_header("Account risks", "Pick a risk → read the story → open numbers only if you need them.")

    alerts = data["churn_alerts"]
    if not alerts:
        st.info("No churn alerts yet. Use Regenerate demo data under Advanced in the sidebar.")
        return

    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["churn_investigations"]
        if "alert" in row
    }
    alert, _inv, brief = _pick_alert(
        alerts, inv_map, build_churn_briefing, label="Pick a risk"
    )
    account = alert["entity_id"]

    _render_story_summary(brief)

    def _chart():
        if churn_df.empty:
            st.info("No metric history loaded.")
            return
        hist = churn_df[churn_df["account_id"] == account].sort_values("date").reset_index(drop=True)
        metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
        if not metrics:
            metrics = [
                c
                for c in ["weekly_sessions", "active_users", "key_feature_adoption", "nps_score"]
                if c in hist.columns
            ]
        fig = px.line(hist, x="date", y=metrics, title="Metric trend (flagged window shaded)")
        fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
        _plotly_with_source(
            fig,
            hist,
            key=f"churn_trend_{account}_{alert.get('alert_type')}",
            source_label=f"Source metric rows · {account}",
        )

    extras = []
    if show_eval and not churn_gt.empty:
        extras.append(
            (
                "Evaluation ground truth",
                lambda: st.dataframe(
                    churn_gt[churn_gt["account_id"] == account], use_container_width=True
                ),
            )
        )

    _render_technical_section(
        brief=brief,
        alert=alert,
        chart_factory=_chart,
        extra_expanders=extras,
    )
    _inline_ask_panel(
        data,
        page_key="churn",
        hint="e.g. Why does gradual usage decline matter?",
    )


def media_page(data, media_df, media_gt, show_eval: bool):
    _page_header("Campaign issues", "Pick an issue → read the story → charts/source underneath.")

    alerts = data["media_alerts"]
    if not alerts:
        st.info("No campaign alerts yet. Use Regenerate demo data under Advanced in the sidebar.")
        return

    inv_map = {
        row["alert"]["entity_id"] + "|" + row["alert"]["alert_type"]: row
        for row in data["media_investigations"]
        if "alert" in row
    }
    alert, _inv, brief = _pick_alert(
        alerts, inv_map, build_media_briefing, label="Pick an issue"
    )
    campaign = alert["entity_id"]

    _render_story_summary(brief)

    def _chart():
        if media_df.empty:
            st.info("No metric history loaded.")
            return
        hist = media_df[media_df["campaign_id"] == campaign].sort_values("date").reset_index(drop=True)
        metrics = [m for m in (alert.get("metrics_involved") or []) if m in hist.columns]
        if not metrics:
            metrics = [
                c
                for c in ["spend", "conversions", "cpc", "conversion_rate", "frequency"]
                if c in hist.columns
            ]
        fig = px.line(
            hist, x="date", y=metrics[:4], title="Campaign trend (flagged window shaded)"
        )
        fig = _add_anomaly_window(fig, alert.get("start_date"), alert.get("end_date"))
        _plotly_with_source(
            fig,
            hist,
            key=f"media_trend_{campaign}_{alert.get('alert_type')}",
            source_label=f"Source daily rows · {campaign}",
        )

        primary = (alert.get("metrics_involved") or ["value"])[0]
        cmp = pd.DataFrame(
            {
                "series": ["Expected", "Current"],
                "value": [float(alert["expected_value"]), float(alert["current_value"])],
                "metric": [primary, primary],
                "campaign_id": [campaign, campaign],
                "alert_type": [alert.get("alert_type"), alert.get("alert_type")],
                "window_start": [alert.get("start_date"), alert.get("start_date")],
                "window_end": [alert.get("end_date"), alert.get("end_date")],
                "supporting_calculations": [
                    str(alert.get("supporting_calculations")),
                    str(alert.get("supporting_calculations")),
                ],
            }
        )
        fig2 = px.bar(
            cmp,
            x="series",
            y="value",
            title=f"{primary.replace('_', ' ').title()} — expected vs current",
            color="series",
            color_discrete_sequence=["#38bdf8", "#2dd4bf"],
        )
        _plotly_with_source(
            fig2,
            cmp,
            key=f"media_bar_{campaign}_{alert.get('alert_type')}",
            date_col="series",
            source_label="Source for expected vs current",
        )

    extras = []
    if show_eval and not media_gt.empty:
        extras.append(
            (
                "Evaluation ground truth",
                lambda: st.dataframe(
                    media_gt[media_gt["campaign_id"] == campaign], use_container_width=True
                ),
            )
        )

    _render_technical_section(
        brief=brief,
        alert=alert,
        chart_factory=_chart,
        extra_expanders=extras,
    )
    _inline_ask_panel(
        data,
        page_key="media",
        hint="e.g. What does creative fatigue mean for spend?",
    )


def structure_page(data):
    """Portfolio explainer: full product structure in plain English + optional live traces."""
    _page_header(
        "How it works",
        "System map for SignalSentry — LangChain, NemoClaw, OpenClaw, and this demo.",
    )
    st.markdown(
        '<div class="ss-banner">What the product does and how the pieces connect. '
        "Optional live traces are at the bottom.</div>",
        unsafe_allow_html=True,
    )
    cfg = active_model_config()

    st.subheader("In one sentence")
    st.markdown(
        "SignalSentry builds **synthetic** churn and paid-media data, finds anomalies with "
        "**deterministic Python detectors**, then uses **LangChain** to write structured "
        "investigation briefs — locally via **NemoClaw** (`inference.local`), or on this "
        "hosted demo via **mock** / optional **BYOK**."
    )

    st.subheader("End-to-end flow")
    st.code(
        "1  Synthetic generators   (seeded fake accounts + campaigns + ground-truth labels)\n"
        "2  Detectors              (YAML thresholds, baselines, z-scores — no LLM)\n"
        "3  Candidate alerts       (entity, window, current vs expected, metrics)\n"
        "4  LangChain investigator (ChatOpenAI + with_structured_output → Pydantic)\n"
        "5  Inference route        (NemoClaw local  |  NVIDIA BYOK  |  mock templates)\n"
        "6  Dashboard briefs       (plain English → next steps → impact; charts + source)",
        language="text",
    )

    st.subheader("Layers")
    layers = pd.DataFrame(
        [
            {
                "Layer": "Generation",
                "Code": "src/generation/",
                "Does": "Creates reproducible synthetic metrics and injected anomaly labels",
                "LLM?": "No",
            },
            {
                "Layer": "Detection",
                "Code": "src/detection/",
                "Does": "Flags unusual movement vs each entity’s own baseline",
                "LLM?": "No",
            },
            {
                "Layer": "Investigation",
                "Code": "src/agents/",
                "Does": "Turns an alert JSON into a structured CSM/media brief",
                "LLM?": "Yes (or mock)",
            },
            {
                "Layer": "Presentation",
                "Code": "src/presentation/ + app.py",
                "Does": "Plain-English cards, charts, click-to-source, Overview drill-downs",
                "LLM?": "No",
            },
            {
                "Layer": "Evaluation",
                "Code": "src/evaluation/",
                "Does": "Precision/recall vs synthetic ground truth (demo toggle)",
                "LLM?": "No",
            },
            {
                "Layer": "Privacy / audit",
                "Code": "src/privacy/",
                "Does": "Payload preview, investigation log, file inventory",
                "LLM?": "No",
            },
        ]
    )
    st.dataframe(layers, use_container_width=True, hide_index=True)

    st.subheader("What each app page is for")
    pages = pd.DataFrame(
        [
            {"Page": "Home", "Purpose": "Flagged issues at a glance; click charts/cards for detail"},
            {"Page": "Account risks", "Purpose": "One churn alert → English brief + chart/source"},
            {"Page": "Campaign issues", "Purpose": "One media anomaly → English brief + chart/source"},
            {"Page": "How it works", "Purpose": "Full system map (this page) + optional live traces"},
            {"Page": "Privacy", "Purpose": "Synthetic-only confirmation, files read, audit log"},
            {"Page": "Ask me", "Purpose": "Always in the sidebar + at the bottom of each page"},
        ]
    )
    st.dataframe(pages, use_container_width=True, hide_index=True)

    st.subheader("LangChain · NemoClaw · OpenClaw")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div class="ss-brief"><h4>LangChain</h4><p>App library that calls an '
            "OpenAI-compatible chat API and forces a Pydantic investigation schema "
            "(<code>with_structured_output</code>). Same code path in every mode.</p></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="ss-brief"><h4>NemoClaw</h4><p>Local sandbox that exposes '
            "<code>https://inference.local/v1</code> to your host model (e.g. Nemotron). "
            "This is the intended <b>live</b> backend when you run SignalSentry inside the sandbox.</p></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div class="ss-brief"><h4>OpenClaw</h4><p>Local agent/chat gateway UI '
            "(<code>127.0.0.1:18789</code>). Great for sandbox chatting; <b>not</b> embedded on "
            "Streamlit Cloud because it only exists on your machine.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
| Where you run | Investigation path |
| --- | --- |
| This Streamlit Cloud site (default) | **Mock** templates — no API spend |
| Cloud + sidebar NVIDIA key (BYOK) | **Live LangChain** → `integrate.api.nvidia.com` |
| Local / inside NemoClaw sandbox | **Live LangChain** → `inference.local` (NemoClaw) |
"""
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Active path now", cfg.path_label)
    m2.metric("Model", cfg.model_name)
    m3.metric("Mock?", str(cfg.use_mock))

    st.subheader("Data contract (what the LLM is allowed to see)")
    st.markdown(
        """
1. **Detector alert** — entity id, alert type, severity, date window, current vs expected, supporting math  
2. **Optional metric context** — last few synthetic rows for that entity  
3. **System rules** — only use supplied JSON; label hypotheses; advisory recommendations only  
4. **Structured response** — evidence, likely causes, recommended action, confidence, limitations  

The model never gets live CRM/ad-platform credentials. This demo is **synthetic-only**.
"""
    )

    st.subheader("Repo map")
    st.code(
        "signalsentry/\n"
        "  app.py                 Streamlit UI\n"
        "  config/thresholds.yaml Detector knobs\n"
        "  src/generation/        Synthetic worlds + labels\n"
        "  src/detection/         Anomaly detectors\n"
        "  src/agents/            LangChain + mock investigators + traces\n"
        "  src/presentation/      Plain-English briefing copy\n"
        "  src/evaluation/        Score vs ground truth\n"
        "  src/privacy/           Audit + payload preview\n"
        "  data/generated/        Metrics parquet/csv\n"
        "  data/ground_truth/     Injected labels\n"
        "  data/outputs/          Alerts, investigations, audit log",
        language="text",
    )

    with st.expander("Try a live investigation + LangChain trace", expanded=False):
        st.caption("Runs the same investigator path the dashboard uses.")
        churn_alerts = data.get("churn_alerts") or []
        if not churn_alerts:
            st.info("No alerts yet — generate demo data first.")
        else:
            labels = [f"{a.get('entity_id')} · {a.get('alert_type')}" for a in churn_alerts[:20]]
            pick = st.selectbox("Alert to investigate", labels, key="structure_alert_pick")
            if st.button("Run LangChain investigator", type="primary"):
                alert_raw = churn_alerts[labels.index(pick)]
                alert = CandidateAlert(
                    entity_id=alert_raw["entity_id"],
                    start_date=alert_raw["start_date"],
                    end_date=alert_raw["end_date"],
                    alert_type=alert_raw["alert_type"],
                    severity=Severity(alert_raw["severity"]),
                    metrics_involved=alert_raw.get("metrics_involved") or [],
                    current_value=float(alert_raw.get("current_value") or 0),
                    expected_value=float(alert_raw.get("expected_value") or 0),
                    supporting_calculations=alert_raw.get("supporting_calculations") or {},
                    domain=alert_raw.get("domain") or "churn",
                )
                try:
                    with st.spinner("Investigating…"):
                        result, preview = investigate_churn(alert, config=active_model_config())
                    st.success("Done.")
                    st.json(result.model_dump(mode="json"))
                    with st.expander("Inference payload"):
                        st.json(preview)
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"Live investigation failed: {exc}\n\n"
                        "Try another model in Live LangChain (sidebar), or clear the key to use mock."
                    )

        st.markdown("**Recent traces**")
        log_rows = read_investigation_log(limit=30)
        traced = [r for r in reversed(log_rows) if r.get("langchain_trace")]
        if not traced:
            st.info("No traces yet. Run an investigation above or use Ask about this.")
        else:
            for row in traced[:8]:
                with st.expander(
                    f"{row.get('timestamp', '')} · {row.get('domain')} · "
                    f"{row.get('entity_id')} · {row.get('mode')}"
                ):
                    _render_trace(row.get("langchain_trace"))
                    if row.get("payload_preview"):
                        st.markdown("**Payload preview**")
                        st.json(row["payload_preview"])

    _inline_ask_panel(
        data,
        page_key="structure",
        hint="e.g. How does LangChain talk to NemoClaw?",
    )


def privacy_page():
    _page_header("Privacy", "Synthetic data only — what the app reads and logs.")
    cfg = active_model_config()
    confirm = synthetic_data_confirmation()
    st.success(confirm["message"])

    if cfg.use_mock:
        st.warning(
            "Currently in **mock** mode (no LangChain model calls). "
            "Paste a NVIDIA API key in the sidebar (BYOK) for live LangChain, "
            "or run locally inside NemoClaw with `USE_MOCK_MODEL=false`."
        )
    else:
        st.info(
            f"**Live LangChain** → `{cfg.base_url}` · model `{cfg.model_name}` · path `{cfg.path_label}` "
            f"({'NemoClaw route' if cfg.is_nemoclaw_route else 'OpenAI-compatible / BYOK'})."
        )

    st.write(
        {
            "inference_destination": cfg.destination_label,
            "model_name": cfg.model_name,
            "mode": "mock" if cfg.use_mock else "live-langchain",
            "path_label": cfg.path_label,
            "nemoclaw_route": cfg.is_nemoclaw_route,
            "synthetic_only": confirm["synthetic_only"],
            "byok_session": bool(st.session_state.get("byok_api_key")),
        }
    )
    st.subheader("Files the application reads")
    st.dataframe(pd.DataFrame(list_readable_files()), use_container_width=True)

    st.subheader("Investigation audit log")
    log_rows = read_investigation_log()
    if log_rows:
        st.dataframe(pd.DataFrame([{k: v for k, v in row.items() if k != "payload_preview" and k != "result"} for row in log_rows]), use_container_width=True)
        st.subheader("Exact inference payload preview (latest)")
        st.json(log_rows[-1].get("payload_preview", {}))
    else:
        st.info("No investigations logged yet.")

    st.caption("SignalSentry never sends outbound customer messages or applies automatic campaign changes.")


def main():
    _ensure_nav_state()
    if "byok_api_key" not in st.session_state:
        st.session_state.byok_api_key = ""
    if "byok_model" not in st.session_state:
        st.session_state.byok_model = PUBLIC_MODEL_CHOICES[0]

    # Brand = Home
    if st.sidebar.button("SignalSentry", use_container_width=True, key="brand_home"):
        _go_home()
        st.rerun()
    st.sidebar.markdown(
        '<p class="ss-brand-hint">Click the name to return Home</p>',
        unsafe_allow_html=True,
    )

    cfg = active_model_config()
    st.sidebar.caption("Mock demo" if cfg.use_mock else f"Live · {cfg.model_name}")

    page = st.sidebar.radio(
        "Go to",
        list(NAV_PAGES),
        key="nav_radio",
    )
    st.session_state.nav_page = page

    byok_sidebar()
    with st.sidebar.expander("More options", expanded=False):
        show_eval = st.toggle("Show ground truth", value=False)
        synthetic_data_sidebar_body()

    ensure_demo_data()

    if not (OUTPUTS_DIR / "churn_alerts.json").exists():
        st.warning("No analysis outputs found yet.")
        st.code("python -m src.generation.generate_all\npython -m src.run_analysis", language="bash")

    data = _load_outputs()
    ask_sidebar(data)
    churn_df, media_df, churn_gt, media_gt = _load_metrics()

    page = st.session_state.nav_page
    if page == "Home":
        overview_page(data, show_eval)
    elif page == "Account risks":
        churn_page(data, churn_df, churn_gt, show_eval)
    elif page == "Campaign issues":
        media_page(data, media_df, media_gt, show_eval)
    elif page == "How it works":
        structure_page(data)
    else:
        privacy_page()


if __name__ == "__main__":
    main()
