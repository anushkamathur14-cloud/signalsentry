# SignalSentry

Local-first Python application for **customer churn early warning** and **paid-media campaign anomaly investigation**.

**Primary path:** deterministic Python detectors → **live LangChain investigators** through NVIDIA **NemoClaw** (`https://inference.local/v1`) → Nemotron / OpenAI-compatible backend on the host.

Mock investigators exist only as an offline fallback (CI / Streamlit Community Cloud, where `inference.local` is unreachable).

All data is **synthetic**. Recommendations are **advisory** and require human review.

## Architecture

```text
Synthetic generators → Parquet/CSV + ground-truth labels
        ↓
Deterministic detectors (YAML thresholds)     ← no LLM for numbers
        ↓
Candidate alerts
        ↓
LangChain ChatOpenAI → NemoClaw inference.local → host Nemotron
        ↓
Pydantic investigation reports + Streamlit dashboard
```

| Layer | Role |
| --- | --- |
| `src/generation/` | Seeded churn + media datasets and labeled injections |
| `src/detection/` | Rolling baselines, z-scores, slopes, consistency checks |
| `src/agents/` | **Live LangChain** investigators (+ mock fallback) |
| `src/models/` | Pydantic schemas + NemoClaw-compatible LLM client |
| `src/evaluation/` | Precision / recall / F1 / FPR |
| `src/privacy/` | File inventory, audit log, inference payload preview |
| `src/verify_inference.py` | Live connectivity smoke test |
| `app.py` | Streamlit UI |

## Live NemoClaw + LangChain (intended use)

1. Complete NemoClaw host onboarding so sandbox traffic to `inference.local` forwards to your Nemotron / compatible model.
2. In the project (or sandbox copy):

```bash
cp .env.example .env
# .env.example already sets live mode:
# USE_MOCK_MODEL=false
# MODEL_BASE_URL=https://inference.local/v1
# MODEL_API_KEY=nemoclaw-local-placeholder
# MODEL_NAME=nvidia/nemotron-mini
```

3. Run:

```bash
pip install -e ".[dev]"
python -m src.generation.generate_all
python -m src.verify_inference    # pings LangChain → inference.local
python -m src.run_analysis        # live investigations
streamlit run app.py
```

LangChain uses `ChatOpenAI(base_url=https://inference.local/v1)` + `with_structured_output` (Pydantic). The app never calls public provider domains directly; credentials stay on the NemoClaw/OpenShell host.

If structured `json_schema` fails on your model build, set:

```env
STRUCTURED_OUTPUT_METHOD=json_mode
```

## Offline / Cloud fallback

```env
USE_MOCK_MODEL=true
```

Use this for pytest and [Streamlit Community Cloud](https://share.streamlit.io) demos. Cloud cannot reach `inference.local`.

## Quick start (local package)

Requires **Python 3.9+** (3.11+ preferred).

```bash
cd ~/Projects/signalsentry
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env

python -m src.generation.generate_all
python -m src.run_analysis
streamlit run app.py
pytest
```

### Streamlit Community Cloud demo

1. [share.streamlit.io](https://share.streamlit.io) → New app → `anushkamathur14-cloud/signalsentry` → `app.py`
2. First load auto-generates synthetic data with **mock** investigators
3. For a live NemoClaw showcase, run Streamlit **inside** the NemoClaw environment instead of Community Cloud

## Model configuration

| Variable | Purpose |
| --- | --- |
| `MODEL_BASE_URL` | OpenAI-compatible base URL (`https://inference.local/v1` for NemoClaw) |
| `MODEL_API_KEY` | Placeholder for the managed route (not a cloud provider secret) |
| `MODEL_NAME` | Model id (e.g. `nvidia/nemotron-mini`) |
| `USE_MOCK_MODEL` | `false` = live LangChain; `true` = offline mock |
| `STRUCTURED_OUTPUT_METHOD` | `json_schema` (default) or `json_mode` |
| `SEED` | Synthetic data seed |

## Commands

```bash
python -m src.generation.generate_all
python -m src.verify_inference
python -m src.run_analysis
streamlit run app.py
pytest
```

## Dashboard

- **Overview** — alert counts, severity mix, risk distribution, evaluation summary
- **Customer Churn Risks** — ranked accounts, trends, agent explanation, CSM action
- **Campaign Anomalies** — ranked alerts, actual vs expected, time series, recommendations
- **Privacy and Safety** — files read, inference destination, mock/live flag, audit log, payload preview
- Sidebar **evaluation toggle** reveals synthetic ground-truth labels

## Future Vercel showcase

Streamlit is the supported UI. Vercel cannot host Streamlit directly. After each analysis run, `data/outputs/showcase/` contains a stable JSON bundle you can later render with a static or Next.js site.

## Project layout

```text
signalsentry/
├── app.py
├── README.md
├── pyproject.toml
├── .env.example
├── config/thresholds.yaml
├── data/
│   ├── generated/
│   ├── ground_truth/
│   └── outputs/showcase/
├── src/
│   ├── generation/
│   ├── detection/
│   ├── agents/
│   ├── evaluation/
│   ├── models/
│   ├── privacy/
│   └── run_analysis.py
└── tests/
```

## License

Sample / demo project — synthetic data only.
