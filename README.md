# SignalSentry

Local-first Python application for **customer churn early warning** and **paid-media campaign anomaly investigation**.

Numerical anomalies are detected with deterministic Python. LangChain investigators explain, prioritize, and recommend actions — using a mock mode offline, or an OpenAI-compatible endpoint (including NVIDIA NemoClaw’s `https://inference.local/v1` route).

All data is **synthetic**. Recommendations are **advisory** and require human review. The app never changes campaigns or messages customers.

## Architecture

```text
Synthetic generators → Parquet/CSV + ground-truth labels
        ↓
Deterministic detectors (YAML thresholds)
        ↓
Candidate alerts
        ↓
LangChain / mock investigators → Pydantic reports
        ↓
Evaluation vs ground truth + Streamlit dashboard
```

| Layer | Role |
| --- | --- |
| `src/generation/` | Seeded churn + media datasets and labeled injections |
| `src/detection/` | Rolling baselines, z-scores, slopes, consistency checks |
| `src/agents/` | Mock + LangChain structured-output investigators |
| `src/models/` | Pydantic schemas + LLM client factory |
| `src/evaluation/` | Precision / recall / F1 / FPR |
| `src/privacy/` | File inventory, audit log, inference payload preview |
| `app.py` | Streamlit UI |
| `data/outputs/showcase/` | Portable JSON bundle for a future static/Vercel demo |

## Quick start

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

Dashboard: [http://localhost:8501](http://localhost:8501)

### Streamlit Community Cloud demo

Why Streamlit? The UI is a Streamlit app (`app.py`). Locally, `streamlit run app.py` opens that dashboard. For a **public demo**, deploy the same app to [Streamlit Community Cloud](https://share.streamlit.io):

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → repo `anushkamathur14-cloud/signalsentry` → branch `main` → main file `app.py`.
3. Deploy. First load generates synthetic data + mock investigations automatically (`USE_MOCK_MODEL=true`).
4. Optional secrets (usually not required for the mock demo):

```toml
USE_MOCK_MODEL = "true"
SEED = "42"
```

No live model key is needed for the demo. The cloud app does not call external providers unless you set `USE_MOCK_MODEL=false` and provide a compatible endpoint.

## Model configuration

Environment variables (see `.env.example`):

| Variable | Purpose |
| --- | --- |
| `MODEL_BASE_URL` | OpenAI-compatible base URL |
| `MODEL_API_KEY` | API key / NemoClaw placeholder |
| `MODEL_NAME` | Model id |
| `USE_MOCK_MODEL` | `true` = offline rule-based investigators |
| `SEED` | Synthetic data seed (default `42`) |

Default `.env` uses **mock mode** so the full pipeline works without network access.

### Live OpenAI-compatible endpoint

```env
USE_MOCK_MODEL=false
MODEL_BASE_URL=http://localhost:8000/v1
MODEL_API_KEY=your-local-key
MODEL_NAME=your-model-id
```

### NemoClaw + Nemotron

Inside a NemoClaw sandbox, inference should target the managed route — **not** a public provider domain:

```env
USE_MOCK_MODEL=false
MODEL_BASE_URL=https://inference.local/v1
MODEL_API_KEY=nemoclaw-local-placeholder
MODEL_NAME=nvidia/nemotron-mini
```

Notes:

- Provider credentials stay on the host / OpenShell side; the app only talks to `inference.local`.
- Use a non-sensitive placeholder key if the managed route expects an `Authorization` header.
- No hard-coded cloud provider credentials are shipped with this project.
- Complete NemoClaw onboarding on the host so `inference.local` forwards to your chosen Nemotron / compatible backend.

LangChain usage here is intentionally thin: `ChatOpenAI` + `with_structured_output` over detector context. That matches NemoClaw’s OpenAI-compatible chat-completions path better than a heavy multi-agent graph for this product.

## Commands

```bash
python -m src.generation.generate_all   # write data/generated + data/ground_truth
python -m src.run_analysis              # detect → investigate → evaluate → showcase JSON
streamlit run app.py                    # dashboard
pytest                                  # automated tests
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
