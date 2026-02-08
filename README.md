# Stock Market Pattern Mining

End-to-end stock analysis pipeline that combines ML-based pattern recognition on historical technical indicators with real-time news sentiment analysis to produce short-term trading signals.

Built for **CSCI 5502 - Data Mining** at CU Boulder.

## Features

- **Historical data ingestion** via yfinance for multiple tickers
- **Technical indicator engineering**: daily returns, SMA ratios (5/10/20), RSI (14-period), rolling volatility
- **Binary classification** of next-day price movement (UP/DOWN) using Random Forest or Logistic Regression
- **Model evaluation** with accuracy, precision, recall, F1, and confusion matrix
- **EDA visualizations**: price trends, return distributions, feature correlation heatmaps
- **Live inference** combining model predictions (60%) with news sentiment (40%) into BUY/HOLD/SELL signals
- **News sentiment analysis** via OpenAI LLM summarization with VADER fallback
- **Real-time market data** from Finnhub API (quotes and company news)

## Project Structure

```
├── config.yaml                  # Pipeline configuration (tickers, model params, feature windows)
├── requirements.txt             # Python dependencies
├── run.sh                       # Shell script to run full pipeline
├── src/dmp/
│   ├── cli.py                   # CLI entry point (train, eval, eda, live, all)
│   ├── config.py                # Pydantic configuration models
│   ├── user_config.py           # Environment variable loader for API keys
│   ├── apis/
│   │   ├── finnhub_client.py    # Finnhub API client (quotes, company news)
│   │   └── llm.py               # OpenAI sentiment summarization + VADER fallback
│   ├── data/
│   │   ├── yfinance_loader.py   # Historical OHLCV data downloader
│   │   └── kaggle_loader.py     # Optional Kaggle dataset loader
│   ├── features/
│   │   └── engineering.py       # Technical indicator computation
│   ├── models/
│   │   ├── train.py             # Model training (RandomForest / LogisticRegression)
│   │   └── evaluate.py          # Model evaluation and metrics export
│   ├── pipeline/
│   │   └── inference.py         # Live prediction on latest market data
│   └── visualization/
│       └── eda.py               # EDA plots (price, returns, correlations)
├── tests/
│   ├── test_features.py         # Feature engineering tests
│   └── test_inference.py        # Inference pipeline tests
├── reports/figures/              # Generated EDA plots
├── artifacts/                    # Trained model and metrics (gitignored)
└── contents/
    ├── DataMiningReport.pdf     # Final project report
    └── CSCI 5502 - Project Proposal.pptx
```

## Quickstart

### 1. Set up the environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. (Optional) Configure API keys

Create a `user_config.env` file in the project root:

```
FINNHUB_API_KEY=your-finnhub-key
OPENAI_API_KEY=your-openai-key
```

If these keys are not set, the pipeline still works: it uses placeholder headlines and VADER for sentiment instead of Finnhub + OpenAI.

### 3. Run the pipeline

```bash
# Run everything (train → eval → eda → live inference)
python -m dmp.cli all

# Or use the shell script
bash run.sh

# Or run individual steps
python -m dmp.cli train
python -m dmp.cli eval
python -m dmp.cli eda
python -m dmp.cli live --ticker AAPL
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `train` | Train classifier on historical data, save model to `artifacts/model.joblib` |
| `eval`  | Evaluate saved model on the full dataset, save metrics to `artifacts/eval_metrics.json` |
| `eda`   | Generate EDA plots to `reports/figures/` |
| `live`  | Run one live prediction with news sentiment aggregation |
| `all`   | Run train → eval → eda → live sequentially |

### Live inference options

```bash
# Single prediction
python -m dmp.cli live --ticker MSFT

# Continuous polling loop (Ctrl+C to stop)
python -m dmp.cli live --ticker AAPL --loop --interval 60
```

## How It Works

### Technical Indicators (Features)

| Feature | Description |
|---------|-------------|
| `return_1d` | Daily percentage return |
| `sma_ratio_5/10/20` | Close price divided by 5/10/20-day simple moving average |
| `rsi` | 14-period Relative Strength Index |
| `volatility` | 10-day rolling standard deviation of returns |
| `target_up` | Binary label: 1 if next day's close > today's close |

### Decision Aggregation

Live inference combines the ML model prediction with news sentiment:

```
final_score = 0.6 * model_probability_up + 0.4 * news_sentiment_score

BUY  if final_score >= 0.55
HOLD if 0.45 <= final_score < 0.55
SELL if final_score < 0.45
```

## Configuration

All pipeline parameters are in `config.yaml`:

- **data**: tickers (`AAPL`, `MSFT`, `GOOGL`), date range (2016--2024), data source
- **features**: RSI window, SMA windows, volatility window
- **model**: type (`random_forest` or `logistic_regression`), test split, hyperparameters
- **live**: polling interval, default ticker
- **llm**: OpenAI model, max headlines, VADER fallback toggle

## Output Artifacts

| Path | Contents |
|------|----------|
| `artifacts/model.joblib` | Trained sklearn pipeline + feature list |
| `artifacts/metrics.json` | Training metrics (accuracy, precision, recall, F1) |
| `artifacts/eval_metrics.json` | Full-dataset evaluation + confusion matrix |
| `reports/figures/*_close.png` | Per-ticker closing price plots |
| `reports/figures/return_distribution.png` | Daily return histogram |
| `reports/figures/feature_correlation.png` | Feature correlation heatmap |

## Disclaimer

This project is for educational purposes only and does not constitute financial advice.