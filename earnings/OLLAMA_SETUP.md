# Ollama Setup Guide

How to set up the local AI engine for the earnings report script on a new machine.

The script itself is in git — this guide only covers Ollama installation and model setup.

## 1. Install Ollama

**macOS:**
```bash
brew install ollama
# or
curl -fsSL https://ollama.com/install.sh | sh
```

**Linux (Ubuntu/Debian/CentOS):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Manual download:**
https://ollama.com/download

**Verify:**
```bash
ollama --version
```

## 2. Choose and Pull a Model

Pick a model based on your machine's resources:

| Machine | RAM | Model | Pull Command | Size | Speed/ticker |
|---------|-----|-------|-------------|------|--------------|
| High-end (32GB+, GPU) | 32GB+ | `gpt-oss:20b` | `ollama pull gpt-oss:20b` | ~14GB | ~6s |
| Mid-range (16GB) | 16GB | `llama3.1:8b` | `ollama pull llama3.1:8b` | ~4.7GB | ~4s |
| Low-end (8GB, no GPU) | 8GB | `llama3.2:3b` | `ollama pull llama3.2:3b` | ~2GB | ~8s |
| Minimal Linux VPS | 4-8GB | `phi3:mini` | `ollama pull phi3:mini` | ~2.3GB | ~10s |

Larger models produce better financial analysis. The 3B models may miss nuances but still provide useful trend-based analysis.

```bash
# Example: pull the 8B model for a mid-range machine
ollama pull llama3.1:8b
```

## 3. Run Ollama as a Service

**macOS (brew):**
```bash
brew services start ollama
# Auto-starts on boot
```

**Linux (systemd):**
```bash
sudo systemctl enable ollama
sudo systemctl start ollama
```

**Manual (any OS):**
```bash
ollama serve &
# Or use screen/tmux for persistence
```

**Verify it's running:**
```bash
curl http://localhost:11434/api/tags
# Should return JSON with your installed models
```

## 4. Configure the Earnings Script

The script auto-detects Ollama at startup. If your model name differs from the default (`gpt-oss:20b`), add to your `.env`:

```bash
# Only needed if using a different model or non-default host
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
```

If Ollama is unreachable when the script runs, it automatically falls back to Finnhub analyst data — no manual intervention needed.

## 5. GPU Acceleration (Optional)

- **NVIDIA:** Install CUDA drivers. Ollama auto-detects the GPU.
- **Apple Silicon:** Metal acceleration is automatic.
- **CPU-only:** Works fine, just slower (~2-3x). Acceptable for background cron jobs.

## 6. Quick Test

```bash
# Test Ollama directly
ollama run llama3.1:8b "Respond with just: OK"

# Test the full pipeline (fetches data, runs AI, sends email)
python3 -m pytest test_ai_analysis.py -v -k "real" -s

# Run the earnings script
python3 earnings.py --days 0
# Check logs for: "Ollama connected: <model> ready"
```

## Performance Notes

- The script processes ~30 filtered companies on a typical day
- AI analysis runs sequentially (one ticker at a time, GPU-bound)
- Expect ~3-5 minutes total runtime depending on model size
- Market data fetching is still parallel (50 concurrent requests)
- The script runs as a daily background cron job, so latency is not a concern
