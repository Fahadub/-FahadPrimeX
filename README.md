# FahadPrimeX

A 1B-parameter model with a local web playground. I trained it myself and I'm
releasing it ready to run — clone, double-click, and it works.

It ships with a built-in web interface (English, dark theme): streaming
generation, tuning sliders, and a **live 3D preview** tab that renders the
generated Three.js scenes directly in your browser. Runs on a plain CPU —
no GPU, no cloud, no API keys.

## Quick start (Windows)

1. Install [Python 3.10+](https://www.python.org/downloads/) if you don't have it.
2. Install the two dependencies:

   ```
   pip install torch transformers
   ```

3. Double-click **`run.cmd`**.

That's it. On first run it downloads the model weights (~2.2 GB, one time
only), then starts the server and opens **http://localhost:7777**.

## Manual start (any OS)

```
python setup_models.py        # one-time weights download
python app/serve.py --port 7777
```

Then open http://localhost:7777.

## What's inside

| Path | Purpose |
|---|---|
| `run.cmd` | One-click launcher (port 7777) |
| `app/serve.py` | Dependency-free local server (Python stdlib + torch/transformers) |
| `app/ui.html` | The web playground |
| `models/FahadPrimeX/` | Model configuration, tokenizer, and weights |
| `setup_models.py` | Downloads and verifies the weights from the release |

## Notes

- Generation streams token by token, so you watch the answer build up and can
  stop it any time.
- CPU throughput is a few tokens per second; start with a small max-token
  budget and raise it as needed.
- The 3D preview tab activates automatically whenever the output is a complete
  HTML document.

## License

MIT — do whatever you want with it.
