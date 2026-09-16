"""Local web interface for FahadPrimeX, with real KV-cache continuation.

Context
-------
The checkpoint declares max_position_embeddings = 128000, and the KV cache is
cheap because only 6 of the 16 layers attend (the other 10 are short
convolutions) and those use grouped-query attention with 8 KV heads:

    6 layers x 8 heads x 64 head_dim x 2 (K and V) x 4 bytes
      = 24.6 KB per token
      = 0.20 GB at 8k, 0.81 GB at 32k, 3.22 GB at 128k

So memory is not the limit here. SPEED is: this machine has no CUDA GPU and
generates roughly 2.5 tokens per second, which makes a 128k-token answer a
fourteen-hour job. The interface therefore exposes the model's true limit but
tells the user up front what it will cost.

Why sessions
------------
The naive way to continue a long answer is to resend everything produced so far
as the next prompt - which re-prefills the whole thing on every round, and on
CPU prefill is the expensive part. Instead each conversation keeps its
past_key_values in memory, so continuing costs one token of prefill, not
thousands.

Sessions are keyed by a client-supplied id, capped in number, and evicted
least-recently-used, so a browser refresh cannot leak gigabytes.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "artifacts" / "out" / "FahadPrimeX"

# Advisory API checker: flags identifiers that do not exist in three.js.
# Import is optional so the server still runs if the surface has not been built.
try:
    from validate import validate_code
except Exception:  # pragma: no cover
    def validate_code(_text):
        return {"verdict": "unavailable", "findings": 0,
                "unknown_three_members": [], "unknown_methods": [],
                "unknown_constants": []}

# The identity block exists because of how this model was trained.
#
# "FahadPrimeX" appears in all 1608 training system messages but in ZERO
# assistant messages, and not one training prompt asked about identity. The
# fine-tune therefore learned the name as something it READS, never as a fact it
# can state - asked "when were you trained?" it fell back on the base model's
# pretraining boilerplate ("a large dataset of text from 2023, books, articles,
# websites"), which is simply false for this checkpoint.
#
# Name, author and training date are not recoverable from the weights, because
# no gradient ever pushed them there. Supplying them in the system prompt is the
# honest fix at inference time: the facts are in context, so the model can
# report them instead of inventing. Making the model know them on its own would
# require identity question/answer pairs in the training corpus, which the data
# never had.
SYSTEM_PROMPT = (
    "You are FahadPrimeX, an expert Three.js engineer. You write complete, "
    "runnable, modern Three.js code using ES modules and the official addons. "
    "You are precise about the render loop, colour management, tone mapping, "
    "resource disposal and performance. When you are given existing code you "
    "read it carefully and answer about that exact code."
    "\n\n"
    "Facts about yourself. State these plainly and without hedging when asked:\n"
    "- Your name is FahadPrimeX.\n"
    "- You were created by Fahad in September 2026.\n"
    "- You have 1,170,340,608 parameters and a 128,000 token context window.\n"
    "- You were built by fine-tuning LiquidAI/LFM2.5-1.2B: two QLoRA adapters "
    "(rank 64, trained for 4 epochs on an NVIDIA A100) were folded into the "
    "Instruct and Thinking base checkpoints and then combined with SLERP.\n"
    "- Your training corpus is the official three.js repository: 608 runnable "
    "example documents, 606 deterministic code-analysis pairs, 378 addon "
    "implementations and 16 curated expert Q&A pairs - 1608 records in total.\n"
    "- Your job is Three.js engineering. You do not have internet access and "
    "you do not know events after September 2026.\n"
    "- If you are asked about yourself and the answer is not in these facts, say "
    "that you do not know rather than inventing an answer."
    "\n\n"
    "Accuracy rules. These matter more than sounding confident:\n"
    "- Only use three.js classes, methods and constants you are certain exist. "
    "Never invent a plausible-sounding API such as renderer.setRenderMode().\n"
    "- If you are unsure whether an API exists or which version introduced it, "
    "say so plainly instead of writing code that calls it.\n"
    "- Do not invent URLs, version numbers, release dates or benchmark figures. "
    "If you do not know a concrete value, say you do not know.\n"
    "- When the official documentation is the authority on a detail, say that "
    "rather than guessing at the detail.\n"
    "- Prefer fewer, correct API calls over many speculative ones."
)

STATE = {
    "model": None,
    "tokenizer": None,
    "status": "not-loaded",
    "message": "idle",
    "parameters": None,
    "context_limit": None,
    "model_dir": None,
}

# One cached conversation, keyed by session id, holding past_key_values.
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
MAX_SESSIONS = 4
LOCK = threading.Lock()


def load_model(model_dir: Path) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    STATE["status"] = "loading"
    STATE["message"] = "reading tokenizer"
    started = time.time()

    config = AutoConfig.from_pretrained(str(model_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    STATE["message"] = "reading weights (about 4.7 GB, float32 on CPU)"
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, device_map="cpu")
    model.eval()

    STATE["tokenizer"] = tokenizer
    STATE["model"] = model
    STATE["parameters"] = sum(p.numel() for p in model.parameters())
    STATE["context_limit"] = int(getattr(config, "max_position_embeddings", 4096))
    STATE["model_dir"] = str(model_dir)
    STATE["status"] = "ready"
    STATE["message"] = f"loaded in {time.time() - started:.0f}s"


def ensure_loaded(model_dir: Path) -> None:
    if STATE["status"] == "not-loaded":
        load_model(model_dir)


def get_session(session_id: str) -> dict:
    """Fetch or create a conversation, evicting the oldest when over the cap."""
    if session_id in SESSIONS:
        SESSIONS.move_to_end(session_id)
        return SESSIONS[session_id]
    SESSIONS[session_id] = {"past": None, "length": 0, "created": time.time()}
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return SESSIONS[session_id]


def reset_session(session_id: str) -> None:
    SESSIONS.pop(session_id, None)


class _StopOnFlag:
    """Stopping criteria driven by a shared flag.

    Without this, closing the browser tab leaves model.generate running to
    completion. With max_new_tokens now allowed up to the model's full 128k
    context that is not a small leak: the abandoned thread keeps the GPU-less
    CPU busy and, because generation is serialised behind a lock, blocks every
    later request for as long as fourteen hours. The handler sets the flag on
    disconnect and generate() unwinds at the next step.
    """

    def __init__(self, flag: dict) -> None:
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return bool(self.flag.get("stop"))


def stream_generation(model_dir: Path, prompt: str, max_tokens: int,
                      temperature: float, repetition_penalty: float,
                      no_repeat_ngram: int, session_id: str, reset: bool,
                      stop_flag: dict | None = None):
    """Yield SSE frames; reuse the session's KV cache when continuing."""
    import torch
    from transformers import StoppingCriteriaList, TextIteratorStreamer

    if stop_flag is None:
        stop_flag = {}

    ensure_loaded(model_dir)
    tokenizer = STATE["tokenizer"]
    model = STATE["model"]
    context_limit = STATE["context_limit"] or 4096

    if reset or session_id not in SESSIONS:
        reset_session(session_id)

    session = get_session(session_id)
    continuing = session["past"] is not None and prompt == ""

    if continuing:
        # Continuation: only the assistant header is new, so prefill is a
        # couple of tokens instead of the whole document.
        #
        # The attention mask must span the ENTIRE sequence - cached positions
        # plus the new ones - not just the new tokens. Passing a short mask
        # alongside past_key_values does not raise: generate() simply never
        # terminates, which is exactly how the first version of this hung for
        # the full client timeout while emitting no output at all.
        new_ids = tokenizer("<|im_start|>assistant\n", return_tensors="pt",
                            add_special_tokens=False)["input_ids"]
        past_len = int(session["length"])
        inputs = {
            "input_ids": new_ids,
            "attention_mask": torch.ones(
                (1, past_len + new_ids.shape[1]), dtype=torch.long),
        }
        prefix_len = 0
    else:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        prefix_len = inputs["input_ids"].shape[1]
        session["past"] = None
        session["length"] = 0

    used = session["length"] + prefix_len
    room = max(1, context_limit - used)
    budget = min(max_tokens, room)

    # timeout is essential. TextIteratorStreamer defaults it to None, so its
    # blocking queue.get() waits FOREVER when the generation thread dies without
    # emitting the stop signal - which happens whenever model.generate raises
    # inside the thread, because that exception is caught by nobody. The server
    # then burns CPU indefinitely while emitting no output, and because
    # generation is serialised behind a lock, every later request blocks too.
    # A stalled run was observed consuming 2,596 seconds of CPU this way.
    # With a timeout, next() raises queue.Empty and the loop can notice.
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15.0)

    kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=budget,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        use_cache=True,
        return_dict_in_generate=True,
        stopping_criteria=StoppingCriteriaList([_StopOnFlag(stop_flag)]),
    )
    if session["past"] is not None:
        kwargs["past_key_values"] = session["past"]
    if no_repeat_ngram and no_repeat_ngram > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram

    # Capture the resulting cache so the next call can continue from it.
    holder = {}

    def generate():
        # An exception here would otherwise die with the thread, leaving the
        # streamer waiting on a stop signal that never arrives.
        try:
            with torch.no_grad():
                out = model.generate(**kwargs)
            holder["out"] = out
        except BaseException as exc:  # noqa: BLE001 - must reach the caller
            holder["error"] = repr(exc)
            streamer.end()

    started = time.time()
    thread = threading.Thread(target=generate, daemon=True)
    thread.start()

    produced = 0
    last_progress = time.time()
    collected: list = []
    stalled = False

    while True:
        try:
            chunk = next(streamer)
        except StopIteration:
            break
        except queue.Empty:
            # No token for 15s. Either the model is genuinely slow, or the
            # generation thread has died. Distinguish the two.
            if not thread.is_alive():
                stalled = True
                break
            if time.time() - last_progress > 180:
                stop_flag["stop"] = True
                stalled = True
                thread.join(timeout=30)
                break
            continue

        produced += len(chunk)
        collected.append(chunk)
        last_progress = time.time()
        elapsed = time.time() - started
        yield "data: " + json.dumps({
            "type": "token", "text": chunk,
            "chars": produced,
            "seconds": round(elapsed, 1),
            "chars_per_second": round(produced / elapsed, 1) if elapsed else 0,
        }) + "\n\n"

    thread.join(timeout=10)
    generation_error = holder.get("error")

    if "out" in holder:
        session["past"] = holder["out"].past_key_values
        generated_ids = holder["out"].sequences
        # With a cache supplied, .sequences holds ONLY the newly generated
        # tokens - not the whole conversation. Overwriting the running length
        # with that delta makes the next attention mask far too short, and a
        # short mask alongside past_key_values does not raise: generate() simply
        # never returns. The length has to accumulate.
        new_tokens = int(generated_ids.shape[1])
        session["length"] = (past_len + new_tokens) if continuing else new_tokens
        if stop_flag.get("stop"):
            # The cache is only valid up to whatever actually got generated, so
            # a stopped run must not be continued from.
            session["past"] = None

    # Check the finished text against the real three.js API surface. Findings
    # are advisory: a user-defined helper is not a hallucination, but a method
    # that exists on no class in the library almost certainly is.
    full_text = "".join(collected)
    try:
        validation = validate_code(full_text)
    except Exception as exc:
        validation = {"verdict": "error", "error": repr(exc), "findings": 0}

    yield "data: " + json.dumps({
        "type": "done",
        "chars": produced,
        "seconds": round(time.time() - started, 1),
        "context_used": session["length"],
        "context_limit": context_limit,
        "truncated": budget < max_tokens,
        "stalled": stalled,
        "generation_error": generation_error,
        "validation": validation,
    }) + "\n\n"


class Handler(BaseHTTPRequestHandler):
    model_dir = DEFAULT_MODEL

    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/", "/index.html"):
            ui = Path(__file__).resolve().parent / "ui.html"
            if not ui.exists():
                self._send(500, b"ui.html is missing", "text/plain; charset=utf-8")
                return
            self._send(200, ui.read_bytes(), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/status":
            self._send(200, json.dumps({
                "status": STATE["status"],
                "message": STATE["message"],
                "parameters": STATE["parameters"],
                "context_limit": STATE["context_limit"],
                "sessions": len(SESSIONS),
                "model": str(self.model_dir),
            }).encode(), "application/json; charset=utf-8")
            return

        if parsed.path == "/api/reset":
            session_id = (parse_qs(parsed.query).get("session") or ["default"])[0]
            reset_session(session_id)
            self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            return

        if parsed.path == "/api/generate":
            params = parse_qs(parsed.query)
            prompt = (params.get("prompt") or [""])[0].strip()
            continue_run = (params.get("continue") or ["0"])[0] == "1"
            session_id = (params.get("session") or ["default"])[0]
            if not prompt and not continue_run:
                self._send(400, b'{"error":"prompt is required"}',
                           "application/json; charset=utf-8")
                return

            max_tokens = int((params.get("max_tokens") or ["400"])[0])
            temperature = float((params.get("temperature") or ["0.7"])[0])
            repetition = float((params.get("repetition_penalty") or ["1.15"])[0])
            ngram = int((params.get("no_repeat_ngram") or ["6"])[0])

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()

            stop_flag = {"stop": False}
            with LOCK:
                try:
                    for frame in stream_generation(
                            self.model_dir, "" if continue_run else prompt,
                            max_tokens, temperature, repetition, ngram,
                            session_id, reset=not continue_run,
                            stop_flag=stop_flag):
                        self.wfile.write(frame.encode("utf-8"))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    # Tell the generation thread to unwind instead of running to
                    # the end of a budget that may be tens of thousands of
                    # tokens long.
                    stop_flag["stop"] = True
                except Exception as exc:
                    stop_flag["stop"] = True
                    try:
                        self.wfile.write(("data: " + json.dumps(
                            {"type": "error", "message": repr(exc)}) + "\n\n")
                            .encode("utf-8"))
                        self.wfile.flush()
                    except Exception:
                        pass
                finally:
                    pass
            return

        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="FahadPrimeX web interface")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"model not found: {model_dir}")
        return 1
    Handler.model_dir = model_dir

    if args.preload:
        print("[load] preloading...", flush=True)
        ensure_loaded(model_dir)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("")
    print("=" * 70)
    print(f"  FahadPrimeX serving at http://{args.host}:{args.port}")
    print(f"  model: {model_dir}")
    if STATE["context_limit"]:
        print(f"  context limit: {STATE['context_limit']:,} tokens")
    print("=" * 70)
    print("")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    _exit_code = main()
    if _exit_code:
        raise SystemExit(_exit_code)
