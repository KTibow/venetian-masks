#!/usr/bin/env python3
"""Ask Venice's API whether a model ID gives the model it names.

For each case, the same prompts at temperature 0 go to the ID Venice sells (`sold`) and to the model it is suspected
to be (`twin`), plus genuine references of the sold model where one exists (through OpenRouter, pinned to one host).
Everything is written to results/: every request body (no keys), every raw response, and summary.md.

Env: VENICE_KEY (required), OPENROUTER_KEY (optional, for references)."""
import concurrent.futures as cf, json, os, re, time, urllib.error, urllib.request

VENICE = "https://api.venice.ai/api/v1"
OPENROUTER = "https://openrouter.ai/api/v1"
VK = os.environ["VENICE_KEY"].strip()
OK = os.environ.get("OPENROUTER_KEY", "").strip()

# Greedy prompts: identical models give identical text at temperature 0.
PROMPTS = [
    "Which model are you exactly, and who trained you? One sentence.",
    "Name a random animal. One word.",
    "In exactly two sentences, explain why the sky is blue.",
    "Give one surprising fact about octopuses in one sentence.",
    "Pick a random city. One word.",
]
# Tokenizer prompts (from KTibow/authenticity-autoresearch): billed prompt tokens for BASE, and how much Chinese and
# code text add. Different tokenizers or chat templates give different numbers.
BASE = "Reply with OK."
ZH = "人工智能的发展正在改变我们的生活方式。从智能手机到自动驾驶汽车，机器学习算法无处不在，并且持续影响着教育、医疗和交通等各个领域。"
CODE = "def merge(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i]); i += 1\n        else:\n            out.append(b[j]); j += 1\n    return out + a[i:] + b[j:]\n"
TOKENS = {"BASE": BASE, "+ZH": BASE + "\n" + ZH, "+CODE": BASE + "\n" + CODE}
REASON = "Compute 4837*2961. Work it out carefully, then end with the final number alone on the last line."


def venice(model, think_off=True):
    vp = {"include_venice_system_prompt": False}
    if think_off:
        vp["disable_thinking"] = True
    return {"channel": "venice", "label": f"Venice {model}", "body": {"model": model, "venice_parameters": vp}}


def openrouter(model, host, extra=None):
    return {"channel": "openrouter", "label": f"OpenRouter {model} on {host}",
            "body": {"model": model, "provider": {"only": [host], "allow_fallbacks": False}, **(extra or {})}}


NO_REASONING = {"reasoning": {"enabled": False}}
CASES = [
    {"key": "qwen3-235b-thinking", "title": "`qwen3-235b-a22b-thinking-2507` is Qwen3-235B-A22B-Instruct-2507",
     "sold": venice("qwen3-235b-a22b-thinking-2507", False), "twins": [venice("qwen3-235b-a22b-instruct-2507", False)],
     "refs": [], "reasoning": True},
    {"key": "llama-3.2-3b", "title": "`llama-3.2-3b` is Gemma 4 31B",
     "sold": venice("llama-3.2-3b"), "twins": [venice("google-gemma-4-31b-it")],
     "refs": [openrouter("meta-llama/llama-3.2-3b-instruct", "parasail/bf16")]},
    {"key": "minimax-m27", "title": "`minimax-m27` is MiniMax-M3",
     "sold": venice("minimax-m27"), "twins": [venice("minimax-m25"), openrouter("minimax/minimax-m3", "deepinfra/fp8", NO_REASONING)],
     "refs": [openrouter("minimax/minimax-m2.7", "groq")]},
    {"key": "mimo-v2.5", "title": "`xiaomi-mimo-v2-5` is Venice's MiMo-V2.6-Flash",
     "sold": venice("xiaomi-mimo-v2-5"), "twins": [venice("xiaomi-mimo-v2-6-flash")], "refs": []},
    {"key": "glm-5", "title": "`zai-org-glm-5` is GLM-5.2",
     "sold": venice("zai-org-glm-5"), "twins": [openrouter("z-ai/glm-5.2", "deepinfra/fp4", NO_REASONING)],
     "refs": [openrouter("z-ai/glm-5", "amazon-bedrock", NO_REASONING), openrouter("z-ai/glm-5", "novita/fp8", NO_REASONING)]},
    {"key": "glm-5.1", "title": "`zai-org-glm-5-1` is GLM-5.3",
     "sold": venice("zai-org-glm-5-1"), "twins": [venice("z-ai-glm-5-3", False)],
     "refs": [openrouter("z-ai/glm-5.1", "chutes/fp8", NO_REASONING)]},
    {"key": "e2ee-v4-flash", "title": "`e2ee-deepseek-v4-flash` is DeepSeek-V4.1-Flash-1017",
     "sold": venice("e2ee-deepseek-v4-flash", False), "twins": [], "refs": []},
]
# Shows what a reasoning model's response looks like on Venice with the same request.
REASONING_CONTROL = venice("deepseek-v4-flash", False)


def post(route, messages, max_tokens, extra=None):
    if route["channel"] == "openrouter" and not OK:
        return {"skipped": "no OPENROUTER_KEY"}
    url, key = (VENICE, VK) if route["channel"] == "venice" else (OPENROUTER, OK)
    body = {**route["body"], **(extra or {}), "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    rec = {"route": route["label"], "channel": route["channel"], "request": body, "attempts": []}
    for attempt in range(4):
        t = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            resp = urllib.request.urlopen(urllib.request.Request(
                url + "/chat/completions", json.dumps(body).encode(),
                {"Authorization": "Bearer " + key, "Content-Type": "application/json"}), timeout=300)
            raw = resp.read().decode()
            rec["attempts"].append({"time": t, "status": resp.status, "headers": dict(resp.headers), "body": raw})
            rec["response"] = json.loads(raw)
            return rec
        except urllib.error.HTTPError as e:
            raw = e.read().decode(errors="replace")
            rec["attempts"].append({"time": t, "status": e.code, "headers": dict(e.headers), "body": raw})
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(15 * (attempt + 1))
                continue
            return rec
        except Exception as e:
            rec["attempts"].append({"time": t, "status": None, "error": repr(e)})
            if attempt < 3:
                time.sleep(10)
                continue
            return rec


def jobs_for(case):
    routes = [case["sold"], *case["twins"], *case["refs"]]
    jobs = []
    for r in routes:
        jobs += [(case["key"], r, f"prompt {i + 1}", [{"role": "user", "content": p}], 300, None) for i, p in enumerate(PROMPTS)]
        jobs += [(case["key"], r, f"tokens {k}", [{"role": "user", "content": v}], 16, None) for k, v in TOKENS.items()]
    if case.get("reasoning"):
        for r in [case["sold"], REASONING_CONTROL]:
            jobs += [(case["key"], r, f"reasoning {n + 1}", [{"role": "user", "content": REASON}], 3000,
                      {"reasoning_effort": "high"}) for n in range(2)]
    return jobs


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "venetian-masks"}), timeout=60) as r:
        return json.load(r)


def text(rec):
    try:
        return rec["response"]["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return None


def reasoning_len(rec):
    try:
        m = rec["response"]["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None
    return len(m.get("reasoning_content") or m.get("reasoning") or "")


def usage(rec):
    return (rec or {}).get("response", {}).get("usage") or {}


def id_shape(rec):
    i = (rec or {}).get("response", {}).get("id", "")
    if re.fullmatch(r"chatcmpl-R[A-Za-z0-9]{23}", i):
        return "`chatcmpl-R`+23 (DeepInfra's format)"
    if re.fullmatch(r"chatcmpl-[0-9a-f]{32}(-r\d+)?", i):
        return "`chatcmpl-`+32 hex"
    if i.startswith("gen-"):
        return "`gen-` (OpenRouter)"
    return f"`{i[:24]}…`" if i else "–"


def cost(rec):
    r = (rec or {}).get("response", {})
    if "cost" in r:
        return r["cost"].get("usd")
    return usage(rec).get("cost")


def short(s, n=90):
    if s is None:
        return "*(error)*"
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "…"


def main():
    os.makedirs("results", exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    listing = get(VENICE + "/models")
    json.dump(listing, open("results/venice-models.json", "w"), indent=1, ensure_ascii=False)
    specs = {m["id"]: m.get("model_spec", {}) for m in listing["data"]}

    jobs = [j for c in CASES for j in jobs_for(c)]
    with cf.ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(lambda j: {"case": j[0], "test": j[2], **post(j[1], j[3], j[4], j[5])}, jobs))
    by = {}
    for r in recs:
        by.setdefault(r["case"], []).append(r)
    for k, rs in by.items():
        json.dump(rs, open(f"results/{k}.json", "w"), indent=1, ensure_ascii=False)

    out = [f"# Venice model IDs vs the models they answer as\n",
           f"Run started {started}. Every request is temperature 0, with `include_venice_system_prompt: false` on Venice. "
           "Raw requests and responses are in `results/<case>.json`; Venice's model list (descriptions and prices) is in "
           "`results/venice-models.json`. Costs are what Venice itself billed for each request (`cost.usd` in its response), "
           "or OpenRouter's `usage.cost`.\n"]
    for c in CASES:
        rs = by[c["key"]]
        pick = lambda label, test: next((r for r in rs if r.get("route") == label and r["test"] == test), None)
        routes = [c["sold"], *c["twins"], *c["refs"]]
        sold = c["sold"]["label"]
        out.append(f"\n## {c['title']}\n")
        mid = c["sold"]["body"]["model"]
        s = specs.get(mid)
        if s:
            p = s.get("pricing", {})
            out.append(f"Venice lists `{mid}` as \"{short(s.get('description'), 200)}\" at "
                       f"${p.get('input', {}).get('usd')}/${p.get('output', {}).get('usd')} per million input/output tokens.\n")
        for t in c["twins"]:
            tm = t["body"]["model"]
            if t["channel"] == "venice" and tm in specs:
                p = specs[tm].get("pricing", {})
                out.append(f"Venice lists `{tm}` at ${p.get('input', {}).get('usd')}/${p.get('output', {}).get('usd')}.\n")
        out.append("\n**Same prompts, temperature 0** (✓ = text identical to the sold ID's):\n")
        out.append("| Prompt | " + " | ".join(r["label"] for r in routes) + " |")
        out.append("|---" * (len(routes) + 1) + "|")
        same = {r["label"]: 0 for r in routes}
        for i, p in enumerate(PROMPTS):
            base = text(pick(sold, f"prompt {i + 1}"))
            cells = []
            for r in routes:
                rec = pick(r["label"], f"prompt {i + 1}")
                if rec and "skipped" in rec:
                    cells.append("*(skipped)*")
                    continue
                tx = text(rec)
                mark = ""
                if r["label"] != sold and tx is not None and tx == base and base:
                    mark = "✓ "
                    same[r["label"]] += 1
                cells.append(mark + short(tx))
            out.append(f"| {short(p, 40)} | " + " | ".join(cells) + " |")
        out.append("| **Identical to sold ID** | – | " + " | ".join(f"**{same[r['label']]}/{len(PROMPTS)}**" for r in routes[1:]) + " |")
        out.append("\n**Billing and host** (prompt 1):\n")
        out.append("| Route | Response id | Prompt / completion tokens | Billed USD |")
        out.append("|---|---|---|---|")
        for r in routes:
            rec = pick(r["label"], "prompt 1")
            u = usage(rec)
            out.append(f"| {r['label']} | {id_shape(rec)} | {u.get('prompt_tokens', '–')} / {u.get('completion_tokens', '–')} | {cost(rec) if cost(rec) is not None else '–'} |")
        out.append("\n**Billed prompt tokens** (same tokenizer and chat template give the same numbers):\n")
        out.append("| Route | BASE | +ZH − BASE | +CODE − BASE |")
        out.append("|---|---|---|---|")
        for r in routes:
            n = {k: usage(pick(r["label"], f"tokens {k}")).get("prompt_tokens") for k in TOKENS}
            d = lambda k: n[k] - n["BASE"] if n[k] is not None and n["BASE"] is not None else "–"
            out.append(f"| {r['label']} | {n['BASE'] if n['BASE'] is not None else '–'} | {d('+ZH')} | {d('+CODE')} |")
        if c.get("reasoning"):
            out.append("\n**Reasoning, `reasoning_effort: high`** (a reasoning model returns `reasoning_content`):\n")
            out.append("| Route | Run | Reasoning characters | Reasoning tokens | Completion tokens | Last line | Billed USD |")
            out.append("|---|---|---|---|---|---|---|")
            for r in [c["sold"], REASONING_CONTROL]:
                for n in (1, 2):
                    rec = pick(r["label"], f"reasoning {n}")
                    u = usage(rec)
                    rt = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
                    last = (text(rec) or "").strip().splitlines()[-1:] or ["–"]
                    out.append(f"| {r['label']} | {n} | {reasoning_len(rec)} | {rt if rt is not None else '–'} | {u.get('completion_tokens', '–')} | {short(last[0], 30)} | {cost(rec)} |")
            out.append(f"\n(4837 × 2961 = {4837 * 2961}.)")
        if c["key"] == "e2ee-v4-flash":
            found = {}
            for rec in rs:
                bodies = " ".join(a.get("body", "") for a in rec.get("attempts", []))
                m = re.search(r"The requested model [^\"]*is an alias of[^\"]*", bodies)
                if m:
                    found.setdefault(m.group(0), []).append(rec["test"])
            out.append(f"\n**What the responses themselves say about the model** ({sum(map(len, found.values()))} of {len(rs)} responses):\n")
            for msg, tests in found.items():
                out.append(f"> {msg}\n\n({len(tests)} responses: {', '.join(tests)})")
    out.append(f"\n\nRun finished {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}.")
    open("results/summary.md", "w").write("\n".join(out) + "\n")
    errs = [r for r in recs if "response" not in r and "skipped" not in r]
    print(f"{len(recs)} requests, {len(errs)} without a response")


if __name__ == "__main__":
    main()
