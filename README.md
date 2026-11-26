# Venetian masks

Venice sells access to open models by name, at a price per model. This repo checks whether some of those model IDs give the model they name, using only Venice's own API, plus OpenRouter for genuine copies of the named models to compare against.

At temperature 0, a model gives the same answer to the same prompt every time, and two different models almost never give the same answers. So each Venice ID gets the same prompts as the model it is suspected to be. Every Venice response also says what Venice billed for it (`cost.usd`), so identical answers at different prices come with Venice's own receipts.

## What runs

[`probe.py`](probe.py) runs in GitHub Actions ([workflow](.github/workflows/probe.yml)). It:

1. Saves Venice's public model list (`https://api.venice.ai/api/v1/models`), which has each model's description and price.
2. For each case, sends the same requests to the ID Venice sells, to the model it is suspected to be, and to genuine copies of the sold model where one is available on OpenRouter (pinned to one host with `"allow_fallbacks": false`). Every request is temperature 0, and Venice requests set `include_venice_system_prompt: false` so that Venice adds no system prompt of its own:
   - 5 short prompts, compared character by character.
   - 3 prompts whose billed prompt tokens show the tokenizer and chat template: `Reply with OK.`, then the same with a Chinese sentence added, then with a code snippet added. These prompts come from KTibow/authenticity-autoresearch.
   - For `qwen3-235b-a22b-thinking-2507`, which Venice describes as a model that "reasons step by step before it answers": 2 arithmetic prompts with `reasoning_effort: high`, and the same 2 prompts to `deepseek-v4-flash` to show what reasoning looks like in a Venice response.
3. Writes the raw data: `results/<case>.json` (every request body, every attempt's time, status, response headers and raw response body; the API keys are not included) and `results/venice-models.json`.
4. Writes `results/summary.md`, with one section per case: the listing's description and prices, the answers side by side with identical ones marked, the billed tokens and cost, the response id format (DeepInfra's ids are `chatcmpl-R` plus 23 characters), and the token counts.

The cases:

| Venice ID | Suspected model | Compared with |
|---|---|---|
| `qwen3-235b-a22b-thinking-2507` | Qwen3-235B-A22B-Instruct-2507 | Venice's `qwen3-235b-a22b-instruct-2507` |
| `llama-3.2-3b` | Gemma 4 31B | Venice's `google-gemma-4-31b-it`; genuine Llama 3.2 3B on Parasail |
| `minimax-m27` | MiniMax-M3 | Venice's `minimax-m25`; MiniMax-M3 on DeepInfra; genuine MiniMax-M2.7 on Groq |
| `xiaomi-mimo-v2-5` | MiMo-V2.6-Flash | Venice's `xiaomi-mimo-v2-6-flash` |
| `zai-org-glm-5` | GLM-5.2 | GLM-5.2 on DeepInfra; genuine GLM-5 on Amazon Bedrock and Novita |
| `zai-org-glm-5-1` | GLM-5.3 | Venice's `z-ai-glm-5-3`; genuine GLM-5.1 on Chutes |
| `e2ee-deepseek-v4-flash` | DeepSeek-V4.1-Flash-1017 | The model name in Venice's own responses |

### Attestation

The workflow then signs a [build provenance attestation](https://docs.github.com/en/actions/concepts/security/artifact-attestations) over every file in `results/`. The attestation ties each file's SHA-256 to this repository, the commit and the workflow run. It is also recorded in Sigstore's public [Rekor](https://search.sigstore.dev) transparency log, which fixes the time it was signed. The files and the attestation (`attestation.jsonl`) are published as a release, because release files don't expire like workflow logs do.

## Verify

Download a release's files, then run:

```sh
gh attestation verify summary.md --repo KTibow/venetian-masks --bundle attestation.jsonl
```

The same command works for every other file in the release. Leave out `--bundle` to fetch the attestation from GitHub instead. The attestation names the commit it ran from, so you can read the exact code that produced the files.
