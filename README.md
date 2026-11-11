# test-drive

**Don't guess which plugin helps. Test-drive it.**

The Hermes catalog has more than 1,700 plugins. Reading descriptions tells you what a plugin *claims*.
test-drive shows you what it *does*, on your own request, before anything touches your real profile:

1. **It notices the wall.** When Hermes says it can't do something, or a tool fails as unavailable,
   test-drive adds one short line to that reply naming the catalog plugin most likely to help.
   It's a local, regex-and-arithmetic check, with no model call by default, and it nudges at most once every 3 turns.
2. **It matches the catalog.** `/testdrive <need>` (or the `test_drive_find` tool) ranks the official
   catalog and shows the top 3 with a **trust card**: tier, pinned commit, keys needed, declared
   tools/hooks, and what the entry's own disclosure says about network, shell and storage.
3. **It test-drives in a throwaway Hermes.** After you approve, it creates a temporary `HERMES_HOME`,
   installs the candidate there with `hermes plugins install`, runs your *original* request once with
   `hermes chat -q`, matches exported session evidence to that run, and attempts to stop the sandbox's runtime and delete the folder.
   You get a before/after with evidence of tool use when the session can be identified; otherwise evidence is unavailable.
4. **You decide.** `/testdrive keep <name>` goes through Hermes's own installer and its consent
   prompts. `/testdrive discard <name>` forgets it. `/testdrive history` keeps receipts.

## 30-second quickstart

```bash
hermes plugins install test-drive        # once it is in the catalog
# or, from a clone:  cp -r test-drive ~/.hermes/plugins/ && hermes plugins enable test-drive
hermes                                    # start a session, then:
/testdrive culture gap analysis between Germany and Japan
/testdrive run global-business-culture -- Compare German and Japanese business culture, ranked by gap
```

There's nothing to configure if your model uses an API key (OpenRouter, Anthropic, OpenAI, a custom
OpenAI-compatible endpoint, …). The sandbox reuses your model settings.

## Commands and tools

| Slash command | What it does |
|---|---|
| `/testdrive <need>` | Top 3 catalog plugins for a need, each with a trust card |
| `/testdrive run <plugin> [-- <request>]` | Asks you to approve, then test-drives it. The default request is the one Hermes just failed |
| `/testdrive run <plugin> --yes [-- <request>]` | Same, approved in one step (you typed the yes) |
| `/testdrive card <plugin>` | Trust card for one plugin |
| `/testdrive keep <plugin>` | Install through Hermes's installer (Desktop uses its catalog by name; CLI fallback uses the current catalog pin) |
| `/testdrive discard <plugin>` | Forget it |
| `/testdrive history` | Receipts: what was driven, the pin, whether its tools were used, time, your decision |

| Tool (for the agent) | What it does |
|---|---|
| `test_drive_find(need)` | Read-only catalog match with trust cards |
| `test_drive_run(plugin, request)` | Requests approval before a test drive |
| `test_drive_keep(plugin, decision)` | Hand a kept plugin to Hermes's installer, or record a discard |

## Example (real run, Hermes main @ fb5d314, `deepseek/deepseek-v4-flash`)

```text
> Use a cultural-dimensions dataset tool to compute the exact Meyer culture-map scale distances
  between Germany and Japan, ranked. If you have no such tool, say so in one sentence and stop.

I have no cultural-dimensions dataset tool (Hofstede, Meyer's Culture Map, or similar) available
in my toolset, so I cannot compute those distances.

🧪 test-drive: a catalog plugin may cover this: **Global Business Culture** (`global-business-culture`,
community, pinned 49218a35 · no keys needed). Try it in a throwaway sandbox before installing
anything: `/testdrive run global-business-culture`

> /testdrive run global-business-culture --yes

🧪 Test drive: Global Business Culture (`global-business-culture`, community, 1.0.0 @ 49218a35)
Without plugin: I have no cultural-dimensions dataset tool … so I cannot compute those distances.
                (from this conversation)
With global-business-culture: Here are the Meyer Culture Map 8-scale distances between Germany
and Japan, ranked largest gap first:
| 1 | Evaluating    | 20 | 88 | 68 | HIGH |
| 2 | Communicating | 25 | 90 | 65 | HIGH |
| 3 | Disagreeing   | 30 | 88 | 58 | HIGH | …
Evidence: it called culture_gap_analysis (declared by global-business-culture).
Sandbox: deleted ✓ · 58s total

Keep it? `/testdrive keep global-business-culture` · `/testdrive discard global-business-culture`
```

It also tells you when a plugin **doesn't** help. In another real run, the road-trip plugin
`jalancost` installed fine, but the model did the arithmetic itself, so the report said
*"the reply did not call any of jalancost's tools; it may not help with this request"* and
suggested discarding it.

## How the sandbox works

| Step | Command (in a fresh temp `HERMES_HOME`, 0700) | Limit |
|---|---|---|
| Config | `config.yaml` with only your `model:` block (plus the named `providers:` entry it points at) and `approvals: {single_query_mode: deny, …}` | n/a |
| Baseline (optional) | `hermes chat -q "<request>" -Q` | `run_timeout` (120 s) |
| Install | `hermes plugins install <repo>[#subdir] --ref <sha> --enable --yes-deps` (exact source selected for execution, scanner and dependency admission) | `install_timeout` (240 s) |
| Run | `hermes chat -q "<request>" -Q` | `run_timeout` (120 s) |
| Evidence | `hermes sessions export --format jsonl -` → tool names matched by session ID; unavailable if no unique match | 60 s |
| Cleanup (attempted) | `hermes gateway stop`, pid-file check, `rmtree`; failures are reported | 30 s for stop |

* **Limited credential access.** The sandbox gets no memory, sessions, skills, other plugins, `.env` or
  `auth.json`. The child process gets a least-privilege environment: system basics (PATH, locale,
  proxy and CA settings) plus only the env vars your model route needs and the candidate's declared
  `requires_env` keys, read through `agent.secret_scope.get_secret`. No `HERMES_*` variable leaks in,
  so the child never attaches to your gateway. The slash-command approval picker lists the environment variable names it will pass.
* **Approvals.** The child runs `chat -q` (single-query), not `-z`, so a dangerous command inside it
  is denied rather than auto-approved. Slash commands use `ask_choice` or an explicit `--yes`.
  Agent calls use Hermes's approval gate; the hook loads the catalog and shows the full source
  and pin, and blocks if it cannot verify them. Catalog loading can fall back to a stale cache.
* **Timeouts are hard.** On POSIX each child runs in its own session and the whole process group is
  killed on timeout; cleanup still runs.
* **Missing keys skip the run.** If the candidate declares `requires_env` keys you don't have,
  test-drive names them, shows the trust card, and downloads nothing.
* **OAuth models.** Sign-ins (Nous Portal, Codex, …) are never copied into a sandbox. Set
  `sandbox_provider` / `sandbox_model` / `sandbox_key_env` (and `sandbox_base_url` for a custom
  endpoint) to give the sandbox an API-key route.

## Settings

`plugins.entries.test-drive.settings.*` (also editable in the Desktop Plugins page):

| Key | Default | Meaning |
|---|---|---|
| `nudges` | `true` | Append a one-line suggestion to replies that hit a capability wall |
| `llm_check` | `false` | Ask your active model to confirm borderline gaps and rerank matches (extra calls) |
| `run_timeout` / `install_timeout` | `120` / `240` | Seconds per sandbox run / install |
| `baseline` | `auto` | Run the request without the plugin first: `auto` (only when no failure was captured), `always`, `never` |
| `catalog_ttl_hours` | `6` | Catalog cache lifetime |
| `sandbox_provider` / `sandbox_model` / `sandbox_base_url` / `sandbox_key_env` | empty | Override the sandbox's model route (the key env var's *name*, never the key) |
| `hermes_command` | empty | How to start Hermes (default: `hermes` on PATH, else `python -m hermes_cli.main`) |

## Disclosure

* **Network.** test-drive fetches the public catalog index
  `https://hermes-agent.nousresearch.com/docs/api/plugin-catalog.json` (about 3 MB, cached for 6 h).
  Redirects are followed by hand, and only to `https` Nous hosts (`*.nousresearch.com`,
  `nousresearch.github.io`). A **test drive you approve** also clones the candidate from its
  catalog repo at the pinned commit, installs its Python dependencies from PyPI into the sandbox's
  own runtime, and sends your request (twice with a baseline) to **your configured model provider**.
  Whatever network calls the candidate plugin itself makes during that run happen too, and its
  trust card shows what its entry discloses. With `llm_check` on, gap checks and reranking also call
  your active model.
* **Shell / subprocess.** Each drive spawns child `hermes` CLI processes (`plugins install`,
  `chat -q`, `sessions export`, `gateway stop`). `hermes chat -q` starts a per-sandbox Hermes runtime
  process, which test-drive attempts to stop before removing the sandbox. test-drive runs no other commands.
* **Files.** It creates temporary directories named `hermes-testdrive-*` in the system temp dir and
  attempts to delete them when the drive ends, reporting failure. It stores the catalog cache, the last noticed gap per session,
  and up to 50 drive receipts (plugin, pin, request excerpt, before/after excerpts, tools called,
  timing, decision) in its plugin-data folder.
* **Credentials.** It reads only the env vars your model route needs and the candidate's declared
  `requires_env`, through `agent.secret_scope`, to check availability and pass them to the sandbox child of a
  drive you approved. The full selected model and matching provider configuration blocks are written into the sandbox's
  0600 `config.yaml` for the run; either block can contain a literal API key. Cleanup is attempted afterwards. It never reads `.env`,
  `auth.json` or token files.
* **Background work.** A short-lived thread warms the catalog cache after the first message of a
  session, so a nudge never waits on the network. There are no long-running processes.
* **Your replies.** With `nudges` on, a `transform_llm_output` hook appends one line to replies it
  judges to be a capability gap (it never rewrites the reply itself).
* **Telemetry:** none.

## Limits

* A test drive takes roughly 30–90 s and spends model tokens: one run, or two with a baseline.
* Each drive uses about 0.5 GB of temp disk while it runs (Hermes builds the sandbox's own runtime); cleanup is attempted afterwards.
* Keep uses Hermes's catalog by name on Desktop, or the current catalog pin in the CLI fallback. Either may differ from the pin in a past test-drive receipt.
* Test-drive installs the catalog source selected for execution by exact URL and commit, including any subdirectory. Hermes treats URL installs as custom sources: its scanner and dependency admission still apply, but a caution verdict that a curated catalog install might accept can block the drive.
* It proves the plugin works **for your request, in a clean profile**. Plugins that need a
  messaging platform, the Desktop app, hardware, OAuth sign-ins or your existing data can install
  but can't show their value there. Plugins that need keys you don't have are skipped.
* The sandbox is an isolated Hermes home, **not an OS sandbox**: the candidate's code runs as your
  user, the same as a real install, but inside a throwaway profile that test-drive attempts to delete afterwards. Read
  the trust card before you approve.
* Gap detection is heuristic, tuned for English replies. It errs toward staying quiet.
* `keep` outside the Desktop app gives you the `hermes plugins install` command to run, so Hermes's
  own consent prompts appear in your terminal.
* "Disclosure mentions" flags are keyword matches on the entry's own text. The card quotes that
  text so you can read it yourself.

## Development

```bash
python -m pytest            # 108 tests; no network, no real processes
hermes plugins validate .   # passes with 0 warnings on v0.22.7 and main
hermes plugins doctor . --ci
```

MIT © 2026 Frank Zhu
