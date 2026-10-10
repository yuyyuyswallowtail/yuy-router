# yuy

A semantic command router for the terminal. You type a plain sentence, `yuy` matches it against the commands in `commands.json` using embeddings (EmbeddingGemma 2 through llama.cpp), and runs the best match. There is no chat LLM and no `eval`: only commands you registered yourself can run.

    yuy "calculate (2 + 3) * 4 ^ 2"
    yuy "increase volume 10"
    yuy "set brightness to 60"
    yuy "gpu info"
    yuy "show the folder tree depth 2"

## How it works

1. The embedding server (`llama-server --embeddings`) is started automatically in the background. Double starts are prevented with a health check, a process scan, and a file lock.
2. The sentence and each command description are turned into vectors, and the score is the cosine similarity. A command gets a +0.10 bonus when one of its `keywords` appears in the sentence, or when the sentence has a math expression (for `expression` commands).
3. The command runs only when the score is at least 0.75 (`THRESHOLD`) and the gap to the runner-up is at least 0.03 (`MIN_MARGIN`). Otherwise nothing runs.

Sentences can be English or Indonesian: descriptions are English, and `keywords` and operator words (`plus`, `times`, `divided by`, `tambah`, `kali`, `dibagi`, ...) cover both.

## Requirements

- Linux (system info reads `/proc` and `/sys`) and Python 3.9+
- llama.cpp (`llama-server`) and an EmbeddingGemma 2 model in GGUF format
- Optional: Go (calculator), `wpctl`/`pactl`/`amixer` (volume), `brightnessctl` (brightness)

## Install

    git clone git@github.com:yuyyuyswallowtail/yuy-router.git
    cd yuy-router
    cp config.example.json config.json    # set the llama-server and model paths
    mkdir -p ~/.local/bin
    ln -sf "$PWD/bin/yuy" ~/.local/bin/yuy

Make sure `~/.local/bin` is in your `PATH`.

## Usage

    yuy "your sentence"
    yuy --list      list commands
    yuy --status    embedding server status
    yuy --stop      stop the embedding server

## Adding a command

Add one object to `commands.json`:

| Field | Purpose |
|---|---|
| `desc` | Description matched against the sentence. The clearer, the more accurate. |
| `cmd` | Command and arguments (no shell). Placeholders: `{ROOT}`, `{LLAMA_SERVER}`. |
| `cwd` | Working directory. `~` for home, `.` for the folder where `yuy` was started. |
| `args` | Optional. `expression` (math expression) or `query` (the whole sentence as one argument). |
| `keywords` | Optional. Words/phrases that add a score bonus. |
| `sudo` | Optional. `true` runs the command with sudo and always asks for confirmation. |
| `timeout` | Optional. Time limit in seconds (default 30, or 3600 for sudo). |

## Safety

- Only commands in `commands.json` can run. The sentence is never executed as a shell command, it is only passed as a single argument.
- Commands whose working directory is inside home run without confirmation (`AUTO_ALLOW_HOME` in `router.py`). Folders outside home always ask first.
- `sudo` commands always ask, whatever the working directory.
- The embedding server only listens on `127.0.0.1`.

## Layout

    router.py        router and embedding server manager
    commands.json    command list
    config.json      llama-server and model paths (not tracked by git)
    bin/yuy          launcher
    math/calc.go     calculator (any operands and operators)
    system/info.py   system info and controls
