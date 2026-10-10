# yuy

A semantic command router for the terminal. You type a plain sentence, `yuy` matches it against the commands in `commands.json` using embeddings (EmbeddingGemma 2 through llama.cpp), and runs the best match. There is no chat LLM and no `eval`: only commands you registered yourself can run.

    yuy "calculate (2 + 3) * 4 ^ 2"
    yuy "increase volume 10"
    yuy "set brightness to 60"
    yuy "calculate 7 x 6 and gpu info"
    yuy "show the folder tree depth 2"

## How it works

1. The embedding server (`llama-server --embeddings`) is started automatically in the background. Double starts are prevented with a health check, a process scan, and a file lock.
2. The sentence and each command description are turned into vectors, and the score is the cosine similarity. A command gets a +0.10 bonus when one of its `keywords` appears in the sentence, or when the sentence has a math expression (for `expression` commands).
3. The command runs only when the score is at least 0.75 (`THRESHOLD`) and the gap to the runner-up is at least 0.03 (`MIN_MARGIN`). Otherwise nothing runs.

Sentences can be English or Indonesian: descriptions are English, and `keywords` and operator words (`plus`, `times`, `divided by`, `tambah`, `kali`, `dibagi`, ...) cover both.

## Multiple tasks

One sentence can hold several tasks joined by `and`, `then`, `after that`, `dan`, `lalu`, `kemudian`, `terus`, `serta`, a comma, or a semicolon:

    yuy "calculate (2 + 3) * 4 ^ 2 and gpu info"
    yuy "gpu info, cpu info and battery status"
    yuy "calculate 7 x 6 then what time is it"

- Every part is routed on its own and must pass the same score and gap checks. If any part is unclear, the whole sentence is treated as one task, so sentences like "capacity and status of the battery" still work.
- The tasks run in order, each under its own label (add `-d` to see the plan with scores first). `sudo` tasks ask for confirmation one by one, and exact duplicates run once.
- If a task cannot be built (for example no math expression is found), nothing from the plan runs.
- Give each part enough context: `calculate 3 + 3 and calculate 4 * 5` works, `calculate 3 + 3 and 4 * 5` does not.

## Input and quoting

The shell reads `( ) * & ; | < > $ !` before `yuy` even starts, so a sentence with those characters must be quoted. Three ways to pass text:

    yuy "calculate (2 + 3) * 4"      # quoted
    yuy gpu info                      # fine without quotes when there are no special characters
    yuy                               # interactive prompt: type the sentence as-is, no quoting
    echo 'calculate (2 + 3) * 4' | yuy   # piped input

In the interactive prompt (`yuy> `), type `exit` or press Ctrl-D to quit.

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
    yuy -l, --list      list commands
    yuy -s, --status    embedding server status
    yuy -S, --stop      stop the embedding server
    yuy -h, --help      show the help
    yuy -d, --debug     show scores, matches, and the exact commands that run

By default only the result of each task is printed. With `-d` you also get the top matches, scores, gaps, and the exact command lines.

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

## File and web tools

`tools/files.py` adds file and web commands. Paths must be inside your home folder, and `edit` and `delete` always ask first (`"confirm": true`).
yuy “search TODO in ~/Projects” # by name, falls back to file content
yuy “search content ‘useState’ in ./src”
yuy “read ~/notes.txt lines 10-40”
yuy ‘create ~/notes.txt with “hello”’
yuy “create folder ~/Projects/new-app”
yuy ‘edit ~/notes.txt replace “old” with “new”’ # add the word all to replace every match
yuy ‘edit ~/notes.txt append “one more line”’
yuy “delete ~/old.txt” # moved to ~/.local/share/yuy-trash
yuy “fetch https://example.com” # page text in the terminal
yuy “open https://example.com in browser” # xdg-open

Quote the text you want to write or replace. Edits keep a backup in the trash folder.
