# docs/ — project hub

This folder is the working record for the staged-approach prepositioning model. Code history lives in `git log`; everything below is what git cannot tell you.

| File | What it holds | Update when |
|---|---|---|
| `MODEL_CHANGELOG.md` | What each model change means for the written documents (batched, per project convention) | Any model change |
| `DECISIONS.md` | Why we chose X over Y, with the evidence | A choice is made or reversed |
| `ui_handoff/` | Package for the UI build team (brief, stories, data contract, architecture, roadmap) | Scope or data shapes change |
| `formulation_*.md`, `*.tex` | Formulation and method write-ups | Batched pass after the model settles |

Raw assistant-session transcripts are NOT kept in the repo. They contain local paths and personal details, and they are large. They remain in `~/.claude/projects/<working-directory>/`; the distilled conclusions belong in `DECISIONS.md`.
