# LEXam Dataset

This repository evaluates only on LEXam.

## Download

1. Download the LEXam JSONL file from your licensed/source distribution.
2. Place it at `data/lexam/lexam.jsonl`.

## Expected Schema

Each JSONL row must include:

- `id`: unique string identifier
- `input`: question text
- `gold`: reference answer (short answer or MCQ label)

Example:

```json
{"id":"lexam-1","input":"Question text","gold":"B"}
```

## Loader

Use `load_lexam(path_or_dir)` from `data/lexam/loader.py`.
If `path_or_dir` is a directory, the loader expects `lexam.jsonl` inside it.
