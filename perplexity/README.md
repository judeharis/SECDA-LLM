# Perplexity Corpus

This folder stores the local corpus used for `llama-perplexity` runs.

## Download Wikitext-2

From the repository root:

```bash
./perplexity/get-wikitext-2.sh
```

After download, the corpus file is:

- `perplexity/wikitext-2-raw/wiki.test.raw`

The VS Code launch configurations for `llama-perplexity` are configured to use this path.

## Quick-check input

`ppl_input.txt` is the short input used for the SECDA migration checks and the
recorded perplexity values in `test_status.json`: the first 4096 bytes of
llama.cpp's `README.md` at `06938ac12` (md5 `b6d844ba5fc667529f6efd3cf3cdd204`).
With MobileLLM-125M-HF Q2_K and `-c 128 --chunks 2 -t 1` it gives 3.3449 at
`-b 128` (x86 simulation and CPU) and, at `-b 16 -ub 16`, 3.3429 in x86
simulation, 3.3637 on the x86 CPU, 3.3636 on the KV260 and 3.3596 on the KV260
CPU.
