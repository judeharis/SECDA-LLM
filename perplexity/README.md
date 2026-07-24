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
