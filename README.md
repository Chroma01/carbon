# Carbon

Genomic foundation models from Hugging Face.

Each model release lives in its own directory, with its own dependencies,
training recipes, and evals. Run commands from inside that directory.

| Directory | Models | Contents |
|---|---|---|
| [`carbon/`](carbon) | [Carbon-500M](https://huggingface.co/HuggingFaceBio/Carbon-500M), [Carbon-3B](https://huggingface.co/HuggingFaceBio/Carbon-3B), [Carbon-8B](https://huggingface.co/HuggingFaceBio/Carbon-8B) | Zero-shot DNA evals, finetuning recipes, embedding analysis |

```bash
cd carbon
uv sync
```

## Paper

The Carbon tech report is available on
[bioRxiv](https://www.biorxiv.org/content/10.64898/2026.05.22.727119v1).

## Acknowledgements

Carbon is a joint collaboration between the research teams at Hugging Face, Zhongguancun Academy, and TIGEM/University of Naples “Federico II”.

## License

Apache 2.0.
