# Sample inputs

Use these as quick checks for the ingestion pipeline (no retrieval / no vector DB).

## Website pages

```bash
python main.py --url https://example.org/about --output ./output/example_web.json
```

## URL list folder (`data/urls`)

Put one or more list files (`*.txt`, `*.url`, or `*.list`) under `./data/urls/`, one URL per line (`#` starts a comment). Then either:

```bash
python main.py --url-dir ./data/urls -o ./output/embedding_chunks.json
```

Or use the convenience script (defaults to `./data/urls` and `./output/embedding_chunks.json`):

```bash
python run_urls_from_data.py --save-intermediate
```

## Institute report PDF

Place a PDF under `./data/institute_reports/` and run:

```bash
python main.py --pdf ./data/institute_reports/report1.pdf --output ./output/report_chunks.json --save-intermediate
```

## Scholarly paper PDF

Place a PDF under `./data/scholarly_papers/` (path hints help classification) and run:

```bash
python main.py --pdf ./data/scholarly_papers/paper1.pdf --output ./output/paper_chunks.json --save-intermediate
```

## Multiple inputs

```bash
python main.py \
  --pdf ./data/institute_reports/report1.pdf ./data/scholarly_papers/paper1.pdf \
  --url https://example.org/about https://example.org/projects \
  --output ./output/embedding_chunks.json \
  --save-intermediate
```

Set `OPENAI_API_KEY` in `.env` when you want LLM-assisted segmentation for long or noisy units.
