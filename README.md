# Scout

Scout è una pipeline locale per convertire documenti con Docling, creare chunk con
`HybridChunker`, calcolare embeddings GGUF e salvarli in LanceDB.
L'unico entry point applicativo è `src/document_indexer.py`.

## Setup

```powershell
uv sync
```

I modelli e i database restano esterni al progetto:

```text
models/
  Qwen3-Embedding-0.6B-Q8_0.gguf
data/
  lancedb/
```

Scarica il modello embedding predefinito, oppure anche il modello chat:

```powershell
.\scripts\download_models.ps1
.\scripts\download_models.ps1 -Model chat
.\scripts\download_models.ps1 -Model gemma
.\scripts\download_models.ps1 -Model reranker
.\scripts\download_models.ps1 -Model all
```

Su macOS:

```bash
bash scripts/download_models.sh all
bash scripts/install_llama_backend.sh
bash scripts/create_archive.sh
```

## Indicizzazione e ricerca

```powershell
# usa automaticamente la GPU se la build llama.cpp lo supporta
uv run python -m src.document_indexer ingest .\corpus `
  --collection italia `
  --model .\models\Qwen3-Embedding-0.6B-Q8_0.gguf `
  --db-path data\lancedb

# default: 900 token per chunk e overlap automatico del 15%
# per impostare l'overlap esplicitamente:
uv run python -m src.document_indexer ingest .\corpus `
  --collection italia --chunk-size 900 --chunk-overlap 135 `
  --model .\models\Qwen3-Embedding-0.6B-Q8_0.gguf --db-path data\lancedb

# ricerca hybrid (semantica + full-text): senza --collection cerca ovunque
uv run python -m src.document_indexer query "Ecolabel" `
  --collection italia --db-path data\lancedb

# limita il reranking ai primi 12 risultati RRF, 1024 token e contesto 2048
uv run python -m src.document_indexer query "Ecolabel" `
  --collection italia --rerank-candidates 12 --rerank-max-tokens 1024 `
  --reranker-context 2048 `
  --db-path data\lancedb

# più collection: ripetere --collection
uv run python -m src.document_indexer query "bilancio" `
  --collection italia --collection azienda --db-path data\lancedb

# ricerca solo testuale
uv run python -m src.document_indexer query "Ecolabel" `
  --mode fts --db-path data\lancedb

# ricerca ibrida senza reranking
uv run python -m src.document_indexer query "Ecolabel" `
  --mode hybrid --no-rerank --db-path data\lancedb

# forza il reranking anche quando FTS e vector concordano
uv run python -m src.document_indexer query "Ecolabel" `
  --mode hybrid --always-rerank --db-path data\lancedb

# ricostruisce solo l'indice testuale
uv run python -m src.document_indexer reindex-fts --db-path data\lancedb

# installa automaticamente il backend llama.cpp più adatto oppure CPU
.\scripts\install_llama_backend.ps1
```

`--gpu-layers auto` abilita l'offload quando disponibile; è possibile passare
un numero intero per forzare i layer. Docling preserva titoli, paragrafi e liste;
il chunking usa 900 token con overlap automatico del 15%.
Formati supportati: PDF, Markdown, testo, reStructuredText, Word (`.doc`, `.docx`),
Excel (`.xls`, `.xlsx`), PowerPoint (`.ppt`, `.pptx`) e OpenDocument
(`.odt`, `.ods`, `.odp`).

Gli ingest successivi elaborano solo file nuovi o modificati e disattivano quelli
rimossi dalla directory. Un errore di embedding lascia ricercabile l'ultima versione
completa. La ricerca `hybrid` usa automaticamente il reranker Qwen3 Q8 solo quando
i primi risultati FTS e vector non concordano; `--no-rerank` lo disabilita e
`--always-rerank` lo forza. I default sono 12 candidati, 1024 token per documento e
contesto 2048. `--flash-attn` permette il confronto sul backend Metal.

Per più query consecutive, `shell` carica subito l'embedding e carica il reranker al
primo caso ambiguo, riutilizzandolo poi:

```powershell
uv run python -m src.document_indexer shell `
  --collection italia --db-path data\lancedb
```

Per mantenere i modelli caldi tra processi/client diversi, avvia il server locale:

```powershell
uv run python -m src.document_indexer serve `
  --collection italia --db-path data\lancedb
```

Poi invia le query a `http://127.0.0.1:8181/query` con JSON
`{"query":"Ecolabel","rerank":false}`. `GET /collections` restituisce le collezioni attive,
mentre `GET /health` verifica che il server sia attivo; termina con `Ctrl-C`.

`POST /query` accetta:

```json
{
  "query": "come funziona Ecolabel?",
  "collections": ["italia", "azienda"],
  "mode": "hybrid",
  "top_k": 5,
  "rerank": null
}
```

`query` è obbligatorio; `collections` filtra le collezioni; `mode` può essere
`vector`, `fts` o `hybrid`; `top_k` limita i risultati. `rerank` accetta
`true`, `false` o `null`: se assente o `null`, il server usa il reranking
adattivo, che è il default.

Gli altri endpoint sono:

- `GET /documents?collection=italia`: elenco dei documenti attivi.
- `GET /documents/{id}`: metadati e testo ricostruito dai chunk.
- `GET /status`: conteggi di documenti, chunk e collezioni.
- `POST /feedback`: riceve `{"document_id":"...","relevant":true,"query":"..."}`.
- `POST /ingest`: multipart con uno o più campi `files` e il campo
  `collection`; accetta PDF, Markdown, testo e reStructuredText.

Il benchmark incluso misura recall, reciprocal rank e latenza sulle query note:

```powershell
uv run python -m src.document_indexer benchmark .\test\benchmark_queries.json `
  --collection italia --db-path data\lancedb
```

Su macOS, lo script compila la versione bloccata in `uv.lock` con Metal e termina con
errore se l'offload GPU non è disponibile:

```bash
bash scripts/install_llama_backend.sh
```

## Storage

LanceDB contiene le tabelle `documents` e `chunks`, con provenienza, collection,
hash e vettori. `vector` usa la similarità semantica, `fts` l'indice
full-text e `hybrid` fonde entrambe con Reciprocal Rank Fusion.

## Test e build Windows

```powershell
uv run python -m unittest -v test.test_document_indexer
uv sync --extra build
uv run --extra build pyinstaller rag.spec --noconfirm
```
