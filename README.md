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

## Script di supporto

Gli script PowerShell sono per Windows; quelli Bash per macOS e Linux:

| Operazione | Windows | macOS | Linux |
|---|---|---|---|
| Scaricare i modelli | `download_models.ps1` | `download_models.sh` | `download_models.sh` |
| Installare il backend llama.cpp | `install_llama_backend.ps1` | `install_llama_backend.sh` (Metal) | comandi sotto |
| Creare l'archivio del progetto | `create_archive.ps1` | `create_archive.sh` | `create_archive.sh` |

### Download dei modelli

Windows PowerShell:

```powershell
.\scripts\download_models.ps1
.\scripts\download_models.ps1 -Model chat
.\scripts\download_models.ps1 -Model gemma
.\scripts\download_models.ps1 -Model reranker
.\scripts\download_models.ps1 -Model all
```

macOS e Linux (`curl` richiesto):

```bash
bash scripts/download_models.sh embedding
bash scripts/download_models.sh gemma
bash scripts/download_models.sh chat
bash scripts/download_models.sh reranker
bash scripts/download_models.sh all
```

Gli script non riscaricano file esistenti. Per forzarne la sostituzione:

```powershell
.\scripts\download_models.ps1 -Model embedding -Force
```

```bash
FORCE=1 bash scripts/download_models.sh embedding
```

### Backend llama.cpp

Su Windows lo script prova, nell'ordine, CUDA, Vulkan e CPU:

```powershell
.\scripts\install_llama_backend.ps1
```

Su macOS compila la versione bloccata in `uv.lock` con Metal e verifica che
l'offload GPU sia disponibile:

```bash
bash scripts/install_llama_backend.sh
```

Su Linux non viene usato lo script Metal. `uv sync` installa il backend CPU;
per NVIDIA CUDA è disponibile la wheel CUDA 12.4:

```bash
uv pip install --reinstall --no-cache-dir llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Per Vulkan, dopo avere installato SDK Vulkan, compilatore C/C++ e CMake:

```bash
CMAKE_ARGS="-DGGML_VULKAN=on" uv pip install --reinstall --no-cache-dir \
  --no-binary llama-cpp-python llama-cpp-python
```

### Archivio del progetto

Gli script creano `Archive/scout-<timestamp>.zip`, escludendo ambiente virtuale,
build, modelli, database e repository Git. Su macOS/Linux è richiesto `zip`:

```powershell
.\scripts\create_archive.ps1
```

```bash
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

```

`--gpu-layers auto` abilita l'offload quando disponibile; è possibile passare
un numero intero per forzare i layer. Docling preserva titoli, paragrafi e liste;
il chunking usa 900 token con overlap automatico del 15%. Le sezioni da almeno
1 MB usano il line chunker per evitare il comportamento superlineare di HybridChunker.
Formati supportati: PDF, Markdown, testo, reStructuredText, Word (`.doc`, `.docx`),
Excel (`.xls`, `.xlsx`), PowerPoint (`.ppt`, `.pptx`) e OpenDocument
(`.odt`, `.ods`, `.odp`).

Gli ingest successivi elaborano solo file nuovi o modificati ed eliminano dal database
quelli rimossi dalla directory. Un errore di embedding lascia ricercabile l'ultima versione
completa; l'indice FTS esistente viene aggiornato incrementalmente. La ricerca
`hybrid` usa automaticamente il reranker Qwen3 Q8 solo quando
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
mentre `GET /health` verifica che il server sia attivo. La documentazione interattiva
è disponibile su `http://127.0.0.1:8181/docs`, con lo schema OpenAPI grezzo su
`http://127.0.0.1:8181/openapi.json`. Termina il server con `Ctrl-C`.

## Collection registry e update

Le collection possono essere registrate senza indicizzare subito i file. Il registro
predefinito è `index.yml` (JSON valido anche come YAML), e contiene percorso e filtro:

```powershell
uv run python -m src.document_indexer collection add .\corpus `
  --name italia --pattern "**/*.md"
uv run python -m src.document_indexer collection list
uv run python -m src.document_indexer update --collection italia `
  --db-path data\lancedb
uv run python -m src.document_indexer collection delete italia `
  --db-path data\lancedb
uv run python -m src.document_indexer document add .\corpus\nota.md `
  --collection italia --db-path data\lancedb
uv run python -m src.document_indexer document delete <document-id> `
  --db-path data\lancedb
```

`collection add` non copia né modifica i file, ma indicizza subito quelli che
rispettano il pattern, come QMD. `update` sincronizza le collection registrate;
`collection delete` rimuove la registrazione e i documenti indicizzati.

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

- `GET /collections`: collection registrate in `index.yml`.
- `GET /collections/{name}`: dettaglio della collection.
- `POST /collections`: registra e indicizza
  `{"name":"italia","path":"...","pattern":"**/*.md"}`.
- `PUT /collections/{name}`: aggiorna percorso o filtro.
- `DELETE /collections/{name}`: elimina la registrazione e i documenti indicizzati.
- `GET /documents?collection=italia`: elenco dei documenti attivi.
- `GET /documents/{id}`: metadati e testo ricostruito dai chunk.
- `POST /documents` con JSON indicizza un file locale, ad esempio
  `{"path":"C:\\docs\\nota.md","collection":"italia"}`; con multipart supporta
  il drag-and-drop.
- `DELETE /documents/{id}`: elimina un documento e i suoi chunk.
- `GET /status`: conteggi di documenti, chunk e collezioni.
- `GET /config`: configurazione attiva (modelli, gpu_layers, min_score, parametri rerank).
- `PUT /config`: aggiorna le stesse chiavi, le persiste in `config.json` e ricarica
  il modello di embedding se `model_path` cambia. All'avvio vale la priorità
  argomento CLI > `config.json` > default.
- `POST /feedback`: riceve `{"document_id":"...","relevant":true,"query":"..."}`.
- `POST /ingest` con JSON avvia l'update delle collection registrate; con multipart
  mantiene l'ingest drag-and-drop tramite i campi `files` e `collection`.
- `POST /update`: alias JSON per avviare l'update, opzionalmente filtrato da
  `{"collections":["italia"]}`.
- `POST /documents`: alias multipart per aggiungere documenti direttamente.

Il benchmark incluso misura recall, reciprocal rank e latenza sulle query note:

```powershell
uv run python -m src.document_indexer benchmark .\test\benchmark_queries.json `
  --collection italia --db-path data\lancedb
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
