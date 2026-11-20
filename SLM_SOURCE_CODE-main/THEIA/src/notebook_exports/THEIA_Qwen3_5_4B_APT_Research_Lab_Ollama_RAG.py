#!/usr/bin/env python
# coding: utf-8

# # THEIA APT Research Lab — Qwen3.5-4B
# 
# **Dataset:** `ISMAIL1216/APT-Detection-Qwen-IsolationForest`, folder `THEIA/`  
# **Objective:** reproducible APT telemetry analysis, evidence-grounded interactive investigation with Qwen3.5-4B, optional model adaptation, and defensible evaluation.
# 
# ## Architecture and research design
# 
# 1. Discover exact THEIA files, formats, schemas, sizes and metadata.
# 2. Load and validate data without assuming column names or file formats.
# 3. Identify event/entity/time/ground-truth/Isolation Forest fields explicitly.
# 4. Establish leakage-resistant train/validation/test splits.
# 5. Evaluate supplied anomaly scores or establish a separate detection baseline.
# 6. Build a searchable evidence index (RAG) over structured THEIA records.
# 7. Query Qwen3.5-4B interactively with multi-turn history and evidence citations.
# 8. Optionally fine-tune with LoRA/QLoRA on a suitable cloud GPU using curated, verified instruction examples.
# 9. Save experiment metadata, outputs, metrics and limitations.
# 
# **Important:** Qwen is the investigation/explanation model, not automatically the intrusion detector. RAG gives the model access to telemetry without changing its weights. Fine-tuning on raw log rows is not a sound detection objective. Keep Isolation Forest/detection metrics separate from LLM answer quality. Never claim detection accuracy without reliable ground-truth labels and a held-out test design.

# ## 0. Hardware strategy
# 
# This notebook is configured for your laptop: **NVIDIA RTX 4050 (6 GB VRAM) + 16 GB system RAM**.
# 
# - Use **Qwen3.5-4B through Ollama** for local, quantized inference. The official Ollama `qwen3.5:4b` package is approximately 3.3–4.0 GB on disk; runtime memory also includes the context/KV cache and Ollama overhead.
# - Start with a **4,096-token context** and moderate generation length. Keep other GPU-heavy applications closed while testing.
# - Ollama manages GPU placement/offload automatically. Do not force all layers onto the GPU; monitor VRAM and system RAM during the first run.
# - Use the laptop for THEIA file inspection, preprocessing, classical ML, RAG and interactive investigation.
# - Do **not** attempt full-parameter training on this laptop. If fine-tuning is justified after dataset validation, plan a separate cloud-GPU LoRA/QLoRA experiment and verify architecture support first.
# - Large THEIA files and embedding jobs must be sampled or processed in batches; do not load everything into RAM blindly.
# 
# The model is an evidence-explanation/investigation assistant. It does not replace Isolation Forest or establish detection accuracy by itself.

# ## 1. Install dependencies

# In[4]:


get_ipython().run_line_magic('pip', 'install -U huggingface_hub pandas pyarrow numpy scikit-learn matplotlib ollama ipywidgets tqdm psutil')


# ## 2. Imports and configuration

# In[5]:


import os, re, json, time, sys, platform, hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np, pandas as pd
from huggingface_hub import HfApi, snapshot_download
from tqdm.auto import tqdm
from IPython.display import display, Markdown, clear_output

REPO_ID = "ISMAIL1216/APT-Detection-Qwen-IsolationForest"
REPO_TYPE = "dataset"
DATA_SUBDIR = "THEIA"
OLLAMA_MODEL = "qwen3.5:4b"
ROOT = Path.cwd() / "theia_apt_research"
RAW_DIR, WORK_DIR, RESULTS_DIR, INDEX_DIR = [ROOT/x for x in ["raw","work","results","index"]]
for p in [RAW_DIR, WORK_DIR, RESULTS_DIR, INDEX_DIR]: p.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now().strftime("THEIA_%Y%m%d_%H%M%S")
SEED = 42
np.random.seed(SEED)
print("Python:", sys.version, "| Platform:", platform.platform())
print("Workspace:", ROOT.resolve())


# ## 3. Discover the exact files in THEIA on Hugging Face
# This is deliberately the first data step. If the repository is private, authenticate with `hf auth login`; do not paste tokens into notebook cells.

# In[6]:


api = HfApi()
repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
theia_files = [f for f in repo_files if f.startswith(DATA_SUBDIR.rstrip("/") + "/")]
if not theia_files:
    raise RuntimeError("No files found under THEIA/. Check repo access and folder name.")
inventory = []
for f in theia_files:
    try:
        info = api.get_paths_info(repo_id=REPO_ID, paths=[f], repo_type=REPO_TYPE)[0]
        size = getattr(info, "size", None)
    except Exception: size = None
    inventory.append({"path": f, "extension": Path(f).suffix.lower(), "size_bytes": size})
inventory_df = pd.DataFrame(inventory).sort_values("path")
display(inventory_df)
print("File count:", len(inventory_df), "| Extensions:", inventory_df.extension.value_counts().to_dict())


# ## 4. Download only THEIA
# Review the inventory first. This downloads the selected subfolder and caches it locally.

# In[7]:


download_root = Path(snapshot_download(
    repo_id=REPO_ID, repo_type=REPO_TYPE,
    allow_patterns=[f"{DATA_SUBDIR}/**"], local_dir=str(RAW_DIR)
))
THEIA_DIR = download_root / DATA_SUBDIR
print("THEIA folder:", THEIA_DIR)


# ## 5. Local manifest and hashes
# Pickle files are not opened automatically: unpickling untrusted files can execute arbitrary code.

# In[8]:


manifest = []
for p in THEIA_DIR.rglob("*"):
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
        manifest.append({"file":str(p.relative_to(THEIA_DIR)), "ext":p.suffix.lower(),
                         "size_mb":round(p.stat().st_size/1024**2,3), "sha256":h.hexdigest()})
manifest_df = pd.DataFrame(manifest).sort_values("file")
display(manifest_df)
manifest_df.to_csv(RESULTS_DIR/f"{RUN_ID}_manifest.csv", index=False)


# ## 6. Load supported tabular files conservatively
# CSV/TSV/JSON/JSONL/Parquet are supported here. Unknown formats (including `.bin`, `.pkl`, graph dumps) are inventoried but not guessed. Large files are sampled.

# In[9]:


MAX_ROWS = 200_000
TABLES, LOAD_AUDIT = {}, []
def load_json_table(path):
    rows=[]
    try:
        with open(path, encoding="utf-8") as f:
            for i,line in enumerate(f):
                if i>=MAX_ROWS: break
                if line.strip(): rows.append(json.loads(line))
        if rows: return pd.json_normalize(rows), "jsonl"
    except Exception: pass
    obj=json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj,list): return pd.json_normalize(obj[:MAX_ROWS]), "json"
    if isinstance(obj,dict): return pd.json_normalize([obj]), "json-object"
    raise ValueError("Unsupported JSON structure")

for p in THEIA_DIR.rglob("*"):
    if not p.is_file(): continue
    ext=p.suffix.lower(); df=None; status="unsupported; inspect format"
    try:
        if ext==".csv": df=pd.read_csv(p,nrows=MAX_ROWS,low_memory=False)
        elif ext in [".tsv",".tab"]: df=pd.read_csv(p,sep="\t",nrows=MAX_ROWS,low_memory=False)
        elif ext in [".json",".jsonl",".ndjson"]: df,fmt=load_json_table(p)
        elif ext in [".parquet",".pq"]:
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(p)
            print("Parquet:",p.name,"rows:",pf.metadata.num_rows,"row groups:",pf.num_row_groups)
            # Read a bounded row-group prefix rather than loading the full file.
            batches=pf.iter_batches(batch_size=50_000)
            batch=next(iter(batches),None)
            if batch is not None: df=batch.to_pandas().head(MAX_ROWS)
        if df is not None:
            key=str(p.relative_to(THEIA_DIR)); TABLES[key]=df; status="loaded sample"
    except Exception as e: status=f"ERROR {type(e).__name__}: {e}"
    LOAD_AUDIT.append({"file":str(p.relative_to(THEIA_DIR)),"extension":ext,
                       "status":status,"rows_loaded":None if df is None else len(df)})
display(pd.DataFrame(LOAD_AUDIT))


# ## 7. Inspect schemas, nulls, duplicates and sample records
# Review the output before selecting an event table. Column suggestions later are only suggestions, not automatic mappings.

# In[10]:


for name,df in TABLES.items():
    print("\n",name,"shape=",df.shape)
    schema=pd.DataFrame({"column":df.columns.astype(str),
        "dtype":[str(x) for x in df.dtypes],
        "null_pct":[round(df[c].isna().mean()*100,2) for c in df.columns],
        "examples":[str(df[c].dropna().astype(str).unique()[:4].tolist()) for c in df.columns]})
    display(schema); display(df.head(3))
    print("duplicate rows:",df.duplicated().sum())


# ## 8. Select the actual event table and map fields explicitly
# Choose only after reviewing the schema. Keep graph metadata, ground truth, and detector outputs as separate sources.

# In[11]:


print("Available tables:")
for k,v in TABLES.items(): print(k, v.shape, list(v.columns))
PRIMARY_TABLE_KEY = None  # Set exact key, e.g. "THEIA/events.parquet"
events = TABLES[PRIMARY_TABLE_KEY].copy() if PRIMARY_TABLE_KEY in TABLES else None

def suggest_fields(cols):
    patterns={
      "timestamp":r"time|timestamp|datetime|date|\bts\b",
      "source_entity":r"subject|source|src|actor|process",
      "target_entity":r"object|target|dest|dst",
      "event_type":r"event|operation|action|syscall|predicate|type",
      "ground_truth":r"ground.?truth|malicious|attack|label|class|target",
      "iforest_score":r"isolation|iforest|anomaly.?score|score",
    }
    return {k:[c for c in cols if re.search(p,str(c),re.I)] for k,p in patterns.items()}
if events is not None: print(json.dumps(suggest_fields(events.columns),indent=2))
else: print("Select PRIMARY_TABLE_KEY after schema review.")


# ## 9. Verified canonical field mapping
# Set only columns that really exist and whose meanings are documented. Never map an Isolation Forest prediction to the ground-truth label.

# In[12]:


FIELD_MAP = {
 "event_time": None, "source_id": None, "target_id": None,
 "event_type": None, "ground_truth": None, "iforest_score": None
}
def normalize_fields(df, mapping):
    out=df.copy()
    for canonical,source in mapping.items():
        if source is not None:
            if source not in out.columns: raise KeyError(source)
            out[canonical]=out[source]
    if "event_time" in out:
        out["event_time"]=pd.to_datetime(out["event_time"],errors="coerce",utc=True)
    return out
events_norm = normalize_fields(events,FIELD_MAP) if events is not None and any(FIELD_MAP.values()) else None
if events_norm is not None: display(events_norm.head())


# ## 10. Leakage-resistant splitting
# Prefer incident/session/graph-level or chronological splits. Random row splitting can leak related events/entities between train and test. Use predefined dataset splits if provided.

# In[13]:


def chronological_split(df,time_col="event_time",train_frac=.70,val_frac=.15):
    if time_col not in df or df[time_col].isna().all():
        raise ValueError("No verified timestamp. Use graph/session/incident/file-level splits instead.")
    d=df.sort_values(time_col).reset_index(drop=True); n=len(d)
    a,b=int(n*train_frac),int(n*(train_frac+val_frac))
    return d.iloc[:a].copy(),d.iloc[a:b].copy(),d.iloc[b:].copy()
# After field mapping and review:
# train_df, val_df, test_df = chronological_split(events_norm)


# ## 11. Evaluate supplied Isolation Forest outputs only against verified ground truth
# If no ground truth exists, report score distributions/ranked investigations, not accuracy, precision, recall or F1.

# In[14]:


from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
def evaluate_binary_scores(y_true,scores,threshold=None):
    z=pd.DataFrame({"y":y_true,"score":scores}).dropna()
    if z.y.nunique()!=2: raise ValueError("Need exactly two verified ground-truth classes.")
    result={"n":len(z),"roc_auc":roc_auc_score(z.y.astype(int),z.score),
            "average_precision":average_precision_score(z.y.astype(int),z.score)}
    if threshold is not None:
        pred=(z.score>=threshold).astype(int)
        result["report"]=classification_report(z.y.astype(int),pred,output_dict=True,zero_division=0)
        result["confusion_matrix"]=confusion_matrix(z.y.astype(int),pred).tolist()
    return result
# metrics = evaluate_binary_scores(test_df["ground_truth"],test_df["iforest_score"])


# ## 12. Create compact evidence documents
# Build documents from verified event rows with source file and row identifiers. For very large data, aggregate by entity/time window and index in batches.

# In[15]:


def safe_value(x):
    if pd.isna(x): return "unknown"
    if isinstance(x,(dict,list)): return json.dumps(x,ensure_ascii=False,default=str)
    return str(x)
documents=[]
if events_norm is not None:
    for rid,row in events_norm.head(100_000).iterrows():
        text="THEIA event evidence: "+"; ".join(f"{c}={safe_value(v)}" for c,v in row.items())
        documents.append({"doc_id":f"{PRIMARY_TABLE_KEY}::row_{rid}",
                          "text":text,"row_id":int(rid),"source_file":PRIMARY_TABLE_KEY})
print("Documents:",len(documents))


# ## 13. Retrieval index (RAG)
# 
# RAG grounds the assistant in retrieved records without updating Qwen weights. Similarity is a retrieval score, **not** an anomaly score. This version uses the local Ollama `nomic-embed-text` embedding model and normalized NumPy cosine similarity. It does not import FAISS or Sentence Transformers, avoiding the Windows DLL blocks encountered in this environment.
# 
# Before running this cell, download the embedding model in a VS Code terminal:
# 
# ```powershell
# ollama pull nomic-embed-text
# ```
# 
# The index is capped at 30,000 evidence documents by default to protect your 16 GB RAM laptop.

# In[16]:


import ollama

# RAG embeddings run through Ollama, avoiding SentenceTransformers, FAISS,
# and the blocked PyArrow datasets import path on this Windows machine.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Conservative laptop limits: build the index in small batches.
MAX_INDEX_DOCS = 30000
EMBED_BATCH_SIZE = 32
index_documents = documents[:MAX_INDEX_DOCS]
embedding_matrix = None

def _response_embeddings(response):
    """Support both dict-like and object-style Ollama responses."""
    values = response.get("embeddings") if isinstance(response, dict) else response.embeddings
    return np.asarray(values, dtype=np.float32)

def ollama_embed(texts):
    """Create normalized embeddings with the local Ollama embedding endpoint."""
    if isinstance(texts, str):
        texts = [texts]
    response = ollama_client.embed(model=EMBED_MODEL, input=texts)
    vectors = _response_embeddings(response)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)

def build_retrieval_index(docs, batch_size=EMBED_BATCH_SIZE):
    if not docs:
        raise ValueError("No evidence documents are available to index.")
    all_vectors = []
    texts = [str(d.get("text", "")) for d in docs]
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding THEIA evidence"):
        batch = texts[start:start + batch_size]
        all_vectors.append(ollama_embed(batch))
    matrix = np.vstack(all_vectors).astype(np.float32)

    # Persist vectors and source documents for reproducibility.
    np.save(INDEX_DIR / "theia_embeddings.npy", matrix)
    with open(INDEX_DIR / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\\n")
    return matrix

if len(documents) > MAX_INDEX_DOCS:
    print(f"Indexing first {MAX_INDEX_DOCS:,} of {len(documents):,} evidence documents.")
    print("Increase MAX_INDEX_DOCS only after checking RAM and index-build time.")

try:
    if index_documents:
        embedding_matrix = build_retrieval_index(index_documents)
        print("RAG index ready:", embedding_matrix.shape)
        print("Embedding model:", EMBED_MODEL)
        print("Retrieval backend: normalized NumPy cosine similarity")
    else:
        print("No evidence documents found. Complete the evidence-document preparation cell first.")
except Exception as e:
    print("Could not build the Ollama embedding index.")
    print("In a VS Code terminal, run: ollama pull nomic-embed-text")
    print("Then verify Ollama is running and rerun this cell.")
    raise

def retrieve_evidence(question, k=8):
    if embedding_matrix is None:
        raise RuntimeError("Build the RAG index first.")
    q = ollama_embed([question])[0]
    # Embeddings are L2-normalized, so dot product equals cosine similarity.
    scores = embedding_matrix @ q
    top_ids = np.argsort(scores)[::-1][:min(k, len(index_documents))]
    return [
        {"similarity": float(scores[i]), **index_documents[int(i)]}
        for i in top_ids
    ]


# In[23]:


# Inspect THEIA pipeline state

print("========== DATASET PIPELINE ==========")

for name in [
    "RAW_DIR",
    "TABLES",
    "PRIMARY_TABLE_KEY",
    "PRIMARY_TABLE",
    "EVENT_FIELD_MAP",
    "documents",
    "EMBED_MODEL",
    "INDEX_DIR"
]:
    value = globals().get(name, "NOT DEFINED")

    if name == "TABLES" and isinstance(value, dict):
        print(f"\n{name}:")
        for key, table in value.items():
            print(
                f"  {key}: {table.shape if hasattr(table, 'shape') else type(table)}"
            )
    elif name == "documents" and isinstance(value, list):
        print(f"{name}: {len(value)} documents")
    else:
        print(f"{name}: {str(value)[:500]}")

print("\n========== POSSIBLE EVIDENCE VARIABLES ==========")

for name in globals():
    if any(word in name.lower() for word in
           ["document", "evidence", "event", "table", "primary"]):
        print(name)


# In[24]:


# THEIA repository and local file investigation

from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

print("=" * 60)
print("1. FILES AVAILABLE IN HUGGING FACE REPOSITORY")
print("=" * 60)

repo_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

theia_files = [
    f for f in repo_files
    if f.lower().startswith("theia/")
]

for f in theia_files:
    print(f)

print("\nTotal THEIA repository files:", len(theia_files))

print("\n" + "=" * 60)
print("2. FILES DOWNLOADED LOCALLY")
print("=" * 60)

local_files = [
    p for p in RAW_DIR.rglob("*")
    if p.is_file()
]

for p in local_files:
    print(
        str(p.relative_to(RAW_DIR)),
        "|",
        round(p.stat().st_size / (1024 * 1024), 3),
        "MB"
    )

print("\nTotal local files:", len(local_files))

print("\n" + "=" * 60)
print("3. SPLIT MANIFEST CONTENT")
print("=" * 60)

manifest = TABLES.get("split_manifest.csv")

if manifest is not None:
    display(manifest.head(10))
    print("Columns:", list(manifest.columns))
else:
    print("Manifest not found in TABLES.")


# In[25]:


# THEIA dataset structure inspection

print("=" * 65)
print("1. DATASET FILE COUNTS")
print("=" * 65)

print(inventory_df.groupby("extension").size())

print("\nFiles by split:")
for split in ["train", "validation", "val", "test"]:
    files = [
        f for f in inventory_df["path"]
        if f.lower().startswith(f"theia/{split}/")
        and f.lower().endswith(".parquet")
    ]
    if files:
        print(f"{split}: {len(files)} Parquet files")

print("\n" + "=" * 65)
print("2. PARQUET FILE SIZE SUMMARY")
print("=" * 65)

parquet_files = inventory_df[
    inventory_df["extension"] == ".parquet"
].copy()

if not parquet_files.empty:
    parquet_files["size_mb"] = (
        parquet_files["size_bytes"].fillna(0) / (1024**2)
    )
    print("Total Parquet files:", len(parquet_files))
    print("Total size (GB):", round(parquet_files["size_mb"].sum()/1024, 2))
    display(parquet_files.groupby(
        parquet_files["path"].str.split("/").str[1]
    )["size_mb"].agg(["count", "sum", "max"]))

print("\n" + "=" * 65)
print("3. SPLIT MANIFEST")
print("=" * 65)

manifest = TABLES.get("split_manifest.csv")

if manifest is not None:
    print("Shape:", manifest.shape)
    print("Columns:", list(manifest.columns))
    display(manifest.to_string(index=False))


# In[26]:


from pathlib import Path

print("Local THEIA files:")

local_parquet = list(THEIA_DIR.rglob("*.parquet"))

print("Parquet files downloaded:", len(local_parquet))

for p in local_parquet[:15]:
    print(p.relative_to(THEIA_DIR))


# In[27]:


# THEIA: inspect Parquet metadata without loading all data

import pyarrow.parquet as pq
from pathlib import Path

print("=" * 65)
print("THEIA PARQUET SCHEMA INSPECTION")
print("=" * 65)

for split in ["train", "validation", "test"]:

    split_dir = THEIA_DIR / split
    files = sorted(split_dir.glob("*.parquet"))

    print(f"\n{'=' * 50}")
    print(f"SPLIT: {split.upper()}")
    print("Available files:", len(files))

    if not files:
        print("No files found.")
        continue

    # Inspect only the first shard
    sample_file = files[0]

    try:
        parquet_file = pq.ParquetFile(sample_file)
        metadata = parquet_file.metadata

        print("Sample file:", sample_file.name)
        print("Rows:", metadata.num_rows)
        print("Row groups:", metadata.num_row_groups)
        print("Columns:", metadata.num_columns)
        print("\nSCHEMA:")
        print(parquet_file.schema)

    except Exception as error:
        print("PARQUET INSPECTION ERROR:", repr(error))


# In[32]:


from huggingface_hub import HfApi
from pathlib import Path

api = HfApi()

print("========== REMOTE FILE STRUCTURE ==========")

remote_files = api.list_repo_files(
    repo_id="ISMAIL1216/APT-Detection-Qwen-IsolationForest",
    repo_type="dataset"
)

for split in ["train", "validation", "test"]:
    files = [
        f for f in remote_files
        if f.startswith(f"THEIA/{split}/")
    ]

    print(f"\n{split.upper()}: {len(files)} files")

    extensions = {}
    for f in files:
        ext = Path(f).suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    print("File formats:", extensions)
    print("First 5 files:")
    for f in files[:5]:
        print(" ", f)


print("\n========== LOCAL FILE STRUCTURE ==========")

for split in ["train", "validation", "test"]:
    folder = THEIA_DIR / split

    if not folder.exists():
        print(split, "FOLDER NOT FOUND:", folder)
        continue

    files = [p for p in folder.iterdir() if p.is_file()]

    print(f"\n{split.upper()}: {len(files)} files")

    extensions = {}
    for f in files:
        ext = f.suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    print("File formats:", extensions)

    for f in files[:5]:
        print(" ", f.name, f.stat().st_size, "bytes")


print("\n========== MANIFEST ==========")

manifest = TABLES.get("split_manifest.csv")

if manifest is not None:
    print(manifest.to_string(index=False))


# In[33]:


from huggingface_hub import HfApi
from pathlib import Path
import pandas as pd

api = HfApi()

print("=" * 60)
print("1. REMOTE TRAIN FILES")
print("=" * 60)

remote_files = api.list_repo_files(
    repo_id=REPO_ID,
    repo_type="dataset"
)

for split in ["train", "validation", "test"]:
    files = [
        f for f in remote_files
        if f.startswith(f"THEIA/{split}/")
    ]

    print(f"\n{split.upper()}: {len(files)} files")

    for f in files[:5]:
        print(" ", f)

    extensions = {}
    for f in files:
        ext = Path(f).suffix.lower()
        extensions[ext] = extensions.get(ext, 0) + 1

    print("Extensions:", extensions)


print("\n" + "=" * 60)
print("2. LOCAL TRAIN FILES")
print("=" * 60)

for split in ["train", "validation", "test"]:
    folder = THEIA_DIR / split
    files = sorted(folder.glob("*")) if folder.exists() else []

    print(f"\n{split.upper()}: {len(files)} files")

    for f in files[:5]:
        print(" ", f.name, f.stat().st_size, "bytes")


print("\n" + "=" * 60)
print("3. COMPLETE SPLIT MANIFEST")
print("=" * 60)

manifest = TABLES.get("split_manifest.csv")

if manifest is not None:
    print(manifest.to_string(index=False))
    print("\nManifest data types:")
    print(manifest.dtypes)


# In[34]:


import duckdb

sample_file = str(
    THEIA_DIR / "train" / "part-00001.parquet"
).replace("\\", "/")

con = duckdb.connect()

print("Testing THEIA Parquet reader...")

schema = con.execute(
    "DESCRIBE SELECT * FROM read_parquet(?)",
    [sample_file]
).fetchdf()

print("\nPARQUET SCHEMA:")
display(schema)

print("\nSAMPLE RECORDS:")

sample = con.execute(
    "SELECT * FROM read_parquet(?) LIMIT 5",
    [sample_file]
).fetchdf()

display(sample)

print("\nRows in sample shard:")

count = con.execute(
    "SELECT COUNT(*) FROM read_parquet(?)",
    [sample_file]
).fetchone()[0]

print(count)


# In[35]:


# ============================================================
# THEIA - FIRST SHARD EXPLORATORY DATA ANALYSIS
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

sample_file = str(
    THEIA_DIR / "train" / "part-00001.parquet"
).replace("\\", "/")

print("=" * 65)
print("1. EVENT TYPE DISTRIBUTION")
print("=" * 65)

event_types = con.execute("""
    SELECT event_type, COUNT(*) AS event_count
    FROM read_parquet(?)
    GROUP BY event_type
    ORDER BY event_count DESC
""", [sample_file]).fetchdf()

display(event_types)

print("\n" + "=" * 65)
print("2. OPERATION DISTRIBUTION")
print("=" * 65)

operations = con.execute("""
    SELECT operation, COUNT(*) AS operation_count
    FROM read_parquet(?)
    GROUP BY operation
    ORDER BY operation_count DESC
""", [sample_file]).fetchdf()

display(operations)

print("\n" + "=" * 65)
print("3. TIMESTAMP RANGE")
print("=" * 65)

timestamps = con.execute("""
    SELECT
        MIN(timestamp_ns) AS min_timestamp,
        MAX(timestamp_ns) AS max_timestamp,
        COUNT(*) AS total_records,
        COUNT(DISTINCT timestamp_ns) AS unique_timestamps
    FROM read_parquet(?)
""", [sample_file]).fetchdf()

display(timestamps)

print("\n" + "=" * 65)
print("4. MISSING VALUE ANALYSIS")
print("=" * 65)

missing = con.execute("""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(*) FILTER (WHERE timestamp_ns IS NULL) AS missing_timestamp,
        COUNT(*) FILTER (WHERE sequence IS NULL) AS missing_sequence,
        COUNT(*) FILTER (WHERE thread_id IS NULL) AS missing_thread_id,
        COUNT(*) FILTER (WHERE event_type IS NULL) AS missing_event_type,
        COUNT(*) FILTER (WHERE operation IS NULL) AS missing_operation,
        COUNT(*) FILTER (WHERE size IS NULL) AS missing_size
    FROM read_parquet(?)
""", [sample_file]).fetchdf()

display(missing)

print("\n" + "=" * 65)
print("5. PRESENCE INDICATOR DISTRIBUTION")
print("=" * 65)

presence = con.execute("""
    SELECT
        subject_present,
        object_present,
        object2_present,
        COUNT(*) AS record_count
    FROM read_parquet(?)
    GROUP BY ALL
    ORDER BY record_count DESC
    LIMIT 20
""", [sample_file]).fetchdf()

display(presence)

print("\nFIRST SHARD ANALYSIS COMPLETED")


# In[36]:


# ============================================================
# THEIA - ADVANCED EVENT BEHAVIOUR ANALYSIS
# First training shard only
# ============================================================

import duckdb
import pandas as pd
import numpy as np

con = duckdb.connect()

sample_file = str(
    THEIA_DIR / "train" / "part-00001.parquet"
).replace("\\", "/")

# 1. Thread activity
print("=" * 60)
print("1. THREAD ACTIVITY")
print("=" * 60)

thread_stats = con.execute("""
    SELECT
        COUNT(DISTINCT thread_id) AS unique_threads,
        MIN(thread_id) AS min_thread,
        MAX(thread_id) AS max_thread,
        COUNT(*) AS total_events
    FROM read_parquet(?)
""", [sample_file]).fetchdf()

display(thread_stats)

# 2. Sequence behaviour
print("\n2. SEQUENCE ANALYSIS")

sequence_stats = con.execute("""
    SELECT
        MIN(sequence) AS min_sequence,
        MAX(sequence) AS max_sequence,
        COUNT(DISTINCT sequence) AS unique_sequences,
        COUNT(*) - COUNT(DISTINCT sequence) AS duplicate_sequence_rows
    FROM read_parquet(?)
""", [sample_file]).fetchdf()

display(sequence_stats)

# 3. Event size distribution
print("\n3. EVENT SIZE DISTRIBUTION")

size_stats = con.execute("""
    SELECT
        COUNT(size) AS valid_sizes,
        AVG(size) AS average_size,
        MIN(size) AS minimum_size,
        MAX(size) AS maximum_size,
        MEDIAN(size) AS median_size,
        STDDEV_POP(size) AS size_stddev
    FROM read_parquet(?)
""", [sample_file]).fetchdf()

display(size_stats)

# 4. Event type and operation relationship
print("\n4. EVENT TYPE / OPERATION RELATIONSHIP")

relationship = con.execute("""
    SELECT
        event_type,
        operation,
        COUNT(*) AS event_count
    FROM read_parquet(?)
    GROUP BY event_type, operation
    ORDER BY event_count DESC
    LIMIT 25
""", [sample_file]).fetchdf()

display(relationship)

# 5. Timestamp interval analysis
print("\n5. TIMESTAMP INTERVALS")

time_stats = con.execute("""
    WITH ordered AS (
        SELECT
            timestamp_ns,
            LAG(timestamp_ns) OVER (
                ORDER BY timestamp_ns
            ) AS previous_timestamp
        FROM read_parquet(?)
    )
    SELECT
        COUNT(*) AS total_intervals,
        MIN(timestamp_ns - previous_timestamp) AS min_interval_ns,
        MEDIAN(timestamp_ns - previous_timestamp) AS median_interval_ns,
        MAX(timestamp_ns - previous_timestamp) AS max_interval_ns
    FROM ordered
    WHERE previous_timestamp IS NOT NULL
""", [sample_file]).fetchdf()

display(time_stats)

print("\nFIRST SHARD BEHAVIOUR ANALYSIS COMPLETED")


# In[37]:


import duckdb
import pandas as pd

con = duckdb.connect()

print("Starting multi-shard THEIA profiling...")

profile_results = []

sample_plan = {
    "train": [1, 2, 3, 50, 100],
    "validation": [1, 2, 3],
    "test": [1, 2, 3]
}

for split, shard_numbers in sample_plan.items():

    for shard_number in shard_numbers:

        file_path = str(
            THEIA_DIR / split /
            f"part-{shard_number:05d}.parquet"
        ).replace("\\", "/")

        query = """
        SELECT
            COUNT(*) AS total_events,
            COUNT(DISTINCT thread_id) AS unique_threads,
            COUNT(DISTINCT event_type) AS unique_event_types,
            COUNT(DISTINCT sequence) AS unique_sequences,
            COUNT(*) - COUNT(DISTINCT sequence)
                AS repeated_sequence_rows,
            AVG(size) AS average_size,
            MEDIAN(size) AS median_size,
            MIN(timestamp_ns) AS min_timestamp,
            MAX(timestamp_ns) AS max_timestamp
        FROM read_parquet(?)
        """

        result = con.execute(
            query, [file_path]
        ).fetchdf()

        result.insert(0, "shard", shard_number)
        result.insert(0, "split", split)

        profile_results.append(result)

        print(f"Completed: {split} shard {shard_number}")

profile_df = pd.concat(
    profile_results,
    ignore_index=True
)

print("\nMULTI-SHARD PROFILE")
display(profile_df)

print("\nEVENT DISTRIBUTION BY SPLIT")

for split, shard_numbers in sample_plan.items():

    files = [
        str(
            THEIA_DIR / split /
            f"part-{number:05d}.parquet"
        ).replace("\\", "/")
        for number in shard_numbers
    ]

    distribution = con.execute("""
        SELECT
            event_type,
            COUNT(*) AS event_count
        FROM read_parquet(?, union_by_name=True)
        GROUP BY event_type
        ORDER BY event_count DESC
    """, [files]).fetchdf()

    print(f"\n{split.upper()}")
    display(distribution)


# In[38]:


# ============================================================
# THEIA: EVENT DISTRIBUTION SHIFT ANALYSIS
# Compare sampled shards across all splits
# ============================================================

import pandas as pd
import duckdb
from pathlib import Path

con = duckdb.connect()

event_profiles = []

for split, shard_numbers in sample_plan.items():

    for shard_number in shard_numbers:

        file_path = str(
            THEIA_DIR / split /
            f"part-{shard_number:05d}.parquet"
        ).replace("\\", "/")

        result = con.execute("""
            SELECT
                event_type,
                COUNT(*) AS event_count
            FROM read_parquet(?)
            GROUP BY event_type
        """, [file_path]).fetchdf()

        total = result["event_count"].sum()

        result["percentage"] = (
            result["event_count"] / total * 100
        )

        result["split"] = split
        result["shard"] = shard_number

        event_profiles.append(result)

event_profiles_df = pd.concat(
    event_profiles,
    ignore_index=True
)

print("EVENT DISTRIBUTION BY INDIVIDUAL SHARD")
display(
    event_profiles_df.sort_values(
        ["split", "shard", "percentage"],
        ascending=[True, True, False]
    )
)

# Average shard-level percentages within each split
print("\nAVERAGE EVENT PERCENTAGES BY SPLIT")

split_comparison = (
    event_profiles_df
    .groupby(["split", "event_type"])["percentage"]
    .mean()
    .reset_index()
    .pivot(
        index="event_type",
        columns="split",
        values="percentage"
    )
    .fillna(0)
)

display(split_comparison.round(4))

# Save analysis for reproducibility
output_path = RESULTS_DIR / "event_distribution_shift.csv"

event_profiles_df.to_csv(output_path, index=False)

print("\nSaved:", output_path)


# In[39]:


# ============================================================
# THEIA: WEIGHTED EVENT DISTRIBUTION COMPARISON
# ============================================================

import numpy as np
import pandas as pd

# Pool actual event counts across sampled shards
pooled = (
    event_profiles_df
    .groupby(["split", "event_type"])["event_count"]
    .sum()
    .reset_index()
)

# Calculate weighted percentages
pooled["percentage"] = (
    pooled["event_count"] /
    pooled.groupby("split")["event_count"].transform("sum")
) * 100

weighted_comparison = (
    pooled
    .pivot(
        index="event_type",
        columns="split",
        values="percentage"
    )
    .fillna(0)
)

print("WEIGHTED EVENT DISTRIBUTIONS (%)")
display(weighted_comparison.round(5))

# Jensen-Shannon divergence
def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / p.sum()
    q = q / q.sum()

    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

event_types = weighted_comparison.index.tolist()

train = weighted_comparison["train"].to_numpy()
validation = weighted_comparison["validation"].to_numpy()
test = weighted_comparison["test"].to_numpy()

print("\nJENSEN-SHANNON DIVERGENCE")
print("Train vs Validation:", js_divergence(train, validation))
print("Train vs Test:", js_divergence(train, test))
print("Validation vs Test:", js_divergence(validation, test))

# Save reproducible results
output_file = RESULTS_DIR / "weighted_event_distribution.csv"
weighted_comparison.to_csv(output_file)

print("\nSaved:", output_file)


# In[40]:


# ============================================================
# THEIA: SPLIT-LEVEL TIMESTAMP AND THREAD ANALYSIS
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()
split_results = []

for split in ["train", "validation", "test"]:

    folder = THEIA_DIR / split

    files = sorted(folder.glob("*.parquet"))

    paths = [
        str(p).replace("\\", "/")
        for p in files
    ]

    print(f"Analysing {split}: {len(paths)} shards")

    result = con.execute("""
        SELECT
            COUNT(*) AS total_events,
            COUNT(DISTINCT thread_id) AS unique_threads,
            COUNT(DISTINCT timestamp_ns) AS unique_timestamps,
            MIN(timestamp_ns) AS first_timestamp_ns,
            MAX(timestamp_ns) AS last_timestamp_ns,
            MIN(sequence) AS minimum_sequence,
            MAX(sequence) AS maximum_sequence,
            COUNT(DISTINCT event_type) AS event_types
        FROM read_parquet(?, union_by_name=True)
    """, [paths]).fetchdf()

    result.insert(0, "split", split)
    split_results.append(result)

split_summary = pd.concat(
    split_results,
    ignore_index=True
)

print("\nFULL SPLIT SUMMARY")
display(split_summary)

# Save results
output = RESULTS_DIR / "split_temporal_thread_summary.csv"
split_summary.to_csv(output, index=False)

print("Saved:", output)


# In[41]:


# ============================================================
# THEIA: TEMPORAL BOUNDARIES AND THREAD OVERLAP
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

split_files = {
    split: [
        str(p).replace("\\", "/")
        for p in sorted((THEIA_DIR / split).glob("*.parquet"))
    ]
    for split in ["train", "validation", "test"]
}

# 1. Timestamp boundaries
print("=" * 65)
print("1. TIMESTAMP BOUNDARIES")
print("=" * 65)

for split, files in split_files.items():

    result = con.execute("""
        SELECT
            MIN(timestamp_ns) AS first_ns,
            MAX(timestamp_ns) AS last_ns,
            MIN(sequence) AS min_sequence,
            MAX(sequence) AS max_sequence
        FROM read_parquet(?, union_by_name=True)
    """, [files]).fetchdf()

    result["first_utc"] = pd.to_datetime(
        result["first_ns"], unit="ns", utc=True
    )
    result["last_utc"] = pd.to_datetime(
        result["last_ns"], unit="ns", utc=True
    )

    print(f"\n{split.upper()}")
    display(result)

# 2. Distinct thread IDs per split
print("\n" + "=" * 65)
print("2. THREAD ID OVERLAP")
print("=" * 65)

for split, files in split_files.items():
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE {split}_threads AS "
        "SELECT DISTINCT thread_id "
        "FROM read_parquet(?, union_by_name=True) "
        "WHERE thread_id IS NOT NULL",
        [files]
    )

for left, right in [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]:
    overlap = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT thread_id FROM {left}_threads
            INTERSECT
            SELECT thread_id FROM {right}_threads
        )
    """).fetchone()[0]

    print(f"{left} ∩ {right}: {overlap:,} shared thread IDs")

print("\nTEMPORAL AND THREAD ANALYSIS COMPLETED")


# In[42]:


# ============================================================
# THEIA: THREAD OVERLAP AND TEMPORAL CONTINUITY
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

split_files = {
    split: [
        str(p).replace("\\", "/")
        for p in sorted((THEIA_DIR / split).glob("*.parquet"))
    ]
    for split in ["train", "validation", "test"]
}

# Build distinct thread sets
for split, files in split_files.items():
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {split}_threads AS
        SELECT DISTINCT thread_id
        FROM read_parquet(?, union_by_name=True)
        WHERE thread_id IS NOT NULL
    """, [files])

# 1. Thread overlap percentages
print("=" * 65)
print("1. THREAD OVERLAP PERCENTAGES")
print("=" * 65)

pairs = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test")
]

for left, right in pairs:

    overlap = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT thread_id FROM {left}_threads
            INTERSECT
            SELECT thread_id FROM {right}_threads
        )
    """).fetchone()[0]

    left_count = con.execute(
        f"SELECT COUNT(*) FROM {left}_threads"
    ).fetchone()[0]

    right_count = con.execute(
        f"SELECT COUNT(*) FROM {right}_threads"
    ).fetchone()[0]

    print(f"\n{left.upper()} vs {right.upper()}")
    print("Shared threads:", f"{overlap:,}")
    print(f"Share of {left}: {overlap/left_count*100:.2f}%")
    print(f"Share of {right}: {overlap/right_count*100:.2f}%")

# 2. Exact temporal boundary gaps
print("\n" + "=" * 65)
print("2. TEMPORAL BOUNDARY GAPS")
print("=" * 65)

bounds = {}

for split, files in split_files.items():
    bounds[split] = con.execute("""
        SELECT
            MIN(timestamp_ns),
            MAX(timestamp_ns)
        FROM read_parquet(?, union_by_name=True)
    """, [files]).fetchone()

for left, right in [
    ("train", "validation"),
    ("validation", "test")
]:
    gap = bounds[right][0] - bounds[left][1]

    print(
        f"{left} -> {right}:",
        gap,
        "nanoseconds"
    )

# 3. Inspect shared thread examples
print("\n" + "=" * 65)
print("3. EXAMPLE THREADS PRESENT IN ALL SPLITS")
print("=" * 65)

common = con.execute("""
    SELECT thread_id
    FROM train_threads
    INTERSECT
    SELECT thread_id FROM validation_threads
    INTERSECT
    SELECT thread_id FROM test_threads
    LIMIT 10
""").fetchdf()

display(common)

print("\nAnalysis completed.")


# In[43]:


# ============================================================
# THEIA: SHARED THREAD CONTINUITY INVESTIGATION
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

# Reuse common thread examples from previous cell
common_ids = common["thread_id"].tolist()

print("Selected shared thread IDs:", common_ids)

thread_history = []

for split, files in split_files.items():

    result = con.execute("""
        SELECT
            thread_id,
            COUNT(*) AS event_count,
            COUNT(DISTINCT event_type) AS event_types,
            MIN(timestamp_ns) AS first_event_ns,
            MAX(timestamp_ns) AS last_event_ns,
            COUNT(DISTINCT sequence) AS unique_sequences
        FROM read_parquet(?, union_by_name=True)
        WHERE thread_id IN (SELECT unnest(?))
        GROUP BY thread_id
        ORDER BY thread_id
    """, [files, common_ids]).fetchdf()

    result["split"] = split
    thread_history.append(result)

history_df = pd.concat(
    thread_history,
    ignore_index=True
)

history_df["first_event_utc"] = pd.to_datetime(
    history_df["first_event_ns"],
    unit="ns",
    utc=True
)

history_df["last_event_utc"] = pd.to_datetime(
    history_df["last_event_ns"],
    unit="ns",
    utc=True
)

print("\nSHARED THREAD ACTIVITY")
display(
    history_df.sort_values(["thread_id", "split"])
)

# Save research output
output = RESULTS_DIR / "shared_thread_continuity.csv"

history_df.to_csv(output, index=False)

print("\nSaved:", output)


# In[44]:


# ============================================================
# THEIA: THREAD IDENTITY AND SOURCE PARTITION ANALYSIS
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

print("=" * 65)
print("1. THREAD ID / SOURCE PARTITION RELATIONSHIP")
print("=" * 65)

for split, files in split_files.items():

    result = con.execute("""
        SELECT
            COUNT(DISTINCT thread_id) AS unique_threads,
            COUNT(DISTINCT source_partition) AS partitions,
            COUNT(DISTINCT (
                CAST(thread_id AS VARCHAR) || '_' ||
                CAST(source_partition AS VARCHAR)
            )) AS unique_thread_partition_pairs
        FROM read_parquet(?, union_by_name=True)
    """, [files]).fetchdf()

    print(f"\n{split.upper()}")
    display(result)


print("\n" + "=" * 65)
print("2. THREADS ASSOCIATED WITH MULTIPLE PARTITIONS")
print("=" * 65)

for split, files in split_files.items():

    result = con.execute("""
        SELECT
            thread_id,
            COUNT(DISTINCT source_partition) AS partition_count,
            COUNT(*) AS event_count
        FROM read_parquet(?, union_by_name=True)
        GROUP BY thread_id
        HAVING COUNT(DISTINCT source_partition) > 1
        ORDER BY partition_count DESC, event_count DESC
        LIMIT 15
    """, [files]).fetchdf()

    print(f"\n{split.upper()}")
    display(result)


print("\n" + "=" * 65)
print("3. SOURCE PARTITION DISTRIBUTION")
print("=" * 65)

for split, files in split_files.items():

    result = con.execute("""
        SELECT
            source_partition,
            COUNT(*) AS event_count,
            COUNT(DISTINCT thread_id) AS unique_threads
        FROM read_parquet(?, union_by_name=True)
        GROUP BY source_partition
        ORDER BY event_count DESC
        LIMIT 20
    """, [files]).fetchdf()

    print(f"\n{split.upper()}")
    display(result)

print("\nTHREAD IDENTITY ANALYSIS COMPLETED")


# In[45]:


# THEIA: concise thread identity summary

for split, files in split_files.items():

    print("\n" + "=" * 55)
    print(split.upper())
    print("=" * 55)

    summary = con.execute("""
        WITH thread_stats AS (
            SELECT
                thread_id,
                COUNT(DISTINCT source_partition) AS partitions,
                COUNT(*) AS events
            FROM read_parquet(?, union_by_name=True)
            GROUP BY thread_id
        )
        SELECT
            COUNT(*) AS total_threads,
            COUNT(*) FILTER (
                WHERE partitions > 1
            ) AS multi_partition_threads,
            MAX(partitions) AS max_partitions_per_thread,
            SUM(events) AS total_events
        FROM thread_stats
    """, [files]).fetchdf()

    display(summary)

    examples = con.execute("""
        SELECT
            thread_id,
            COUNT(DISTINCT source_partition) AS partition_count,
            LIST(DISTINCT source_partition ORDER BY source_partition)
                AS partitions,
            COUNT(*) AS event_count
        FROM read_parquet(?, union_by_name=True)
        GROUP BY thread_id
        HAVING COUNT(DISTINCT source_partition) > 1
        ORDER BY partition_count DESC, event_count DESC
        LIMIT 5
    """, [files]).fetchdf()

    print("Example threads spanning multiple partitions:")
    display(examples)

print("\nIDENTITY SUMMARY COMPLETED")


# In[46]:


# ============================================================
# THEIA: INVESTIGATE THREAD IDENTITY SEMANTICS
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

for split, files in split_files.items():

    print("\n" + "=" * 60)
    print(split.upper(), "- THREAD ID 1")
    print("=" * 60)

    result = con.execute("""
        SELECT
            thread_id,
            source_partition,
            COUNT(*) AS event_count,
            COUNT(DISTINCT event_type) AS event_types,
            MIN(timestamp_ns) AS first_event,
            MAX(timestamp_ns) AS last_event,
            MIN(sequence) AS min_sequence,
            MAX(sequence) AS max_sequence
        FROM read_parquet(?, union_by_name=True)
        WHERE thread_id = 1
        GROUP BY thread_id, source_partition
        ORDER BY source_partition
    """, [files]).fetchdf()

    display(result)

    print("Partition count:", len(result))
    print("Total events:", result["event_count"].sum())

print("\nTHREAD IDENTITY INVESTIGATION COMPLETED")


# In[47]:


# ============================================================
# THEIA: PARTITION TEMPORAL OVERLAP FOR THREAD ID 1
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

for split, files in split_files.items():

    print("\n" + "=" * 65)
    print(f"{split.upper()} - THREAD ID 1")
    print("=" * 65)

    result = con.execute("""
        SELECT
            source_partition,
            COUNT(*) AS events,
            COUNT(DISTINCT event_type) AS event_types,
            MIN(timestamp_ns) AS first_timestamp,
            MAX(timestamp_ns) AS last_timestamp,
            MIN(sequence) AS first_sequence,
            MAX(sequence) AS last_sequence
        FROM read_parquet(?, union_by_name=True)
        WHERE thread_id = 1
        GROUP BY source_partition
        ORDER BY first_timestamp
    """, [files]).fetchdf()

    result["first_utc"] = pd.to_datetime(
        result["first_timestamp"], unit="ns", utc=True
    )

    result["last_utc"] = pd.to_datetime(
        result["last_timestamp"], unit="ns", utc=True
    )

    display(result)

print("\nPARTITION TEMPORAL INVESTIGATION COMPLETED")


# In[48]:


# ============================================================
# THEIA: CHECK PARTITION TIME OVERLAP FOR THREAD ID 1
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()

for split, files in split_files.items():

    print("\n" + "=" * 60)
    print(f"{split.upper()} - PARTITION OVERLAP")
    print("=" * 60)

    result = con.execute("""
        WITH partition_ranges AS (
            SELECT
                source_partition,
                MIN(timestamp_ns) AS first_ns,
                MAX(timestamp_ns) AS last_ns,
                COUNT(*) AS event_count
            FROM read_parquet(?, union_by_name=True)
            WHERE thread_id = 1
            GROUP BY source_partition
        )
        SELECT
            a.source_partition AS partition_a,
            b.source_partition AS partition_b,
            a.event_count AS events_a,
            b.event_count AS events_b,
            GREATEST(a.first_ns, b.first_ns) AS overlap_start,
            LEAST(a.last_ns, b.last_ns) AS overlap_end,
            GREATEST(
                0,
                LEAST(a.last_ns, b.last_ns)
                - GREATEST(a.first_ns, b.first_ns)
            ) AS overlap_duration_ns
        FROM partition_ranges a
        JOIN partition_ranges b
          ON a.source_partition < b.source_partition
        ORDER BY overlap_duration_ns DESC
        LIMIT 15
    """, [files]).fetchdf()

    display(result)

    if not result.empty:
        print(
            "Partition pairs with overlapping time ranges:",
            (result["overlap_duration_ns"] > 0).sum()
        )

print("\nPARTITION OVERLAP ANALYSIS COMPLETED")


# In[50]:


# ============================================================
# THEIA V2: LOW-MEMORY FEATURE ENGINEERING
# One Parquet shard at a time + disk-based aggregation
# ============================================================

import duckdb
import pandas as pd
import time
from pathlib import Path

# ------------------------------------------------------------
# 1. RESOURCE CONFIGURATION
# ------------------------------------------------------------

con = duckdb.connect()

con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")
con.execute("SET preserve_insertion_order = false")
con.execute("SET enable_progress_bar = true")

FEATURE_DIR = RESULTS_DIR / "features_v2"
TEMP_DIR = RESULTS_DIR / "duckdb_temp_v2"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

con.execute(
    "SET temp_directory = '" +
    str(TEMP_DIR).replace("\\", "/").replace("'", "''") +
    "'"
)

WINDOW_NS = 300_000_000_000  # Five minutes

# ------------------------------------------------------------
# 2. PROCESS ONE SHARD AT A TIME
# ------------------------------------------------------------

for split in ["train", "validation", "test"]:

    print("\n" + "=" * 65)
    print("PROCESSING:", split.upper())
    print("=" * 65)

    shard_dir = THEIA_DIR / split
    partial_dir = FEATURE_DIR / f"{split}_partials"

    partial_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(shard_dir.glob("*.parquet"))

    if not shards:
        raise FileNotFoundError(f"No shards found: {shard_dir}")

    print("Total shards:", len(shards))

    start = time.time()

    for i, shard in enumerate(shards, start=1):

        partial_file = partial_dir / f"partial_{i:04d}.parquet"

        # Resume completed work
        if partial_file.exists() and partial_file.stat().st_size > 0:
            print(f"[{i}/{len(shards)}] Already processed")
            continue

        print(f"[{i}/{len(shards)}] Processing {shard.name}")

        input_path = str(shard).replace("\\", "/")
        output_path = str(partial_file).replace("\\", "/")

        output_sql = output_path.replace("'", "''")

        query = f"""
        COPY (
            SELECT
                thread_id,

                FLOOR(
                    CAST(timestamp_ns AS DOUBLE) / {WINDOW_NS}
                )::BIGINT AS window_id,

                source_partition,

                COUNT(*) AS event_count,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_WRITE'
                ) AS write_events,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_READ'
                ) AS read_events,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_OPEN'
                ) AS open_events,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_MMAP'
                ) AS mmap_events,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_EXECUTE'
                ) AS execute_events,

                COUNT(*) FILTER (
                    WHERE event_type = 'EVENT_CONNECT'
                ) AS connect_events,

                COUNT(*) FILTER (
                    WHERE operation = 'write'
                ) AS operation_write_count,

                COUNT(*) FILTER (
                    WHERE operation = 'read'
                ) AS operation_read_count,

                COUNT(*) FILTER (
                    WHERE operation = 'open'
                ) AS operation_open_count,

                COUNT(size) AS valid_size_count,

                COUNT(*) FILTER (
                    WHERE size IS NULL
                ) AS missing_size_count,

                SUM(
                    CASE WHEN size IS NOT NULL
                    THEN CAST(size AS DOUBLE) ELSE 0 END
                ) AS size_sum,

                SUM(
                    CASE WHEN size IS NOT NULL
                    THEN POWER(CAST(size AS DOUBLE), 2)
                    ELSE 0 END
                ) AS size_sumsq,

                MIN(size) AS size_min,
                MAX(size) AS size_max,

                MIN(sequence) AS min_sequence,
                MAX(sequence) AS max_sequence,

                MIN(timestamp_ns) AS first_event_ns,
                MAX(timestamp_ns) AS last_event_ns,

                COUNT(*) FILTER (
                    WHERE subject_present = true
                ) AS subject_present_count,

                COUNT(*) FILTER (
                    WHERE object_present = true
                ) AS object_present_count,

                COUNT(*) FILTER (
                    WHERE object2_present = true
                ) AS object2_present_count

            FROM read_parquet('{input_path}')

            WHERE thread_id IS NOT NULL
              AND timestamp_ns IS NOT NULL

            GROUP BY
                thread_id,
                window_id,
                source_partition

        ) TO '{output_sql}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """

        try:
            con.execute(query)

        except Exception:
            if partial_file.exists():
                partial_file.unlink()
            raise

        print(
            f"   Saved: {partial_file.name} | "
            f"{partial_file.stat().st_size / (1024**2):.2f} MB"
        )

    print(
        f"\nShard processing time: "
        f"{(time.time() - start) / 60:.2f} minutes"
    )

    # --------------------------------------------------------
    # 3. MERGE PARTIAL RESULTS
    # --------------------------------------------------------

    print("\nMerging partial features...")

    partial_files = sorted(partial_dir.glob("partial_*.parquet"))

    if len(partial_files) != len(shards):
        raise RuntimeError(
            f"Expected {len(shards)} partials, "
            f"found {len(partial_files)}"
        )

    final_file = FEATURE_DIR / f"{split}_thread_5m_features.parquet"
    temp_file = FEATURE_DIR / f"{split}_features.tmp.parquet"

    if temp_file.exists():
        temp_file.unlink()

    partial_paths = [
        str(p).replace("\\", "/")
        for p in partial_files
    ]

    temp_sql = str(temp_file).replace("\\", "/").replace("'", "''")

    merge_query = f"""
    COPY (
        SELECT
            thread_id,
            window_id,

            window_id * {WINDOW_NS} AS window_start_ns,

            SUM(event_count) AS event_count,

            COUNT(DISTINCT source_partition)
                AS source_partition_count,

            SUM(write_events) AS write_events,
            SUM(read_events) AS read_events,
            SUM(open_events) AS open_events,
            SUM(mmap_events) AS mmap_events,
            SUM(execute_events) AS execute_events,
            SUM(connect_events) AS connect_events,

            SUM(operation_write_count) AS operation_write_count,
            SUM(operation_read_count) AS operation_read_count,
            SUM(operation_open_count) AS operation_open_count,

            SUM(valid_size_count) AS valid_size_count,
            SUM(missing_size_count) AS missing_size_count,

            SUM(size_sum) AS size_sum,
            SUM(size_sumsq) AS size_sumsq,

            MIN(size_min) AS size_min,
            MAX(size_max) AS size_max,

            MIN(min_sequence) AS min_sequence,
            MAX(max_sequence) AS max_sequence,

            MIN(first_event_ns) AS first_event_ns,
            MAX(last_event_ns) AS last_event_ns,

            MAX(last_event_ns) - MIN(first_event_ns)
                AS event_span_ns,

            SUM(subject_present_count) AS subject_present_count,
            SUM(object_present_count) AS object_present_count,
            SUM(object2_present_count) AS object2_present_count,

            CASE
                WHEN SUM(valid_size_count) > 0
                THEN SUM(size_sum) / SUM(valid_size_count)
                ELSE 0
            END AS size_mean,

            CASE
                WHEN SUM(valid_size_count) > 0
                THEN SQRT(GREATEST(
                    0,
                    SUM(size_sumsq) / SUM(valid_size_count)
                    - POWER(
                        SUM(size_sum) / SUM(valid_size_count), 2
                    )
                ))
                ELSE 0
            END AS size_stddev,

            CASE
                WHEN SUM(event_count) > 0
                THEN SUM(event_count) / 300.0
                ELSE 0
            END AS events_per_second

        FROM read_parquet(?, union_by_name = true)

        GROUP BY thread_id, window_id

    ) TO '{temp_sql}'
    (FORMAT PARQUET, COMPRESSION ZSTD)
    """

    con.execute(merge_query, [partial_paths])

    if final_file.exists():
        final_file.unlink()

    temp_file.replace(final_file)

    # --------------------------------------------------------
    # 4. VERIFY FINAL FEATURE DATASET
    # --------------------------------------------------------

    check = con.execute("""
        SELECT
            COUNT(*) AS feature_rows,
            COUNT(DISTINCT thread_id) AS unique_threads,
            COUNT(DISTINCT window_id) AS unique_windows,
            SUM(event_count) AS represented_events
        FROM read_parquet(?)
    """, [str(final_file).replace("\\", "/")]).fetchdf()

    print("\nFINAL FEATURE DATASET:")
    display(check)

    print("Saved:", final_file)

    # Release memory between splits
    con.execute("CHECKPOINT")

print("\n" + "=" * 65)
print("ALL SPLIT FEATURE DATASETS COMPLETED")
print("=" * 65)


# In[51]:


# ============================================================
# STAGE 2: FEATURE DATASET VALIDATION
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

FEATURE_DIR = RESULTS_DIR / "features_v2"

for split in ["train", "validation", "test"]:

    path = FEATURE_DIR / f"{split}_thread_5m_features.parquet"

    print("\n" + "=" * 60)
    print(split.upper(), "FEATURE VALIDATION")
    print("=" * 60)

    if not path.exists():
        raise FileNotFoundError(path)

    file_path = str(path).replace("\\", "/")

    # Schema
    schema = con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [file_path]
    ).fetchdf()

    print("\nFEATURE COLUMNS:")
    display(schema)

    # Dataset statistics
    summary = con.execute("""
        SELECT
            COUNT(*) AS feature_rows,
            COUNT(DISTINCT thread_id) AS unique_threads,
            COUNT(DISTINCT window_id) AS unique_windows,
            SUM(event_count) AS represented_events,
            MIN(event_count) AS minimum_events,
            MAX(event_count) AS maximum_events,
            COUNT(*) FILTER (
                WHERE event_count IS NULL
            ) AS missing_event_counts
        FROM read_parquet(?)
    """, [file_path]).fetchdf()

    print("\nDATASET SUMMARY:")
    display(summary)

    # Check duplicate thread-window combinations
    duplicates = con.execute("""
        SELECT COUNT(*) AS duplicate_groups
        FROM (
            SELECT thread_id, window_id
            FROM read_parquet(?)
            GROUP BY thread_id, window_id
            HAVING COUNT(*) > 1
        )
    """, [file_path]).fetchdf()

    print("\nDUPLICATE THREAD-WINDOW GROUPS:")
    display(duplicates)

    # Inspect examples
    print("\nSAMPLE FEATURES:")
    sample = con.execute("""
        SELECT *
        FROM read_parquet(?)
        LIMIT 5
    """, [file_path]).fetchdf()

    display(sample)

print("\nFEATURE DATASET VALIDATION COMPLETED")


# In[52]:


# ============================================================
# STAGE 2: ISOLATION FOREST - DATA PREPARATION
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

FEATURE_DIR = RESULTS_DIR / "features_v2"

train_path = FEATURE_DIR / "train_thread_5m_features.parquet"
val_path = FEATURE_DIR / "validation_thread_5m_features.parquet"
test_path = FEATURE_DIR / "test_thread_5m_features.parquet"

# Load aggregated feature datasets, NOT raw telemetry
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

print("TRAIN:", train_df.shape)
print("VALIDATION:", val_df.shape)
print("TEST:", test_df.shape)

# Behavioural feature columns
BASE_FEATURES = [
    "event_count",
    "source_partition_count",
    "write_events",
    "read_events",
    "open_events",
    "mmap_events",
    "execute_events",
    "connect_events",
    "operation_write_count",
    "operation_read_count",
    "operation_open_count",
    "valid_size_count",
    "missing_size_count",
    "size_mean",
    "size_stddev",
    "size_min",
    "size_max",
    "event_span_ns",
    "subject_present_count",
    "object_present_count",
    "object2_present_count",
]

# Verify required columns
missing = [
    c for c in BASE_FEATURES
    if c not in train_df.columns
    or c not in val_df.columns
    or c not in test_df.columns
]

if missing:
    raise ValueError(f"Missing features: {missing}")

def make_model_features(df):

    X = df[BASE_FEATURES].copy()

    # Convert counts and highly skewed magnitudes
    log_columns = [
        "event_count",
        "size_mean",
        "size_stddev",
        "size_min",
        "size_max",
        "event_span_ns",
    ]

    for col in log_columns:
        X[col] = np.log1p(
            np.maximum(X[col].astype(float), 0)
        )

    # Behavioural proportions
    denominator = df["event_count"].clip(lower=1)

    for col in [
        "write_events",
        "read_events",
        "open_events",
        "mmap_events",
        "execute_events",
        "connect_events",
        "operation_write_count",
        "operation_read_count",
        "operation_open_count",
        "subject_present_count",
        "object_present_count",
        "object2_present_count",
        "missing_size_count",
    ]:
        X[col + "_ratio"] = df[col] / denominator

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X.astype("float32")

X_train_all = make_model_features(train_df)
X_val = make_model_features(val_df)
X_test = make_model_features(test_df)

# Reproducible bounded training sample
MAX_TRAIN_ROWS = min(100_000, len(X_train_all))

X_train = X_train_all.sample(
    n=MAX_TRAIN_ROWS,
    random_state=42
)

print("\nMODEL DATA READY")
print("Training sample:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\nFeature columns:", len(X_train.columns))
print("All values finite:", np.isfinite(X_train).all().all())


# In[53]:


# ============================================================
# STAGE 2: TRAIN ISOLATION FOREST
# ============================================================

from sklearn.ensemble import IsolationForest
from pathlib import Path
import joblib
import time

MODEL_DIR = RESULTS_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

start = time.time()

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=2
)

print("Training Isolation Forest...")
model.fit(X_train)

elapsed = time.time() - start

model_path = MODEL_DIR / "theia_isolation_forest.joblib"

joblib.dump(model, model_path)

print("\nMODEL TRAINING COMPLETED")
print("Training rows:", len(X_train))
print("Trees:", model.n_estimators)
print("Training time:", round(elapsed / 60, 2), "minutes")
print("Saved:", model_path)


# In[54]:


# ============================================================
# STAGE 3: VALIDATION + TEST ANOMALY SCORING
# ============================================================

import numpy as np
import pandas as pd

def score_in_batches(model, X, batch_size=25000):

    scores = []

    for start in range(0, len(X), batch_size):

        end = min(start + batch_size, len(X))

        batch = X.iloc[start:end]

        batch_scores = model.decision_function(batch)

        scores.extend(batch_scores)

        print(f"Scored {end:,} / {len(X):,}")

    return np.asarray(scores)


# 1. Score validation
print("=" * 60)
print("VALIDATION SCORING")
print("=" * 60)

val_scores = score_in_batches(model, X_val)

# Select a 1% investigation budget on validation
threshold = np.percentile(val_scores, 1)

print("\nValidation threshold:", threshold)


# 2. Score test
print("\n" + "=" * 60)
print("TEST SCORING")
print("=" * 60)

test_scores = score_in_batches(model, X_test)

test_predictions = test_scores <= threshold


# 3. Distribution summary
print("\n" + "=" * 60)
print("ANOMALY SCORE SUMMARY")
print("=" * 60)

score_summary = pd.DataFrame({
    "split": ["validation", "test"],
    "rows": [len(val_scores), len(test_scores)],
    "minimum_score": [
        val_scores.min(), test_scores.min()
    ],
    "median_score": [
        np.median(val_scores), np.median(test_scores)
    ],
    "maximum_score": [
        val_scores.max(), test_scores.max()
    ],
    "flagged_rows": [
        int((val_scores <= threshold).sum()),
        int(test_predictions.sum())
    ],
    "flagged_percentage": [
        float((val_scores <= threshold).mean() * 100),
        float(test_predictions.mean() * 100)
    ]
})

display(score_summary.round(5))


# 4. Save outputs
val_output = val_df[["thread_id", "window_id"]].copy()
val_output["anomaly_score"] = val_scores
val_output["flagged"] = val_scores <= threshold

test_output = test_df[["thread_id", "window_id"]].copy()
test_output["anomaly_score"] = test_scores
test_output["flagged"] = test_predictions

val_output.to_parquet(
    RESULTS_DIR / "validation_anomaly_scores.parquet",
    index=False
)

test_output.to_parquet(
    RESULTS_DIR / "test_anomaly_scores.parquet",
    index=False
)

print("\nSAVED:")
print("Validation anomaly scores")
print("Test anomaly scores")
print("\nSCORING COMPLETED")


# In[55]:


# ============================================================
# THEIA: FINAL ANOMALY SCORING REPORT
# ============================================================

import pandas as pd
import numpy as np

val_file = RESULTS_DIR / "validation_anomaly_scores.parquet"
test_file = RESULTS_DIR / "test_anomaly_scores.parquet"

val_report = pd.read_parquet(val_file)
test_report = pd.read_parquet(test_file)

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print("Total records:", len(val_report))
print("Flagged:", int(val_report["flagged"].sum()))
print(
    "Flagged percentage:",
    round(val_report["flagged"].mean() * 100, 4), "%"
)

print("\nValidation score statistics:")
display(val_report["anomaly_score"].describe())

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print("Total records:", len(test_report))
print("Flagged:", int(test_report["flagged"].sum()))
print(
    "Flagged percentage:",
    round(test_report["flagged"].mean() * 100, 4), "%"
)

print("\nTest score statistics:")
display(test_report["anomaly_score"].describe())

print("\n" + "=" * 60)
print("MOST ANOMALOUS TEST WINDOWS")
print("=" * 60)

display(
    test_report
    .sort_values("anomaly_score")
    .head(20)
)

print("\nREPORT COMPLETED")


# In[56]:


# ============================================================
# STAGE 4: ANOMALOUS THREAD BEHAVIOUR INVESTIGATION
# ============================================================

import pandas as pd
import numpy as np

# Join anomaly scores with their original features
investigation = test_df.merge(
    test_output[["thread_id", "window_id", "anomaly_score", "flagged"]],
    on=["thread_id", "window_id"],
    how="inner"
)

print("Total joined records:", len(investigation))

# 1. Top 20 anomalous windows
print("\n1. TOP 20 ANOMALOUS WINDOWS")

top_anomalies = (
    investigation
    .sort_values("anomaly_score")
    .head(20)
)

display(top_anomalies[[
    "thread_id",
    "window_id",
    "anomaly_score",
    "event_count",
    "write_events",
    "read_events",
    "open_events",
    "mmap_events",
    "execute_events",
    "connect_events",
    "size_mean",
    "size_max",
    "source_partition_count"
]])

# 2. Compare flagged vs unflagged behaviour
print("\n2. FLAGGED VS UNFLAGGED FEATURE COMPARISON")

compare_columns = [
    "event_count",
    "write_events",
    "read_events",
    "open_events",
    "mmap_events",
    "execute_events",
    "connect_events",
    "size_mean",
    "size_stddev",
    "size_max",
    "source_partition_count"
]

comparison = investigation.groupby("flagged")[
    compare_columns
].median().T

comparison.columns = ["Normal", "Flagged"]

display(comparison.round(3))

# 3. Temporal concentration
print("\n3. ANOMALIES BY TIME WINDOW")

temporal = (
    investigation
    .groupby("window_id")
    .agg(
        total_windows=("flagged", "size"),
        flagged_windows=("flagged", "sum"),
        median_score=("anomaly_score", "median")
    )
    .reset_index()
)

temporal["flagged_percentage"] = (
    temporal["flagged_windows"] /
    temporal["total_windows"] * 100
)

display(
    temporal
    .sort_values("flagged_windows", ascending=False)
    .head(20)
)

# 4. Save investigation data
output = RESULTS_DIR / "test_anomaly_investigation.parquet"

investigation.to_parquet(output, index=False)

print("\nSaved:", output)
print("ANOMALY INVESTIGATION COMPLETED")


# In[58]:


# ============================================================
# FIND ALL DOWNLOADED THEIA PARQUET FILES
# ============================================================

from pathlib import Path

print("RAW DIRECTORY:", RAW_DIR)
print("EXISTS:", RAW_DIR.exists())

print("\nDIRECTORY STRUCTURE:")

for p in RAW_DIR.iterdir():
    print("[DIR]" if p.is_dir() else "[FILE]", p.name)

print("\nSEARCHING FOR PARQUET FILES RECURSIVELY...")

all_parquet = list(RAW_DIR.rglob("*.parquet"))

print("TOTAL PARQUET FILES FOUND:", len(all_parquet))

# Show a few examples grouped by parent directory
from collections import Counter

folder_counts = Counter(
    str(p.parent) for p in all_parquet
)

for folder, count in folder_counts.most_common():
    print(f"{count:4} files | {folder}")

print("\nFIRST 10 FILES:")

for p in all_parquet[:10]:
    print(p)


# In[59]:


# ============================================================
# LOCATE ORIGINAL TEST PARQUET SHARDS
# ============================================================

from pathlib import Path

all_parquet = list(RAW_DIR.rglob("*.parquet"))

test_files = [
    p for p in all_parquet
    if p.parent.name.lower() == "test"
]

print("TEST SHARDS FOUND:", len(test_files))

if not test_files:
    print("No test folder found.")
    print("Check the directory listing from Step 1.")
else:
    print("\nFIRST 10 TEST SHARDS:")
    for p in test_files[:10]:
        print(p)

    print("\nTEST DATA LOCATION:")
    print(test_files[0].parent)


# In[60]:


import duckdb

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

first_raw = str(test_files[0]).replace("\\", "/")

print("ORIGINAL THEIA EVENT SCHEMA:")

display(
    con.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [first_raw]
    ).fetchdf()
)

print("\nORIGINAL EVENT SAMPLES:")

display(
    con.execute(
        "SELECT * FROM read_parquet(?) LIMIT 5",
        [first_raw]
    ).fetchdf()
)


# In[61]:


# ============================================================
# STAGE 5: VERIFY FEATURE-TO-RAW EVENT MAPPING
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

RAW_TEST = RAW_DIR / "THEIA" / "test"
RAW_GLOB = str(RAW_TEST / "*.parquet").replace("\\", "/")

# Select top 5 anomalous test windows
top5 = (
    test_output
    .sort_values("anomaly_score")
    .head(5)[["thread_id", "window_id", "anomaly_score"]]
    .copy()
)

print("TOP 5 ANOMALOUS WINDOWS:")
display(top5)

# Check the timestamp-to-window mapping.
# Candidate: 5-minute windows (300 billion nanoseconds).
mapping_check = con.execute("""
    SELECT
        timestamp_ns,
        thread_id,
        timestamp_ns // 300000000000 AS calculated_window_id
    FROM read_parquet(?)
    LIMIT 10
""", [RAW_GLOB]).fetchdf()

print("\nRAW TIMESTAMP MAPPING:")
display(mapping_check)

# Compare candidate mapping against the actual feature records.
feature_mapping = test_df[
    ["thread_id", "window_id"]
].drop_duplicates()

raw_mapping = con.execute("""
    SELECT
        thread_id,
        timestamp_ns // 300000000000 AS calculated_window_id,
        COUNT(*) AS raw_event_count
    FROM read_parquet(?)
    GROUP BY thread_id, calculated_window_id
""", [RAW_GLOB]).fetchdf()

comparison = feature_mapping.merge(
    raw_mapping,
    left_on=["thread_id", "window_id"],
    right_on=["thread_id", "calculated_window_id"],
    how="inner"
)

print("\nMAPPING VERIFICATION:")
print("Feature thread-window pairs:", len(feature_mapping))
print("Matched raw thread-window pairs:", len(comparison))
print(
    "Mapping coverage:",
    round(len(comparison) / max(len(feature_mapping), 1) * 100, 3),
    "%"
)

display(comparison.head(10))

print("\nMAPPING CHECK COMPLETED")


# In[62]:


# ============================================================
# STAGE 6: RETRIEVE ORIGINAL EVENTS FOR TOP 5 ANOMALIES
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

RAW_TEST = RAW_DIR / "THEIA" / "test"
RAW_GLOB = str(RAW_TEST / "*.parquet").replace("\\", "/")

# Select five most anomalous thread-window pairs
top5 = (
    test_output
    .sort_values("anomaly_score")
    .head(5)[["thread_id", "window_id", "anomaly_score"]]
    .copy()
)

con.register("top5_windows", top5)

print("SELECTED ANOMALOUS WINDOWS:")
display(top5)

# Retrieve original events belonging to these exact pairs
raw_query = """
SELECT
    r.timestamp_ns,
    r.sequence,
    r.thread_id,
    (r.timestamp_ns // 300000000000) AS window_id,
    r.event_type,
    r.operation,
    r.size,
    r.subject_present,
    r.object_present,
    r.object2_present,
    r.source_partition,
    r.source_record_index,
    t.anomaly_score

FROM read_parquet(?) AS r

INNER JOIN top5_windows AS t
    ON r.thread_id = t.thread_id
    AND (r.timestamp_ns // 300000000000) = t.window_id

ORDER BY
    r.thread_id,
    r.timestamp_ns,
    r.sequence
"""

print("\nRETRIEVING ORIGINAL TELEMETRY...")

anomalous_events = con.execute(
    raw_query,
    [RAW_GLOB]
).fetchdf()

print("\nRAW EVENTS RETRIEVED:", len(anomalous_events))

# Convert nanosecond timestamps for readability
anomalous_events["event_time_utc"] = pd.to_datetime(
    anomalous_events["timestamp_ns"],
    unit="ns",
    utc=True
)

print("\nORIGINAL EVENT SAMPLES:")
display(anomalous_events.head(30))

# Save evidence
evidence_path = RESULTS_DIR / "top5_anomalous_raw_events.parquet"

anomalous_events.to_parquet(
    evidence_path,
    index=False
)

print("\nSAVED:", evidence_path)
print("RAW EVENT RETRIEVAL COMPLETED")


# In[63]:


# ============================================================
# STAGE 6: EVENT BEHAVIOUR ANALYSIS
# ============================================================

print("=" * 65)
print("1. EVENT TYPE DISTRIBUTION")
print("=" * 65)

event_distribution = (
    anomalous_events
    .groupby(["thread_id", "window_id", "event_type"])
    .size()
    .reset_index(name="event_count")
    .sort_values(
        ["thread_id", "event_count"],
        ascending=[True, False]
    )
)

display(event_distribution)

print("\n" + "=" * 65)
print("2. OPERATION DISTRIBUTION")
print("=" * 65)

operation_distribution = (
    anomalous_events
    .groupby(["thread_id", "window_id", "operation"])
    .size()
    .reset_index(name="operation_count")
    .sort_values(
        ["thread_id", "operation_count"],
        ascending=[True, False]
    )
)

display(operation_distribution)

print("\n" + "=" * 65)
print("3. THREAD ACTIVITY SUMMARY")
print("=" * 65)

thread_summary = (
    anomalous_events
    .groupby(["thread_id", "window_id"])
    .agg(
        total_events=("sequence", "size"),
        unique_sequences=("sequence", "nunique"),
        unique_event_types=("event_type", "nunique"),
        first_timestamp=("timestamp_ns", "min"),
        last_timestamp=("timestamp_ns", "max"),
        mean_event_size=("size", "mean"),
        max_event_size=("size", "max"),
        anomaly_score=("anomaly_score", "first")
    )
    .reset_index()
    .sort_values("anomaly_score")
)

display(thread_summary)

print("\nEVENT BEHAVIOUR ANALYSIS COMPLETED")


# In[64]:


# ============================================================
# STAGE 7: ANOMALOUS THREADS VS HIGH-ACTIVITY CONTROLS
# ============================================================

import pandas as pd
import numpy as np
import duckdb

# 1. Select five unflagged windows with the highest event counts
test_candidates = test_df.merge(
    test_output[
        ["thread_id", "window_id", "anomaly_score", "flagged"]
    ],
    on=["thread_id", "window_id"],
    how="inner"
)

controls = (
    test_candidates[test_candidates["flagged"] == False]
    .sort_values("event_count", ascending=False)
    .head(5)
    [["thread_id", "window_id", "anomaly_score", "event_count"]]
    .copy()
)

print("HIGH-ACTIVITY CONTROL WINDOWS:")
display(controls)

# 2. Retrieve their original telemetry
con.register("control_windows", controls)

RAW_TEST = RAW_DIR / "THEIA" / "test"
RAW_GLOB = str(RAW_TEST / "*.parquet").replace("\\", "/")

control_events = con.execute("""
    SELECT
        r.timestamp_ns,
        r.sequence,
        r.thread_id,
        r.timestamp_ns // 300000000000 AS window_id,
        r.event_type,
        r.operation,
        r.size,
        r.source_partition
    FROM read_parquet(?) r
    INNER JOIN control_windows c
        ON r.thread_id = c.thread_id
        AND r.timestamp_ns // 300000000000 = c.window_id
    ORDER BY r.thread_id, r.timestamp_ns, r.sequence
""", [RAW_GLOB]).fetchdf()

print("\nCONTROL RAW EVENTS:", len(control_events))

# 3. Build comparable behavioural summaries
def summarize_group(df, group_name):

    result = (
        df.groupby(["thread_id", "window_id"])
        .agg(
            total_events=("sequence", "size"),
            unique_sequences=("sequence", "nunique"),
            unique_event_types=("event_type", "nunique"),
            unique_operations=("operation", "nunique"),
            first_timestamp=("timestamp_ns", "min"),
            last_timestamp=("timestamp_ns", "max"),
            mean_size=("size", "mean"),
            max_size=("size", "max"),
            source_partitions=("source_partition", "nunique")
        )
        .reset_index()
    )

    result["group"] = group_name

    duration = (
        result["last_timestamp"] -
        result["first_timestamp"]
    ).clip(lower=1)

    result["events_per_second"] = (
        result["total_events"] * 1e9 / duration
    )

    return result


flagged_summary = summarize_group(
    anomalous_events, "FLAGGED"
)

control_summary = summarize_group(
    control_events, "CONTROL"
)

print("\nFLAGGED THREAD SUMMARY:")
display(flagged_summary)

print("\nCONTROL THREAD SUMMARY:")
display(control_summary)

# 4. Compare event type distributions
print("\nFLAGGED EVENT TYPES:")
display(
    anomalous_events.groupby("event_type")
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .to_frame()
)

print("\nCONTROL EVENT TYPES:")
display(
    control_events.groupby("event_type")
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .to_frame()
)

# 5. Compare operation distributions
print("\nFLAGGED OPERATIONS:")
display(
    anomalous_events.groupby("operation")
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .to_frame()
)

print("\nCONTROL OPERATIONS:")
display(
    control_events.groupby("operation")
    .size()
    .sort_values(ascending=False)
    .rename("count")
    .to_frame()
)

print("\nFLAGGED VS CONTROL INVESTIGATION COMPLETED")


# In[65]:


# ============================================================
# STAGE 7.1: EXPANDED CONTROL GROUP
# ============================================================

import numpy as np
import pandas as pd

# Get flagged windows and their event-count range
flagged_features = test_candidates[
    test_candidates["flagged"] == True
].copy()

flagged_features["event_count"] = pd.to_numeric(
    flagged_features["event_count"]
)

low_count = flagged_features["event_count"].quantile(0.10)
high_count = flagged_features["event_count"].quantile(0.90)

print("FLAGGED EVENT COUNT RANGE")
print("10th percentile:", low_count)
print("90th percentile:", high_count)

# Select unflagged windows with comparable activity
control_pool = test_candidates[
    (test_candidates["flagged"] == False) &
    (test_candidates["event_count"] >= low_count) &
    (test_candidates["event_count"] <= high_count)
].copy()

print("\nMATCHED CONTROL POOL:", len(control_pool))

if len(control_pool) < 100:
    raise ValueError(
        "Fewer than 100 matched controls. "
        "We need to expand the event-count range."
    )

# Reproducible sample
expanded_controls = control_pool.sample(
    n=100,
    random_state=42
)

display(
    expanded_controls[
        ["thread_id", "window_id", "event_count", "anomaly_score"]
    ].head(20)
)

print("\nEXPANDED CONTROL GROUP READY")


# In[66]:


# ============================================================
# STAGE 7.2: RETRIEVE EXPANDED CONTROL EVENTS
# ============================================================

import duckdb

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

RAW_GLOB = str(
    (RAW_DIR / "THEIA" / "test" / "*.parquet")
).replace("\\", "/")

con.register("expanded_controls", expanded_controls)

expanded_control_events = con.execute("""
    SELECT
        r.timestamp_ns,
        r.sequence,
        r.thread_id,
        r.event_type,
        r.operation,
        r.size,
        r.source_partition
    FROM read_parquet(?) r
    INNER JOIN expanded_controls c
        ON r.thread_id = c.thread_id
        AND r.timestamp_ns // 300000000000 = c.window_id
""", [RAW_GLOB]).fetchdf()

print("EXPANDED CONTROL EVENTS:",
      len(expanded_control_events))

# Calculate event-type percentages
def event_percentages(df, group_name):

    result = (
        df["event_type"]
        .value_counts(normalize=True)
        .mul(100)
        .rename(group_name)
    )

    return result

comparison = pd.concat([
    event_percentages(anomalous_events, "Flagged"),
    event_percentages(expanded_control_events, "Control")
], axis=1).fillna(0)

comparison["Difference"] = (
    comparison["Flagged"] - comparison["Control"]
)

print("\nEVENT TYPE DISTRIBUTION (%)")
display(comparison.round(4))

# Operation proportions
operation_comparison = pd.concat([
    event_percentages(
        anomalous_events.assign(
            event_type=anomalous_events["operation"]
        ),
        "Flagged"
    ),
    event_percentages(
        expanded_control_events.assign(
            event_type=expanded_control_events["operation"]
        ),
        "Control"
    )
], axis=1).fillna(0)

print("\nOPERATION DISTRIBUTION (%)")
display(operation_comparison.round(4))

print("\nEXPANDED COMPARISON COMPLETED")


# In[67]:


# ============================================================
# STAGE 7.3: VERIFY RAW EVENT COUNTS AND EVENT MAPPINGS
# ============================================================

import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("SET memory_limit = '3GB'")
con.execute("SET threads = 1")

RAW_GLOB = str(
    (RAW_DIR / "THEIA" / "test" / "*.parquet")
).replace("\\", "/")

# Use the exact flagged and control windows
flagged_targets = (
    flagged_features[["thread_id", "window_id"]]
    .drop_duplicates()
    .copy()
)
flagged_targets["group_name"] = "FLAGGED"

control_targets = (
    expanded_controls[["thread_id", "window_id"]]
    .drop_duplicates()
    .copy()
)
control_targets["group_name"] = "CONTROL"

targets = pd.concat(
    [flagged_targets, control_targets],
    ignore_index=True
).drop_duplicates(
    ["thread_id", "window_id"]
)

con.register("investigation_targets", targets)

# 1. Count original raw events for every selected window
raw_counts = con.execute("""
    SELECT
        t.group_name,
        t.thread_id,
        t.window_id,
        COUNT(*) AS raw_event_count,
        COUNT(DISTINCT r.sequence) AS unique_sequences
    FROM read_parquet(?) r
    INNER JOIN investigation_targets t
        ON r.thread_id = t.thread_id
        AND r.timestamp_ns // 300000000000 = t.window_id
    GROUP BY
        t.group_name,
        t.thread_id,
        t.window_id
    ORDER BY t.group_name, raw_event_count DESC
""", [RAW_GLOB]).fetchdf()

print("RAW EVENT COUNTS:")
display(
    raw_counts.groupby("group_name")["raw_event_count"]
    .describe()
)

print("\nCOUNTS PER WINDOW:")
display(raw_counts.head(20))

# 2. Compare raw counts with feature counts
feature_counts = test_candidates[
    ["thread_id", "window_id", "event_count", "flagged"]
].drop_duplicates(
    ["thread_id", "window_id"]
)

count_check = raw_counts.merge(
    feature_counts,
    on=["thread_id", "window_id"],
    how="left"
)

count_check["count_difference"] = (
    count_check["raw_event_count"] -
    count_check["event_count"]
)

print("\nFEATURE COUNT VS RAW COUNT:")
display(count_check.head(30))

# 3. Inspect event_type vs operation mapping
mapping = con.execute("""
    SELECT
        r.event_type,
        r.operation,
        COUNT(*) AS event_count
    FROM read_parquet(?) r
    INNER JOIN investigation_targets t
        ON r.thread_id = t.thread_id
        AND r.timestamp_ns // 300000000000 = t.window_id
    GROUP BY r.event_type, r.operation
    ORDER BY event_count DESC
""", [RAW_GLOB]).fetchdf()

print("\nEVENT TYPE / OPERATION CROSS-TAB:")
display(mapping)

print("\nMAPPING AND COUNT DIAGNOSTICS COMPLETED")


# In[68]:


# ============================================================
# STAGE 7.4: GROUP-WISE EVENT MAPPING AND COUNT VALIDATION
# ============================================================

# 1. Check count consistency separately
print("COUNT VALIDATION BY GROUP")

count_check["count_match"] = (
    count_check["raw_event_count"] ==
    count_check["event_count"]
)

display(
    count_check.groupby("group_name").agg(
        windows=("thread_id", "count"),
        raw_events=("raw_event_count", "sum"),
        feature_events=("event_count", "sum"),
        mismatches=("count_match", lambda x: (~x).sum()),
        mean_raw_events=("raw_event_count", "mean"),
        max_raw_events=("raw_event_count", "max")
    ).round(2)
)

# 2. Retrieve separate event-type/operation counts
group_mapping = con.execute("""
    SELECT
        t.group_name,
        r.event_type,
        r.operation,
        COUNT(*) AS event_count
    FROM read_parquet(?) r
    INNER JOIN investigation_targets t
        ON r.thread_id = t.thread_id
        AND r.timestamp_ns // 300000000000 = t.window_id
    GROUP BY
        t.group_name,
        r.event_type,
        r.operation
    ORDER BY
        t.group_name,
        event_count DESC
""", [RAW_GLOB]).fetchdf()

# 3. Calculate percentages within each group
group_mapping["percentage"] = (
    group_mapping["event_count"] /
    group_mapping.groupby("group_name")["event_count"]
    .transform("sum") * 100
)

print("\nFLAGGED EVENT TYPE / OPERATION MAPPING")
display(
    group_mapping[
        group_mapping["group_name"] == "FLAGGED"
    ].head(30).round(4)
)

print("\nCONTROL EVENT TYPE / OPERATION MAPPING")
display(
    group_mapping[
        group_mapping["group_name"] == "CONTROL"
    ].head(30).round(4)
)

# 4. Compare event-type percentages directly
event_type_comparison = group_mapping.pivot_table(
    index="event_type",
    columns="group_name",
    values="percentage",
    aggfunc="sum",
    fill_value=0
)

event_type_comparison["difference"] = (
    event_type_comparison["FLAGGED"] -
    event_type_comparison["CONTROL"]
)

print("\nGROUP-WISE EVENT TYPE COMPARISON (%)")
display(event_type_comparison.round(4))

# 5. Compare operation percentages directly
operation_comparison = group_mapping.pivot_table(
    index="operation",
    columns="group_name",
    values="event_count",
    aggfunc="sum",
    fill_value=0
)

operation_comparison = (
    operation_comparison.div(
        operation_comparison.sum(axis=0),
        axis=1
    ) * 100
)

operation_comparison["difference"] = (
    operation_comparison["FLAGGED"] -
    operation_comparison["CONTROL"]
)

print("\nGROUP-WISE OPERATION COMPARISON (%)")
display(operation_comparison.round(4))

print("\nSTAGE 7.4 COMPLETED")


# In[ ]:


# ================================================================
# STAGE 8: COMPLETE THEIA RAG + QWEN3.5-4B LOCAL INVESTIGATION
# ================================================================
# Architecture:
# THEIA evidence -> Ollama embeddings -> NumPy retrieval
# -> Qwen3.5-4B -> Interactive multi-turn investigation
#
# No FAISS, sentence-transformers, or PyArrow imports.
# ================================================================

import os
import json
import time
import requests
import numpy as np
import pandas as pd
import ipywidgets as widgets

from pathlib import Path
from datetime import datetime, timezone
from IPython.display import display, Markdown, clear_output

# ------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------

OLLAMA_URL = "http://localhost:11434"

LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 6
EMBED_BATCH_SIZE = 24

INDEX_DIR = Path(INDEX_DIR) if "INDEX_DIR" in globals() else (
    Path.cwd() / "theia_apt_research" / "index"
)

INDEX_DIR.mkdir(parents=True, exist_ok=True)

INDEX_FILE = INDEX_DIR / "theia_qwen_embeddings.npy"
DOCS_FILE = INDEX_DIR / "theia_qwen_evidence.json"

print("=" * 65)
print("THEIA QWEN3.5-4B RESEARCH SYSTEM")
print("=" * 65)

print("LLM:", LLM_MODEL)
print("Embedding model:", EMBED_MODEL)
print("Index directory:", INDEX_DIR)

# ------------------------------------------------
# 2. CHECK OLLAMA CONNECTION AND MODELS
# ------------------------------------------------

response = requests.get(
    f"{OLLAMA_URL}/api/tags",
    timeout=20
)

response.raise_for_status()

installed_models = [
    item["name"]
    for item in response.json().get("models", [])
]

print("\nINSTALLED OLLAMA MODELS:")
for model in installed_models:
    print("-", model)

def model_exists(required):
    return any(
        name == required or name.startswith(required + ":")
        for name in installed_models
    )

if not model_exists(LLM_MODEL):
    raise RuntimeError(
        f"{LLM_MODEL} is missing. Run: ollama pull {LLM_MODEL}"
    )

if not model_exists(EMBED_MODEL):
    raise RuntimeError(
        f"{EMBED_MODEL} is missing. Run: ollama pull {EMBED_MODEL}"
    )

print("\nOLLAMA CONNECTION: SUCCESS")

# ------------------------------------------------
# 3. OLLAMA EMBEDDING FUNCTION
# ------------------------------------------------

def get_embeddings(texts):

    if isinstance(texts, str):
        texts = [texts]

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    embeddings = np.asarray(
        result["embeddings"],
        dtype=np.float32
    )

    if embeddings.ndim != 2:
        raise ValueError("Invalid embedding dimensions.")

    return embeddings


# Test embedding model
test_embedding = get_embeddings(
    ["THEIA APT telemetry investigation test."]
)

EMBEDDING_DIM = test_embedding.shape[1]

print("Embedding model: WORKING")
print("Embedding dimensions:", EMBEDDING_DIM)

# ------------------------------------------------
# 4. BUILD EVIDENCE DOCUMENTS FROM NOTEBOOK RESULTS
# ------------------------------------------------

documents = []

def add_document(doc_id, title, content, category="research"):

    content = str(content).strip()

    if not content:
        return

    documents.append({
        "doc_id": str(doc_id),
        "title": str(title),
        "category": str(category),
        "text": content
    })


def add_dataframe_document(
    variable_name,
    title,
    category="analysis",
    max_rows=80
):

    obj = globals().get(variable_name)

    if not isinstance(obj, pd.DataFrame):
        print("Skipping unavailable:", variable_name)
        return

    if obj.empty:
        print("Skipping empty:", variable_name)
        return

    preview = obj.head(max_rows).to_string(index=False)

    content = (
        f"Evidence source: {variable_name}\n"
        f"Title: {title}\n"
        f"Rows: {len(obj)}\n"
        f"Columns: {list(obj.columns)}\n\n"
        f"{preview}"
    )

    add_document(
        f"THEIA-{len(documents)+1:04d}",
        title,
        content,
        category
    )

# 4A. Overall dataset information

add_document(
    "THEIA-OVERVIEW",
    "THEIA dataset overview",
    """
    Dataset: THEIA APT telemetry.

    Original event fields inspected:
    timestamp_ns, sequence, thread_id, event_type,
    operation, size, subject_present, object_present,
    object2_present, source_partition, source_record_index.

    Dataset splits:
    train, validation, test.

    The investigation uses engineered thread/time-window
    features and Isolation Forest anomaly scores.

    Anomaly scores are statistical signals, not proof
    of malicious activity.
    """,
    "dataset"
)

# 4B. Stage 7.4 comparison outputs

for variable, title, category in [
    ("group_mapping", "Flagged and control event-operation mapping", "comparison"),
    ("event_type_comparison", "Event type percentage comparison", "comparison"),
    ("operation_comparison", "Operation percentage comparison", "comparison"),
    ("count_check", "Raw event count verification", "validation"),
    ("flagged_summary", "Flagged thread behavioural summary", "anomaly"),
    ("control_summary", "Control thread behavioural summary", "control"),
    ("flagged_features", "Flagged feature windows", "anomaly"),
    ("expanded_controls", "Expanded control windows", "control"),
    ("test_output", "Isolation Forest test scoring", "model"),
    ("test_candidates", "Test candidate windows", "model")
]:
    add_dataframe_document(
        variable,
        title,
        category,
        max_rows=80
    )

# 4C. Raw event behaviour summaries

def add_raw_event_summary(variable_name, group_name):

    df = globals().get(variable_name)

    if not isinstance(df, pd.DataFrame) or df.empty:
        return

    required = {
        "event_type",
        "operation"
    }

    if not required.issubset(df.columns):
        return

    event_counts = (
        df["event_type"]
        .value_counts()
        .head(30)
        .to_string()
    )

    operation_counts = (
        df["operation"]
        .value_counts()
        .head(30)
        .to_string()
    )

    content = f"""
    GROUP: {group_name}

    Total retrieved events: {len(df)}

    EVENT TYPE DISTRIBUTION:
    {event_counts}

    OPERATION DISTRIBUTION:
    {operation_counts}
    """

    add_document(
        f"THEIA-{len(documents)+1:04d}",
        f"{group_name} raw event distribution",
        content,
        group_name.lower()
    )


add_raw_event_summary(
    "anomalous_events",
    "FLAGGED"
)

add_raw_event_summary(
    "expanded_control_events",
    "CONTROL"
)

# 4D. Selected anomalous windows

raw_df = globals().get("anomalous_events")

if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:

    group_columns = [
        c for c in ["thread_id", "window_id"]
        if c in raw_df.columns
    ]

    if group_columns:

        for keys, group in raw_df.groupby(group_columns):

            if not isinstance(keys, tuple):
                keys = (keys,)

            thread = keys[0]
            window = keys[1] if len(keys) > 1 else "UNKNOWN"

            event_profile = (
                group["event_type"]
                .value_counts()
                .head(15)
                .to_dict()
            )

            operation_profile = (
                group["operation"]
                .value_counts()
                .head(15)
                .to_dict()
            )

            content = f"""
            FLAGGED THREAD INVESTIGATION

            Thread ID: {thread}
            Window ID: {window}

            Retrieved raw events: {len(group)}

            Event types:
            {json.dumps(event_profile)}

            Operations:
            {json.dumps(operation_profile)}

            This window was selected by the anomaly
            investigation pipeline. Its status is not
            independently verified as malicious.
            """

            add_document(
                f"THEIA-WINDOW-{thread}-{window}",
                f"Flagged thread {thread}, window {window}",
                content,
                "anomaly_window"
            )

print("\nEVIDENCE DOCUMENTS CREATED:", len(documents))

if not documents:
    raise RuntimeError(
        "No evidence documents were created. "
        "Check the Stage 7.4 variables."
    )

# ------------------------------------------------
# 5. EMBED ALL EVIDENCE DOCUMENTS
# ------------------------------------------------

print("\nBUILDING EVIDENCE EMBEDDINGS...")

texts = [
    doc["title"] + "\n" + doc["text"]
    for doc in documents
]

embedding_batches = []

for start in range(0, len(texts), EMBED_BATCH_SIZE):

    batch = texts[
        start:start + EMBED_BATCH_SIZE
    ]

    batch_embeddings = get_embeddings(batch)

    embedding_batches.append(batch_embeddings)

    print(
        f"Embedded {min(start + len(batch), len(texts))}"
        f" / {len(texts)}"
    )

vector_matrix = np.vstack(
    embedding_batches
).astype(np.float32)

# Normalize vectors for cosine similarity
norms = np.linalg.norm(
    vector_matrix,
    axis=1,
    keepdims=True
)

norms[norms == 0] = 1

vector_matrix = vector_matrix / norms

# Save local evidence index
np.save(INDEX_FILE, vector_matrix)

with open(DOCS_FILE, "w", encoding="utf-8") as f:
    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\nRAG INDEX CREATED")
print("Documents:", len(documents))
print("Embedding shape:", vector_matrix.shape)
print("Saved:", INDEX_FILE)
print("Saved:", DOCS_FILE)

# ------------------------------------------------
# 6. RETRIEVE RELEVANT EVIDENCE
# ------------------------------------------------

def retrieve_evidence(question, top_k=TOP_K):

    query_vector = get_embeddings(question)

    query_vector = query_vector[0]

    query_norm = np.linalg.norm(query_vector)

    if query_norm == 0:
        raise ValueError("Invalid query embedding.")

    query_vector = query_vector / query_norm

    similarities = vector_matrix @ query_vector

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for idx in top_indices:

        doc = documents[int(idx)]

        results.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "category": doc["category"],
            "similarity": float(similarities[idx]),
            "text": doc["text"]
        })

    return results


# Test retrieval
retrieval_test = retrieve_evidence(
    "Compare flagged and control event type distributions."
)

print("\nRETRIEVAL TEST:")

for item in retrieval_test:
    print(
        item["doc_id"],
        round(item["similarity"], 4),
        item["title"]
    )

# ------------------------------------------------
# 7. QWEN3.5-4B INVESTIGATION ENGINE
# ------------------------------------------------

SYSTEM_PROMPT = """
You are a cybersecurity research assistant investigating
THEIA system-event telemetry.

You are running locally using Qwen3.5-4B.

STRICT RULES:

1. Use the supplied retrieved evidence as the factual basis.
2. Cite evidence using its exact document ID:
   [THEIA-XXXX].
3. Never invent process names, file paths, IP addresses,
   attacker identities, malware families, or attack stages.
4. Clearly distinguish:
   - Observed facts
   - Analytical interpretation
   - Hypotheses
   - Unknown information
5. An Isolation Forest anomaly score is not a probability
   of malicious activity.
6. Do not call an event malicious solely because it is rare.
7. Compare flagged and control behaviour when relevant.
8. If evidence is insufficient, explicitly say so.
9. Explain technical findings in clear research language.
10. Recommend the next investigation step where useful.
"""


conversation_history = []


def ask_qwen(question, history=None):

    if history is None:
        history = []

    evidence = retrieve_evidence(question)

    evidence_context = "\n\n".join([
        (
            f"[{item['doc_id']}] "
            f"TITLE: {item['title']}\n"
            f"CATEGORY: {item['category']}\n"
            f"SIMILARITY: {item['similarity']:.4f}\n"
            f"EVIDENCE:\n{item['text']}"
        )
        for item in evidence
    ])

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Retain recent multi-turn conversation
    messages.extend(history[-8:])

    messages.append({
        "role": "user",
        "content": f"""
QUESTION:
{question}

RETRIEVED THEIA EVIDENCE:
{evidence_context}

Investigate using the supplied evidence.
Cite the relevant evidence IDs.
"""
    })

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 800,
                "num_ctx": 8192
            }
        },
        timeout=600
    )

    response.raise_for_status()

    answer = response.json()["message"]["content"]

    return answer, evidence


# ------------------------------------------------
# 8. INTERACTIVE MULTI-TURN RESEARCH INTERFACE
# ------------------------------------------------

history = []
session_records = []

question_box = widgets.Textarea(
    value="Compare the anomalous and control event behaviour.",
    placeholder="Ask a question about THEIA telemetry...",
    description="Question:",
    layout=widgets.Layout(
        width="100%",
        height="90px"
    )
)

ask_button = widgets.Button(
    description="Investigate with Qwen",
    button_style="primary",
    icon="search",
    layout=widgets.Layout(width="230px")
)

reset_button = widgets.Button(
    description="Reset session",
    button_style="warning"
)

export_button = widgets.Button(
    description="Export session",
    button_style="success"
)

status = widgets.HTML(
    value="<b>Ready:</b> RAG index created."
)

output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #777",
        padding="12px",
        width="100%",
        max_height="600px",
        overflow="auto"
    )
)


def investigate_click(_):

    question = question_box.value.strip()

    if not question:
        status.value = "Enter a question first."
        return

    ask_button.disabled = True

    status.value = (
        "<b>Working:</b> Retrieving evidence and querying Qwen..."
    )

    try:

        answer, evidence = ask_qwen(
            question,
            history
        )

        history.append({
            "role": "user",
            "content": question
        })

        history.append({
            "role": "assistant",
            "content": answer
        })

        session_records.append({
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "question": question,
            "answer": answer,
            "evidence": [
                {
                    "doc_id": e["doc_id"],
                    "title": e["title"],
                    "similarity": e["similarity"]
                }
                for e in evidence
            ]
        })

        with output:

            clear_output(wait=True)

            display(Markdown("## Qwen3.5-4B Investigation"))

            display(Markdown("### Question"))
            display(Markdown(question))

            display(Markdown("### Answer"))
            display(Markdown(answer))

            display(Markdown("### Retrieved evidence"))

            display(pd.DataFrame([
                {
                    "Evidence ID": e["doc_id"],
                    "Title": e["title"],
                    "Similarity": round(e["similarity"], 4)
                }
                for e in evidence
            ]))

        status.value = (
            "<b>Completed:</b> Investigation finished."
        )

    except Exception as error:

        status.value = (
            f"<b>Error:</b> {type(error).__name__}: {error}"
        )

    finally:

        ask_button.disabled = False


def reset_click(_):

    global history

    history = []
    session_records.clear()

    with output:
        clear_output()

    status.value = "<b>Ready:</b> New investigation session."


def export_click(_):

    export_file = INDEX_DIR / (
        "qwen_investigation_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    with open(export_file, "w", encoding="utf-8") as f:

        json.dump(
            session_records,
            f,
            indent=2,
            ensure_ascii=False
        )

    status.value = (
        f"<b>Exported:</b> {export_file}"
    )


ask_button.on_click(investigate_click)
reset_button.on_click(reset_click)
export_button.on_click(export_click)

display(
    widgets.VBox([
        widgets.HTML(
            "<h2>THEIA APT Investigation — Qwen3.5-4B</h2>"
            "<p>Evidence-grounded local multi-turn research assistant</p>"
        ),
        question_box,
        widgets.HBox([
            ask_button,
            reset_button,
            export_button
        ]),
        status,
        output
    ])
)

print("\n" + "=" * 65)
print("THEIA QWEN3.5-4B SYSTEM INITIALIZED")
print("=" * 65)
print("RAG documents:", len(documents))
print("Embedding dimensions:", EMBEDDING_DIM)
print("LLM:", LLM_MODEL)
print("Ready for interactive investigation.")


# ## 14. Qwen3.5-4B local inference via Ollama
# 
# This notebook uses **`qwen3.5:4b`**, not the 9B model.
# 
# 1. Install Ollama for Windows from the official site: https://ollama.com/download
# 2. In a VS Code terminal (the Python `.venv` may remain active), run:
#    ```powershell
#    ollama pull qwen3.5:4b
#    ```
# 3. Test the model:
#    ```powershell
#    ollama run qwen3.5:4b
#    ```
# 4. Exit the chat with `/bye`. Keep the Ollama application/service running.
# 5. Run the next code cell in this notebook to verify the connection and model tag.
# 
# **Memory profile:** the Ollama library lists the 4B package at approximately 3.3–4.0 GB on disk. This is not the same as total runtime memory. The notebook starts at a 4K context to leave room for runtime overhead on a 6 GB GPU. If you see out-of-memory errors, close GPU-heavy apps and keep the context at 4K; do not increase it yet.
# 
# Official model page: https://ollama.com/library/qwen3.5:4b
# 

# In[71]:


import requests
import json

BASE = "http://localhost:11434"

# Check Qwen's supported thinking controls
info = requests.post(
    f"{BASE}/api/show",
    json={"model": "qwen3.5:4b"},
    timeout=60
)
info.raise_for_status()

print("MODEL THINKING CONFIGURATION:")
print(json.dumps(info.json().get("thinking", {}), indent=2))

# Test chat with thinking disabled
payload = {
    "model": "qwen3.5:4b",
    "messages": [
        {
            "role": "user",
            "content": "Reply with exactly: QWEN LOCAL INFERENCE WORKING"
        }
    ],
    "think": False,
    "stream": False,
    "options": {
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 512
    }
}

r = requests.post(
    f"{BASE}/api/chat",
    json=payload,
    timeout=600
)
r.raise_for_status()

data = r.json()
message = data.get("message", {})

print("\nMODEL:", data.get("model"))
print("CONTENT:", repr(message.get("content")))
print("THINKING:", repr(message.get("thinking")))
print("DONE:", data.get("done"))
print("DONE REASON:", data.get("done_reason"))
print("TOKENS GENERATED:", data.get("eval_count"))
print("\nFULL RESPONSE:")
print(json.dumps(data, indent=2))


# ## 17. Optional supervised fine-tuning data
# 
# Only create instruction/answer pairs from reviewed analyst annotations, documented ground-truth descriptions, or carefully checked evidence summaries. Do not convert Isolation Forest scores into fake ground-truth labels. Keep train/validation/test separation at incident/session/entity level.
# 
# Example schema:
# ```json
# {"instruction":"Summarize the observed event sequence for entity X.","input":"[verified evidence IDs and event snippets]","output":"Evidence-based summary with citations and limitations."}
# ```

# In[ ]:


CURATED_SFT_EXAMPLES=[]
def validate_sft(examples):
    for i,x in enumerate(examples):
        if not all(str(x.get(k,"")).strip() for k in ["instruction","input","output"]):
            raise ValueError(f"Invalid example {i}")
    return True
# After human review:
# validate_sft(CURATED_SFT_EXAMPLES)
# Write JSONL, then use a separately versioned LoRA/QLoRA training script on a suitable cloud GPU.


# ## 18. Fine-tuning is a separate cloud-GPU experiment
# 
# Do not attempt full-parameter training of Qwen3.5-4B on the laptop's 6 GB RTX 4050. Local quantized Ollama inference is not the same as training or fine-tuning.
# 
# Before LoRA/QLoRA, verify that the current Transformers/PEFT/TRL versions support the exact Qwen3.5-4B architecture and its hybrid layers. A recipe for an older Qwen generation may not work. Use a suitable cloud GPU, and record quantization, sequence length, batch size, gradient accumulation, optimizer, package versions, model revision and adapter revision. Compare the adapted model with the untouched Qwen3.5-4B baseline on the same held-out incidents.
# 

# ## 19. Research evaluation and reporting
# 
# **Detection (only with verified labels):** ROC-AUC, PR-AUC, precision, recall, F1, confusion matrix, threshold selection on validation, false-positive review, event/entity/session-level metrics.
# 
# **LLM investigation:** evidence support, citation correctness, unsupported-claim rate, completeness rubric, abstention when evidence is insufficient, latency and tokens/sec.
# 
# **RAG:** retrieval hit rate on analyst-labelled queries, evidence recall@k, and whether retrieved records actually support the answer.
# 
# Do not use fluent answers, training loss, anomaly-score ranking, or response speed as substitutes for detection accuracy.

# In[ ]:


metadata={
 "run_id":RUN_ID,"created_at":datetime.now(timezone.utc).isoformat(),
 "dataset_repo":REPO_ID,"subset":DATA_SUBDIR,"model":OLLAMA_MODEL,
 "seed":SEED,"python":sys.version,"platform":platform.platform(),
 "field_map":FIELD_MAP
}
meta_path=RESULTS_DIR/f"{RUN_ID}_metadata.json"
meta_path.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
print("Saved:",meta_path)


# ## 20. Execution checklist
# 
# - [ ] Run cells 1–5 and inspect THEIA file inventory.
# - [ ] Confirm formats and sizes; inspect binary/graph files safely.
# - [ ] Run schema audit and identify the actual event table.
# - [ ] Verify timestamps, entities, event types, labels and Isolation Forest outputs.
# - [ ] Set field mapping explicitly.
# - [ ] Define incident/session/time-aware split.
# - [ ] Build compact evidence docs and RAG index.
# - [ ] Run multi-turn Qwen investigation and export sessions.
# - [ ] Establish detection metrics only if valid labels exist.
# - [ ] Prepare human-reviewed SFT examples only if fine-tuning is justified.
# - [ ] Record limitations and reproducibility metadata.
# 
# **Next step:** run through Section 7 first. The exact schema and file formats in your THEIA folder will determine the final feature engineering, graph aggregation, training objective, and evaluation. Do not start fine-tuning before that inspection.

# In[72]:


# ============================================================
# QWEN3.5-4B: SIMPLE LOCAL INFERENCE FUNCTION
# ============================================================

import requests

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:4b"

def ask_qwen(question, history=None):

    if history is None:
        history = []

    payload = {
        "model": LLM_MODEL,
        "think": False,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity research assistant. "
                    "Answer using the supplied THEIA evidence. "
                    "Do not invent facts. Cite evidence IDs. "
                    "Distinguish observations from hypotheses."
                )
            },
            *history[-8:],
            {
                "role": "user",
                "content": question
            }
        ],
        "options": {
            "temperature": 0.2,
            "num_ctx": 4096,
            "num_predict": 512
        }
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=600
    )

    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError(f"Empty Qwen response: {data}")

    return answer


print("QWEN FUNCTION READY")


# In[73]:


answer = ask_qwen(
    "Explain the difference between anomaly detection and APT detection."
)

print(answer)


# In[74]:


# ============================================================
# THEIA + QWEN3.5-4B — INTERACTIVE EVIDENCE-GROUNDED AI
# Standalone cell: does not modify previous notebook cells
# ============================================================

import requests
import json
import numpy as np
import pandas as pd
import ipywidgets as widgets

from IPython.display import display, Markdown, clear_output
from datetime import datetime

# -------------------- CONFIGURATION --------------------

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 5
MAX_EVIDENCE_DOCS = 1200
EMBED_BATCH_SIZE = 32

print("Checking Ollama models...")

# Verify both models
for model in [LLM_MODEL, EMBED_MODEL]:
    response = requests.post(
        f"{OLLAMA_URL}/api/show",
        json={"model": model},
        timeout=60
    )
    response.raise_for_status()

print("Both Ollama models are available.")

# -------------------- COLLECT EXISTING DATA --------------------

# Only inspect existing DataFrames; do not change them.
priority_words = [
    "test", "anomal", "score", "feature",
    "control", "investig", "event", "validation"
]

available_frames = []

for name, obj in list(globals().items()):
    if isinstance(obj, pd.DataFrame) and not obj.empty:
        available_frames.append((name, obj))

available_frames.sort(
    key=lambda item: (
        not any(word in item[0].lower() for word in priority_words),
        item[0]
    )
)

if not available_frames:
    raise RuntimeError(
        "No existing pandas DataFrames found. "
        "Run your THEIA investigation cells first."
    )

# Convert a controlled sample of existing rows into evidence documents.
evidence_documents = []
remaining = MAX_EVIDENCE_DOCS

for frame_name, frame in available_frames:
    if remaining <= 0:
        break

    # Prioritize smaller investigation tables and sample large tables.
    limit = min(len(frame), remaining, 250)

    if limit <= 0:
        continue

    sample = frame.head(limit)

    for row_index, row in sample.iterrows():
        record = {}

        for column, value in row.items():
            value_text = str(value)

            if len(value_text) > 350:
                value_text = value_text[:350] + "..."

            record[str(column)] = value_text

        evidence_id = f"{frame_name}-R{row_index}"

        evidence_documents.append({
            "id": evidence_id,
            "source": frame_name,
            "text": json.dumps(record, ensure_ascii=False)
        })

    remaining -= limit

if not evidence_documents:
    raise RuntimeError("No evidence documents could be constructed.")

print(f"Evidence documents prepared: {len(evidence_documents)}")
print("Source DataFrames:", [name for name, _ in available_frames])

# -------------------- CREATE EMBEDDINGS --------------------

def create_embeddings(texts):
    all_embeddings = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": batch
            },
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        all_embeddings.extend(result["embeddings"])

    vectors = np.asarray(all_embeddings, dtype=np.float32)

    # Normalize for cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-12)

    return vectors


print("Creating evidence embeddings. Please wait...")

document_texts = [
    f"Evidence ID: {doc['id']}\nSource: {doc['source']}\n{doc['text']}"
    for doc in evidence_documents
]

evidence_vectors = create_embeddings(document_texts)

print("Evidence embeddings created successfully.")

# -------------------- RETRIEVAL --------------------

def retrieve_theia_evidence(question, top_k=TOP_K):

    query_vector = create_embeddings([question])[0]

    similarities = evidence_vectors @ query_vector

    selected_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for index in selected_indices:
        doc = evidence_documents[int(index)]

        results.append({
            "id": doc["id"],
            "source": doc["source"],
            "text": doc["text"],
            "similarity": float(similarities[index])
        })

    return results


# -------------------- QWEN RESPONSE --------------------

theia_chat_history = []

def ask_theia_rag(question):

    if not question.strip():
        return

    evidence = retrieve_theia_evidence(question)

    evidence_context = "\n\n".join([
        (
            f"Evidence ID: {item['id']}\n"
            f"Source: {item['source']}\n"
            f"Similarity: {item['similarity']:.4f}\n"
            f"Data: {item['text']}"
        )
        for item in evidence
    ])

    system_prompt = """
You are THEIA AI Investigator, a cybersecurity research assistant.

Your task is to analyse the supplied THEIA dataset evidence.

STRICT RULES:
1. Base dataset-specific claims only on supplied evidence.
2. Cite evidence using its exact Evidence ID.
3. Never invent process names, IP addresses, file paths,
   threat actors, or missing dataset details.
4. Clearly distinguish:
   - Observations: directly supported by evidence.
   - Hypotheses: possible explanations requiring verification.
5. An anomaly is not automatically an APT.
6. Do not claim detection accuracy without ground-truth labels
   and a valid evaluation.
7. If evidence is insufficient, say so clearly.
8. Answer follow-up questions using conversation history,
   but retrieve and cite relevant evidence for each new question.
"""

    user_message = f"""
QUESTION:
{question}

RETRIEVED THEIA EVIDENCE:
{evidence_context}

Analyse the evidence and cite the exact Evidence IDs.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Preserve recent conversation for follow-up questions.
    messages.extend(theia_chat_history[-8:])

    messages.append({
        "role": "user",
        "content": user_message
    })

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "think": False,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 700
            }
        },
        timeout=600
    )

    response.raise_for_status()

    result = response.json()
    answer = result.get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Qwen returned an empty response.")

    # Store the conversation without the entire evidence payload.
    theia_chat_history.append({
        "role": "user",
        "content": question
    })

    theia_chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, evidence


# -------------------- INTERACTIVE CHAT UI --------------------

chat_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #888",
        padding="12px",
        height="430px",
        overflow_y="auto"
    )
)

question_box = widgets.Textarea(
    placeholder="Ask about THEIA anomalies, threads, events, features...",
    description="Question:",
    layout=widgets.Layout(width="100%", height="85px")
)

send_button = widgets.Button(
    description="Ask Qwen",
    button_style="primary",
    icon="paper-plane"
)

clear_button = widgets.Button(
    description="Clear Chat",
    button_style="warning",
    icon="trash"
)

status_label = widgets.HTML(
    value="<b>Ready.</b> Ask your first THEIA investigation question."
)


def send_question(_):

    question = question_box.value.strip()

    if not question:
        status_label.value = "<b>Please enter a question.</b>"
        return

    send_button.disabled = True
    status_label.value = "<b>Qwen is retrieving evidence and analysing...</b>"

    with chat_output:
        display(Markdown(f"### You\n{question}"))

        try:
            answer, evidence = ask_theia_rag(question)

            display(Markdown(f"### Qwen3.5-4B\n{answer}"))

            display(Markdown("**Retrieved evidence:**"))

            for item in evidence:
                display(Markdown(
                    f"- `{item['id']}` — "
                    f"{item['source']} "
                    f"(similarity: {item['similarity']:.3f})"
                ))

        except Exception as error:
            display(Markdown(f"**Error:** `{str(error)}`"))

    question_box.value = ""
    status_label.value = "<b>Ready for your next question.</b>"
    send_button.disabled = False


def clear_chat(_):

    theia_chat_history.clear()

    with chat_output:
        clear_output()

    status_label.value = (
        "<b>Chat history cleared. Evidence embeddings are retained.</b>"
    )


send_button.on_click(send_question)
clear_button.on_click(clear_chat)

display(Markdown("# THEIA AI Investigator"))
display(Markdown(
    "Powered by **Qwen3.5-4B** + **Nomic Embed Text**"
))
display(status_label)
display(chat_output)
display(question_box)
display(widgets.HBox([send_button, clear_button]))

print("THEIA interactive investigator initialized.")


# In[ ]:


# ================================================================
# THEIA AI INVESTIGATOR
# Qwen3.5-4B + Nomic Embed Text + Evidence-Grounded RAG
# Standalone interactive chat cell
# ================================================================

import requests
import json
import numpy as np
import pandas as pd
import ipywidgets as widgets

from IPython.display import display, Markdown, clear_output

# ------------------------------------------------
# 1. CONFIGURATION
# ------------------------------------------------

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 6
MAX_DOCUMENTS = 2000
BATCH_SIZE = 32

print("=" * 65)
print("INITIALIZING THEIA AI INVESTIGATOR")
print("=" * 65)

# ------------------------------------------------
# 2. VERIFY OLLAMA MODELS
# ------------------------------------------------

for model in [LLM_MODEL, EMBED_MODEL]:

    response = requests.post(
        f"{OLLAMA_URL}/api/show",
        json={"model": model},
        timeout=60
    )

    response.raise_for_status()
    print(f"Verified: {model}")

# ------------------------------------------------
# 3. COLLECT EXISTING THEIA DATAFRAMES
# ------------------------------------------------

# This cell reads existing DataFrames only.
# It does not modify your original DataFrames.

priority_terms = [
    "anomal", "test", "score", "feature",
    "event", "control", "investig",
    "validation", "thread", "output"
]

frames = []

for variable_name, variable in list(globals().items()):

    if (
        isinstance(variable, pd.DataFrame)
        and not variable.empty
        and not variable_name.startswith("_")
    ):

        frames.append((variable_name, variable))

if not frames:

    raise RuntimeError(
        "No DataFrames found. Run your THEIA feature "
        "engineering and investigation cells first."
    )

# Prioritize investigation-related DataFrames.
frames.sort(
    key=lambda item: (
        not any(
            term in item[0].lower()
            for term in priority_terms
        ),
        item[0]
    )
)

# ------------------------------------------------
# 4. BUILD EVIDENCE DOCUMENTS
# ------------------------------------------------

evidence_documents = []

for frame_name, frame in frames:

    if len(evidence_documents) >= MAX_DOCUMENTS:
        break

    available = MAX_DOCUMENTS - len(evidence_documents)

    # Keep more rows from smaller investigation tables.
    limit = min(len(frame), available, 300)

    if limit <= 0:
        continue

    # Prefer highest-scoring records where a score column exists.
    score_columns = [
        col for col in frame.columns
        if any(
            word in str(col).lower()
            for word in ["anomaly_score", "score", "decision_function"]
        )
    ]

    if score_columns:

        try:
            numeric_scores = pd.to_numeric(
                frame[score_columns[0]],
                errors="coerce"
            )

            # More anomalous Isolation Forest scores are generally
            # lower, so use ascending order for those columns.
            if "decision_function" in str(score_columns[0]).lower():
                selected = frame.loc[
                    numeric_scores.sort_values(
                        ascending=True
                    ).head(limit).index
                ]
            else:
                selected = frame.loc[
                    numeric_scores.abs().sort_values(
                        ascending=False
                    ).head(limit).index
                ]

        except Exception:
            selected = frame.head(limit)

    else:
        selected = frame.head(limit)

    for row_index, row in selected.iterrows():

        row_data = {}

        for column, value in row.items():

            if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
                value = "Not available"

            if isinstance(value, (np.integer,)):
                value = int(value)

            elif isinstance(value, (np.floating,)):
                value = float(value)

            elif isinstance(value, np.ndarray):
                value = value.tolist()

            row_data[str(column)] = str(value)[:500]

        evidence_id = f"{frame_name}-ROW-{row_index}"

        evidence_documents.append({
            "evidence_id": evidence_id,
            "source": frame_name,
            "row_index": str(row_index),
            "content": json.dumps(
                row_data,
                ensure_ascii=False
            )
        })

if not evidence_documents:

    raise RuntimeError("Could not construct evidence documents.")

print(f"\nEvidence documents created: {len(evidence_documents)}")
print("DataFrame sources:")

for frame_name, frame in frames:
    print(f"  - {frame_name}: {len(frame)} rows")

# ------------------------------------------------
# 5. GENERATE EMBEDDINGS USING NOM​​IC
# ------------------------------------------------

def theia_create_embeddings(texts):

    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": batch
            },
            timeout=300
        )

        response.raise_for_status()

        result = response.json()

        vectors.extend(result["embeddings"])

    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True
    )

    vectors = vectors / np.maximum(norms, 1e-12)

    return vectors


print("\nGenerating THEIA evidence embeddings...")

evidence_texts = [
    (
        f"Evidence ID: {doc['evidence_id']}\n"
        f"Source: {doc['source']}\n"
        f"Data: {doc['content']}"
    )
    for doc in evidence_documents
]

evidence_embeddings = theia_create_embeddings(
    evidence_texts
)

print("Evidence embeddings generated successfully.")

# ------------------------------------------------
# 6. RETRIEVE RELEVANT EVIDENCE
# ------------------------------------------------

def theia_retrieve(question, top_k=TOP_K):

    query_embedding = theia_create_embeddings(
        [question]
    )[0]

    similarity_scores = (
        evidence_embeddings @ query_embedding
    )

    selected_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    retrieved = []

    for index in selected_indices:

        document = evidence_documents[int(index)]

        retrieved.append({
            "evidence_id": document["evidence_id"],
            "source": document["source"],
            "row_index": document["row_index"],
            "content": document["content"],
            "similarity": float(
                similarity_scores[index]
            )
        })

    return retrieved

# ------------------------------------------------
# 7. QWEN EVIDENCE-BASED ANSWER GENERATION
# ------------------------------------------------

theia_history = []

def theia_generate_answer(question):

    # Use recent conversation to resolve follow-up references.
    retrieval_question = question

    if theia_history:

        recent_questions = [
            item["content"]
            for item in theia_history[-4:]
            if item["role"] == "user"
        ]

        retrieval_question = (
            "Previous investigation context: "
            + " ".join(recent_questions)
            + "\nCurrent question: "
            + question
        )

    retrieved = theia_retrieve(retrieval_question)

    evidence_context = "\n\n".join([
        (
            f"EVIDENCE ID: {item['evidence_id']}\n"
            f"SOURCE: {item['source']}\n"
            f"ROW: {item['row_index']}\n"
            f"SIMILARITY: {item['similarity']:.4f}\n"
            f"DATA: {item['content']}"
        )
        for item in retrieved
    ])

    system_prompt = """
You are THEIA AI Investigator, an expert cybersecurity
research assistant specialising in APT investigation.

Your job is to understand the user's question and provide
a clear, detailed, evidence-grounded answer.

ANSWER FORMAT:

1. QUESTION UNDERSTANDING
Explain briefly what the user is asking.

2. ANALYSIS
Explain the relevant behaviour, feature values, event
patterns, thread activity, or anomaly observations.

3. FINDINGS
Present the findings in clear and understandable language.

4. CONCLUSION
Give a direct answer to the user's question.

5. EVIDENCE REFERENCES
Cite the exact Evidence IDs supplied in the retrieved context.

STRICT RESEARCH RULES:

- Use supplied THEIA evidence for dataset-specific claims.
- Cite actual Evidence IDs exactly as provided.
- Never invent evidence IDs, event values, or feature values.
- Separate direct observations from hypotheses.
- An anomaly does not automatically mean an APT attack.
- Do not claim detection accuracy without ground-truth labels.
- If evidence is insufficient, explicitly explain what is missing.
- Do not invent process names, IP addresses, file paths,
  threat actors, or other unavailable dataset fields.
- Explain technical findings in simple but professional language.
"""

    user_prompt = f"""
USER QUESTION:

{question}

RETRIEVED THEIA DATASET EVIDENCE:

{evidence_context}

Please answer the question using the required answer format.
Cite the exact Evidence IDs supporting your findings.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    # Preserve recent conversation for follow-up questions.
    messages.extend(theia_history[-8:])

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "think": False,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.2,
                "num_ctx": 4096,
                "num_predict": 900
            }
        },
        timeout=600
    )

    response.raise_for_status()

    result = response.json()

    answer = (
        result.get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:

        raise RuntimeError(
            "Qwen generated an empty answer."
        )

    # Store conversational history.
    theia_history.append({
        "role": "user",
        "content": question
    })

    theia_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, retrieved

# ------------------------------------------------
# 8. INTERACTIVE CHAT INTERFACE
# ------------------------------------------------

chat_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #888",
        padding="15px",
        height="500px",
        overflow_y="auto"
    )
)

question_input = widgets.Textarea(
    placeholder=(
        "Ask your THEIA investigation question here...\n"
        "Example: Why was this thread classified as anomalous?"
    ),
    description="Question:",
    layout=widgets.Layout(
        width="100%",
        height="90px"
    )
)

ask_button = widgets.Button(
    description="Ask THEIA AI",
    button_style="primary",
    icon="paper-plane"
)

clear_button = widgets.Button(
    description="New Conversation",
    button_style="warning",
    icon="refresh"
)

status = widgets.HTML(
    value="<b>THEIA AI Investigator is ready.</b>"
)

# ------------------------------------------------
# 9. DISPLAY ANSWER AND SUPPORTING EVIDENCE
# ------------------------------------------------

def theia_handle_question(_):

    question = question_input.value.strip()

    if not question:
        status.value = "<b>Please enter a question.</b>"
        return

    ask_button.disabled = True

    status.value = (
        "<b>Qwen is analysing your question and retrieving evidence...</b>"
    )

    with chat_output:

        display(Markdown(
            f"## Your Question\n{question}"
        ))

        try:

            answer, retrieved = theia_generate_answer(
                question
            )

            # First display the generated explanation.
            display(Markdown(
                "## Qwen3.5-4B — Explanation and Answer"
            ))

            display(Markdown(answer))

            # Then display actual retrieved records.
            display(Markdown(
                "---\n## Supporting THEIA Evidence"
            ))

            for item in retrieved:

                display(Markdown(
                    f"""
### Evidence: `{item['evidence_id']}`

- **Source DataFrame:** `{item['source']}`
- **Row index:** `{item['row_index']}`
- **Semantic similarity:** `{item['similarity']:.4f}`

**Actual retrieved record:**

```json
{json.dumps(json.loads(item['content']), indent=2, ensure_ascii=False)}
```
"""
                ))

            display(Markdown("---"))

        except Exception as error:

            display(Markdown(
                f"### Investigation Error\n`{str(error)}`"
            ))

    question_input.value = ""

    status.value = (
        "<b>Analysis complete. Ask a follow-up question.</b>"
    )

    ask_button.disabled = False


# ------------------------------------------------
# 10. CLEAR CONVERSATION
# ------------------------------------------------

def theia_clear_conversation(_):

    theia_history.clear()

    with chat_output:
        clear_output()

    status.value = (
        "<b>New conversation started. Evidence index retained.</b>"
    )


ask_button.on_click(theia_handle_question)

clear_button.on_click(
    theia_clear_conversation
)

# ------------------------------------------------
# 11. RENDER THE INTERFACE
# ------------------------------------------------

display(Markdown(
    "# THEIA AI INVESTIGATOR\n"
    "### Qwen3.5-4B | Nomic Embeddings | Evidence-Grounded RAG"
))

display(Markdown(
    "Ask questions about anomalies, threads, event patterns, "
    "features and investigation findings."
))

display(status)
display(chat_output)
display(question_input)

display(
    widgets.HBox([
        ask_button,
        clear_button
    ])
)

print("\n" + "=" * 65)
print("THEIA AI INVESTIGATOR INITIALIZED")
print("Evidence retrieval: READY")
print("Qwen inference: READY")
print("Follow-up conversation: READY")
print("=" * 65)


# In[ ]:


# ================================================================
# THEIA AI INVESTIGATOR — IMPROVED VERSION
# Hybrid Retrieval + Qwen3.5-4B + Evidence Display
# Standalone cell. Does not modify existing model/dataframes.
# ================================================================

import requests
import json
import re
import numpy as np
import pandas as pd
import ipywidgets as widgets

from IPython.display import display, Markdown, clear_output

# -------------------- 1. CONFIGURATION --------------------

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen3.5:4b"
EMBED_MODEL = "nomic-embed-text"

TOP_K = 8
MAX_DOCUMENTS = 1800
BATCH_SIZE = 32
MAX_HISTORY = 8

print("Initializing improved THEIA Investigator...")

# Verify models
for model in [LLM_MODEL, EMBED_MODEL]:
    r = requests.post(
        f"{OLLAMA_URL}/api/show",
        json={"model": model},
        timeout=60
    )
    r.raise_for_status()
    print("Available:", model)

# -------------------- 2. FIND RELEVANT DATAFRAMES --------------------

# Only read existing DataFrames. Never change the original objects.

preferred_names = [
    "anomalous_events",
    "test_output",
    "expanded_control_events",
    "group_mapping",
    "test_df",
    "validation_df",
    "test_features",
    "validation_features",
    "train_features",
    "test_thread_5m_features",
    "validation_thread_5m_features",
    "train_thread_5m_features"
]

all_frames = {
    name: obj
    for name, obj in list(globals().items())
    if isinstance(obj, pd.DataFrame) and not obj.empty
}

def frame_priority(name, frame):
    n = name.lower()
    score = 0

    if any(x in n for x in ["anomal", "flagged"]):
        score += 15
    if any(x in n for x in ["test_output", "test_features"]):
        score += 12
    if any(x in n for x in ["investig", "group_mapping"]):
        score += 10
    if any(x in n for x in ["control", "normal"]):
        score += 8
    if "validation" in n:
        score += 6
    if "train" in n:
        score += 2

    columns = " ".join(map(str, frame.columns)).lower()

    if "anomaly_score" in columns or "decision_function" in columns:
        score += 8
    if "event_type" in columns or "operation" in columns:
        score += 5
    if "thread_id" in columns:
        score += 4

    return score

ranked_frames = sorted(
    all_frames.items(),
    key=lambda item: frame_priority(item[0], item[1]),
    reverse=True
)

# Avoid unrelated DataFrames.
relevant_frames = []

for name, frame in ranked_frames:
    name_match = any(
        word in name.lower()
        for word in [
            "anomal", "test", "validation", "control",
            "investig", "feature", "event", "thread",
            "mapping", "output", "score"
        ]
    )

    columns = " ".join(map(str, frame.columns)).lower()

    column_match = any(
        word in columns
        for word in [
            "thread_id", "event_type", "anomaly_score",
            "decision_function", "event_count", "operation"
        ]
    )

    if name_match or column_match:
        relevant_frames.append((name, frame))

if not relevant_frames:
    raise RuntimeError(
        "No relevant THEIA DataFrames found. "
        "Run your feature engineering and investigation cells first."
    )

# -------------------- 3. BUILD BALANCED EVIDENCE INDEX --------------------

evidence_documents = []
per_frame_limit = max(40, MAX_DOCUMENTS // len(relevant_frames))

def safe_value(value):
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return "Not available"
    except Exception:
        pass

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    return str(value)[:400]

for frame_name, frame in relevant_frames:

    remaining = MAX_DOCUMENTS - len(evidence_documents)
    if remaining <= 0:
        break

    limit = min(
        len(frame),
        per_frame_limit,
        remaining
    )

    # If a frame has anomaly scores, prioritize suspicious records.
    score_col = next(
        (
            c for c in frame.columns
            if str(c).lower() in [
                "anomaly_score",
                "decision_function",
                "score"
            ]
        ),
        None
    )

    if score_col is not None:
        numeric = pd.to_numeric(
            frame[score_col],
            errors="coerce"
        )

        if "decision_function" in str(score_col).lower():
            selected_indices = numeric.nsmallest(limit).index
        else:
            selected_indices = numeric.abs().nlargest(limit).index

        selected = frame.loc[selected_indices]

    else:
        # Spread records through the frame rather than only taking
        # the first rows, which can bias the retrieval index.
        positions = np.linspace(
            0,
            len(frame) - 1,
            num=limit,
            dtype=int
        )
        selected = frame.iloc[positions]

    for row_index, row in selected.iterrows():

        record = {
            str(col): safe_value(value)
            for col, value in row.items()
        }

        evidence_documents.append({
            "id": f"{frame_name}-ROW-{row_index}",
            "source": frame_name,
            "row": str(row_index),
            "data": record
        })

if not evidence_documents:
    raise RuntimeError("No evidence records were created.")

print("\nEvidence records:", len(evidence_documents))
print("Indexed DataFrames:")

for name, frame in relevant_frames:
    print(f"  {name}: {len(frame):,} original rows")

# -------------------- 4. NOM​​IC EMBEDDINGS --------------------

def improved_embeddings(texts, mode="document"):

    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        if mode == "document":
            batch = ["search_document: " + x for x in batch]
        else:
            batch = ["search_query: " + x for x in batch]

        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBED_MODEL,
                "input": batch
            },
            timeout=300
        )

        response.raise_for_status()
        vectors.extend(response.json()["embeddings"])

    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


document_texts = []

for doc in evidence_documents:

    document_texts.append(
        f"""
THEIA evidence record.
Evidence ID: {doc['id']}
Source: {doc['source']}
Row: {doc['row']}
Fields and values:
{json.dumps(doc['data'], ensure_ascii=False)}
"""
    )

print("\nBuilding evidence search index...")

document_embeddings = improved_embeddings(
    document_texts,
    mode="document"
)

print("Evidence search index ready.")

# -------------------- 5. HYBRID RETRIEVAL --------------------

STOPWORDS = {
    "what", "which", "where", "when", "why", "how",
    "the", "and", "for", "with", "from", "that",
    "this", "are", "was", "were", "about", "does",
    "explain", "tell", "give", "show", "please",
    "can", "you", "its", "their", "into", "have"
}

def tokenize(text):
    return {
        word for word in re.findall(
            r"[a-zA-Z0-9_]+",
            str(text).lower()
        )
        if len(word) > 2 and word not in STOPWORDS
    }


def retrieve_improved(question, top_k=TOP_K):

    query_vector = improved_embeddings(
        [question],
        mode="query"
    )[0]

    semantic_scores = document_embeddings @ query_vector

    query_terms = tokenize(question)

    lexical_scores = []

    for doc in evidence_documents:

        searchable_text = (
            doc["source"] + " " +
            json.dumps(doc["data"], ensure_ascii=False)
        )

        record_terms = tokenize(searchable_text)

        if query_terms:
            overlap = len(query_terms & record_terms)
            lexical = overlap / len(query_terms)
        else:
            lexical = 0.0

        lexical_scores.append(lexical)

    lexical_scores = np.asarray(lexical_scores)

    # Hybrid ranking: semantic meaning + exact word overlap.
    combined_scores = (
        0.75 * semantic_scores +
        0.25 * lexical_scores
    )

    selected = np.argsort(combined_scores)[::-1][:top_k]

    results = []

    for index in selected:

        doc = evidence_documents[int(index)]

        results.append({
            "id": doc["id"],
            "source": doc["source"],
            "row": doc["row"],
            "data": doc["data"],
            "semantic_score": float(semantic_scores[index]),
            "hybrid_score": float(combined_scores[index])
        })

    return results


# -------------------- 6. QUESTION CLASSIFICATION --------------------

GENERAL_TERMS = [
    "what is apt",
    "define apt",
    "what is an apt",
    "what is advanced persistent threat",
    "what is anomaly detection",
    "explain anomaly detection",
    "what is theia",
    "what is the theia dataset",
    "what does theia mean"
]

def is_general_question(question):

    q = question.lower()

    return any(term in q for term in GENERAL_TERMS)


# -------------------- 7. QWEN RESEARCH ANSWER --------------------

chat_history = []

SYSTEM_PROMPT = """
You are THEIA AI Investigator, a cybersecurity research assistant.

Your purpose is to explain cybersecurity questions and investigate
THEIA system-event evidence.

IMPORTANT ANSWER RULES:

1. Understand the exact question before answering.
2. Give a direct answer first.
3. Explain technical terms in clear professional language.
4. Separate general cybersecurity knowledge from THEIA observations.
5. Cite only Evidence IDs supplied in the current evidence context.
6. Never invent evidence IDs, dataset fields, values, or events.
7. An anomaly is not automatically an APT attack.
8. Do not claim accuracy, precision, recall, or attack detection
   without appropriate ground-truth labels.
9. If the supplied evidence cannot answer the question, say so.
10. Do not invent process names, IP addresses, file paths,
    malware names, or threat actor identities.

For general questions:
Explain the concept using general cybersecurity knowledge.
Do not pretend that low-level event records define the concept.

For THEIA-specific questions:
Explain what the retrieved records actually show.
Use exact field names and values.

For combined questions:
Clearly separate general background and dataset-specific findings.

Keep the answer focused. Do not repeat the question unnecessarily.
"""

def generate_improved_answer(question):

    # Retrieve evidence for every question.
    retrieved = retrieve_improved(question)

    evidence_context = "\n\n".join(
        [
            (
                f"Evidence ID: {item['id']}\n"
                f"Source DataFrame: {item['source']}\n"
                f"Row: {item['row']}\n"
                f"Data: {json.dumps(item['data'], ensure_ascii=False)}"
            )
            for item in retrieved
        ]
    )

    if is_general_question(question):
        evidence_note = """
The question includes a general cybersecurity concept.
Explain that concept using general cybersecurity knowledge.
Do not claim the retrieved event rows define THEIA or APT.
Use the retrieved records only for any dataset-specific observations.
"""
    else:
        evidence_note = """
This is a dataset investigation question.
Base dataset-specific findings on the supplied records.
If the records are insufficient, state what is missing.
"""

    user_prompt = f"""
USER QUESTION:
{question}

INSTRUCTIONS:
{evidence_note}

RETRIEVED THEIA EVIDENCE:
{evidence_context}

Write a clear, useful answer.
For dataset-specific claims, cite the exact Evidence IDs.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    messages.extend(chat_history[-MAX_HISTORY:])

    messages.append({
        "role": "user",
        "content": user_prompt
    })

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": LLM_MODEL,
            "think": False,
            "stream": False,
            "messages": messages,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192,
                "num_predict": 1000
            }
        },
        timeout=600
    )

    response.raise_for_status()

    answer = (
        response.json()
        .get("message", {})
        .get("content", "")
        .strip()
    )

    if not answer:
        raise RuntimeError("Qwen returned an empty answer.")

    # Keep the conversation context concise.
    chat_history.append({
        "role": "user",
        "content": question
    })

    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer, retrieved


# -------------------- 8. CLEAN CHAT INTERFACE --------------------

# No fixed-height bordered output box.
# Answers and evidence are displayed naturally below each question.

answer_output = widgets.Output(
    layout=widgets.Layout(width="100%")
)

evidence_output = widgets.Output(
    layout=widgets.Layout(width="100%")
)

question_input = widgets.Textarea(
    placeholder=(
        "Ask a question about THEIA, APTs, anomaly detection, "
        "threads, event types or model findings..."
    ),
    layout=widgets.Layout(
        width="100%",
        height="75px"
    )
)

ask_button = widgets.Button(
    description="Ask THEIA AI",
    button_style="primary",
    icon="paper-plane"
)

new_button = widgets.Button(
    description="New Conversation",
    icon="refresh"
)

status = widgets.HTML(
    value="<b>Ready</b> — Ask your first question."
)


# -------------------- 9. DISPLAY ANSWER AND EVIDENCE --------------------

def handle_question(_):

    question = question_input.value.strip()

    if not question:
        status.value = "<b>Enter a question first.</b>"
        return

    ask_button.disabled = True
    status.value = "<b>Retrieving evidence and generating answer...</b>"

    with answer_output:
        display(Markdown(f"### You\n{question}"))

        try:

            answer, retrieved = generate_improved_answer(question)

            display(Markdown("### THEIA AI — Answer"))
            display(Markdown(answer))

        except Exception as error:

            display(Markdown(
                f"**Error:** `{str(error)}`"
            ))

            retrieved = []

    if retrieved:

        with evidence_output:

            display(Markdown("### Supporting Evidence"))

            summary_rows = []

            for item in retrieved:

                summary_rows.append({
                    "Evidence ID": item["id"],
                    "Source": item["source"],
                    "Row": item["row"],
                    "Semantic score": round(
                        item["semantic_score"], 4
                    ),
                    "Hybrid score": round(
                        item["hybrid_score"], 4
                    )
                })

            display(pd.DataFrame(summary_rows))

            # Display the actual retrieved data separately.
            for number, item in enumerate(retrieved, 1):

                display(Markdown(
                    f"**Evidence {number}: `{item['id']}`**"
                ))

                display(pd.DataFrame(
                    list(item["data"].items()),
                    columns=["Field", "Actual value"]
                ))

    question_input.value = ""

    status.value = (
        "<b>Completed.</b> Ask a follow-up question."
    )

    ask_button.disabled = False


# -------------------- 10. NEW CONVERSATION --------------------

def start_new_conversation(_):

    chat_history.clear()

    with answer_output:
        clear_output()

    with evidence_output:
        clear_output()

    status.value = (
        "<b>New conversation started.</b> Evidence index retained."
    )


ask_button.on_click(handle_question)
new_button.on_click(start_new_conversation)


# -------------------- 11. RENDER --------------------

display(Markdown(
    "# THEIA AI INVESTIGATOR\n"
    "Qwen3.5-4B · Nomic Embeddings · Hybrid Evidence Retrieval"
))

display(Markdown(
    "Ask questions, receive explanations, and inspect the "
    "supporting THEIA records."
))

display(status)
display(question_input)
display(widgets.HBox([ask_button, new_button]))

display(answer_output)
display(evidence_output)

print("\nTHEIA IMPROVED INVESTIGATOR READY")
print("Hybrid retrieval: READY")
print("Evidence display: READY")
print("Qwen inference: READY")
print("Follow-up conversation: READY")


# In[ ]:




