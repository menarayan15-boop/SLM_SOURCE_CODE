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

# In[ ]:


get_ipython().run_line_magic('pip', 'install -U huggingface_hub datasets pandas pyarrow numpy scikit-learn matplotlib sentence-transformers faiss-cpu ollama ipywidgets tqdm psutil')


# ## 2. Imports and configuration

# In[ ]:


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

# In[ ]:


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

# In[ ]:


download_root = Path(snapshot_download(
    repo_id=REPO_ID, repo_type=REPO_TYPE,
    allow_patterns=[f"{DATA_SUBDIR}/**"], local_dir=str(RAW_DIR)
))
THEIA_DIR = download_root / DATA_SUBDIR
print("THEIA folder:", THEIA_DIR)


# ## 5. Local manifest and hashes
# Pickle files are not opened automatically: unpickling untrusted files can execute arbitrary code.

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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

# In[ ]:


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
# RAG grounds the assistant in retrieved records without updating Qwen weights. Similarity is not an anomaly score.

# In[ ]:


from sentence_transformers import SentenceTransformer
import faiss

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)
vector_index = None

# Avoid unexpectedly embedding a huge dataset on a 16 GB RAM laptop.
# Raise this limit only after checking available RAM and index build time.
MAX_INDEX_DOCS = 30000
index_documents = documents[:MAX_INDEX_DOCS]

if documents:
    if len(documents) > MAX_INDEX_DOCS:
        print(f"Indexing first {MAX_INDEX_DOCS:,} of {len(documents):,} evidence documents.")
        print("For broader coverage, aggregate records by entity/time window or increase MAX_INDEX_DOCS cautiously.")
    texts = [d["text"] for d in index_documents]
    vec = embedder.encode(
        texts, batch_size=16, show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")
    vector_index = faiss.IndexFlatIP(vec.shape[1])
    vector_index.add(vec)
    faiss.write_index(vector_index, str(INDEX_DIR / "theia.faiss"))
    with open(INDEX_DIR / "documents.jsonl", "w", encoding="utf-8") as f:
        for d in index_documents:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def retrieve_evidence(question, k=8):
    if vector_index is None:
        raise RuntimeError("Build the index first.")
    q = embedder.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = vector_index.search(q, min(k, len(index_documents)))
    return [
        {"similarity": float(s), **index_documents[int(i)]}
        for s, i in zip(scores[0], ids[0]) if i >= 0
    ]


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

# In[ ]:


import ollama

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen3.5:4b"  # Local 4B model selected for RTX 4050 6GB VRAM
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Conservative starting settings for 6GB VRAM / 16GB RAM.
# Increase num_ctx only after checking memory use and response stability.
LLM_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "num_ctx": 4096,
    "num_predict": 600,
}

try:
    print("Ollama version:", ollama_client.version())
    installed_models = [
        getattr(m, "model", None) or getattr(m, "name", None)
        for m in ollama_client.list().models
    ]
    print("Installed models:", installed_models)
    if not any(name == OLLAMA_MODEL or name.startswith(OLLAMA_MODEL + ":") for name in installed_models):
        print(f"\nModel not found. In a VS Code terminal run:\n  ollama pull {OLLAMA_MODEL}")
        print("Then rerun this cell.")
    else:
        print(f"Ready to use {OLLAMA_MODEL}")
except Exception as e:
    print("Ollama is unavailable. Start/install Ollama, then run:")
    print(f"  ollama pull {OLLAMA_MODEL}")
    print("Connection details:", repr(e))


# ## 15. Evidence-grounded Qwen investigation

# In[ ]:


SYSTEM_PROMPT = """You are Qwen3.5-4B, an assistant for defensive APT research on THEIA telemetry.
Use only the supplied evidence for claims about this dataset. Separate direct observations,
hypotheses, and missing evidence. Never invent entities, timestamps, operations, attack stages,
or ground truth. Do not call an event malicious merely because it is unusual. Cite exact
evidence identifiers such as [THEIA/file::row_123]. If evidence is insufficient, say so.
This is research assistance, not an autonomous response or containment system."""
def ask_theia(question,history=None,k=8):
    hits=retrieve_evidence(question,k)
    context="\n\n".join(f"[{h['doc_id']}] similarity={h['similarity']:.3f}\n{h['text']}" for h in hits)
    messages=[{"role":"system","content":SYSTEM_PROMPT}]
    if history: messages.extend(history)
    messages.append({"role":"user","content":f"Question: {question}\n\nRetrieved THEIA evidence:\n{context}\n\nCite evidence IDs."})
    r=ollama_client.chat(model=OLLAMA_MODEL,messages=messages,options=LLM_OPTIONS)
    return r.message.content,hits


# ## 16. Interactive multi-turn investigation cell
# Ask multiple questions and follow-ups. Conversation persists until reset. Exports include evidence IDs for auditability.

# In[ ]:


import ipywidgets as w
from IPython.display import display,Markdown,clear_output
history=[]; session=[]
qbox=w.Textarea(placeholder="Ask about entities, event sequences, unusual behavior, or evidence gaps...",description="Question:",layout=w.Layout(width="100%",height="80px"))
ask=w.Button(description="Investigate",button_style="primary")
reset=w.Button(description="Reset session",button_style="warning")
save=w.Button(description="Export session",button_style="success")
status=w.HTML("Ready after data index has been built.")
out=w.Output(layout={"border":"1px solid #ccc","padding":"10px","height":"420px","overflow_y":"auto"})
def ask_click(_):
    global history
    q=qbox.value.strip()
    if not q: status.value="Enter a question."; return
    ask.disabled=True; status.value="Retrieving evidence..."
    try:
        answer,hits=ask_theia(q,history)
        history.extend([{"role":"user","content":q},{"role":"assistant","content":answer}])
        session.append({"time":datetime.now(timezone.utc).isoformat(),"question":q,"answer":answer,
                        "evidence":[{"doc_id":h["doc_id"],"similarity":h["similarity"]} for h in hits]})
        with out:
            display(Markdown("**YOU**\n\n"+q))
            display(Markdown("**QWEN3.5-4B**\n\n"+answer))
            display(pd.DataFrame([{"doc_id":h["doc_id"],"similarity":round(h["similarity"],3),"preview":h["text"][:220]} for h in hits]))
        qbox.value=""; status.value="Completed."
    except Exception as e: status.value=f"Error: {type(e).__name__}: {e}"
    finally: ask.disabled=False
def reset_click(_):
    global history,session
    history=[];session=[]
    with out: clear_output()
    status.value="New investigation."
def save_click(_):
    p=RESULTS_DIR/f"{RUN_ID}_chat.json"
    p.write_text(json.dumps(session,indent=2,ensure_ascii=False),encoding="utf-8")
    status.value=f"Saved: {p}"
ask.on_click(ask_click);reset.on_click(reset_click);save.on_click(save_click)
display(w.VBox([w.HBox([ask,reset,save]),qbox,status,out]))


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
