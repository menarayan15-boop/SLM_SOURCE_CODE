#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive

try:
    drive.mount('/content/drive', force_remount=True)
    print("SUCCESS: Google Drive connected!")
except Exception as e:
    print("Drive connection failed:")
    print(e)


# In[ ]:


import os

THEIA_PATH = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"

print("Checking THEIA dataset location...")

if os.path.exists(THEIA_PATH):
    print("THEIA location found!")

    for root, dirs, files in os.walk(THEIA_PATH):
        print("\nFolder:", root)

        for file in files[:20]:
            path = os.path.join(root, file)
            print(
                file,
                f"({os.path.getsize(path) / (1024**3):.3f} GB)"
            )
else:
    print("THEIA path not accessible.")
    print("We will resolve the shortcut target.")


# In[ ]:


import os
import csv
from collections import Counter

# Original THEIA source
THEIA_PATH = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"

# New output directory
OUTPUT_PATH = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

os.makedirs(OUTPUT_PATH, exist_ok=True)

inventory = []
extension_counts = Counter()
total_size = 0

print("SCANNING COMPLETE THEIA DATASET")
print("=" * 65)

for root, dirs, files in os.walk(THEIA_PATH):

    for filename in files:

        filepath = os.path.join(root, filename)

        try:
            size = os.path.getsize(filepath)
        except OSError as e:
            print("Cannot access:", filepath, e)
            continue

        total_size += size

        lower_name = filename.lower()

        if lower_name.endswith(".json.gz"):
            file_type = "GZIP_COMPRESSED_JSON"

        elif lower_name.endswith(".bin.gz"):
            file_type = "GZIP_COMPRESSED_BINARY"

        elif lower_name.endswith(".gz"):
            file_type = "OTHER_GZIP"

        elif lower_name.endswith(".md5sum"):
            file_type = "CHECKSUM_MANIFEST"

        else:
            file_type = "OTHER"

        extension_counts[file_type] += 1

        inventory.append({
            "file_name": filename,
            "file_path": filepath,
            "size_bytes": size,
            "size_MB": round(size / (1024**2), 3),
            "file_type": file_type
        })

# Summary
print("\nDATASET INVENTORY SUMMARY")
print("-" * 65)

print("Total files:", len(inventory))
print("Total size (GB):", round(total_size / (1024**3), 3))

print("\nFILE TYPE COUNTS")

for file_type, count in extension_counts.items():
    print(f"{file_type}: {count}")

# Save inventory
inventory_csv = os.path.join(
    OUTPUT_PATH,
    "THEIA_FILE_INVENTORY.csv"
)

with open(inventory_csv, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name",
            "file_path",
            "size_bytes",
            "size_MB",
            "file_type"
        ]
    )

    writer.writeheader()
    writer.writerows(inventory)

print("\nInventory saved:", inventory_csv)
print("\nOutput folder created:", OUTPUT_PATH)



# In[ ]:


import os
import gzip
import json
import glob

print("THEIA FORMAT INVESTIGATION")
print("=" * 65)

patterns = [
    "*.json.gz",
    "*.bin.gz"
]

for pattern in patterns:

    files = sorted(
        glob.glob(os.path.join(THEIA_PATH, pattern))
    )

    if not files:
        print(f"\nNo files found for {pattern}")
        continue

    sample_file = files[0]

    print(f"\nFILE TYPE: {pattern}")
    print("Sample:", os.path.basename(sample_file))

    try:
        with gzip.open(sample_file, "rb") as f:
            sample = f.read(4096)

        print("Decompressed sample size:", len(sample))
        print("First 100 bytes:", repr(sample[:100]))

        if pattern == "*.json.gz":

            try:
                text_sample = sample.decode("utf-8-sig")
                print("\nJSON SAMPLE:")
                print(text_sample[:1000])

                try:
                    parsed = json.loads(text_sample)
                    print("\nParsed JSON type:", type(parsed).__name__)
                except json.JSONDecodeError:
                    print(
                        "\nSample is not a complete JSON document. "
                        "It may contain JSON Lines or a partial record."
                    )

            except UnicodeDecodeError:
                print("Sample is not valid UTF-8 text.")

        else:
            print("\nBinary format signature:")
            print(sample[:16].hex(" "))

    except Exception as e:
        print("READING ERROR:", repr(e))

print("\nFORMAT INVESTIGATION FINISHED")


# In[ ]:


import os
import gzip
import json
import csv
import glob
import subprocess
import sys
from collections import Counter

# Install Avro reader only if needed
try:
    import fastavro
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "fastavro"
    ])
    import fastavro

THEIA_PATH = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
OUTPUT_PATH = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

# --------------------------------------------------
# 1. CORRECT FILE CLASSIFICATION
# --------------------------------------------------

inventory = []
counts = Counter()

for root, dirs, files in os.walk(THEIA_PATH):
    for filename in files:

        path = os.path.join(root, filename)
        lower = filename.lower()

        if lower.endswith(".md5sum"):
            category = "CHECKSUM_MANIFEST"
        elif lower.endswith(".gz") and ".json" in lower:
            category = "GZIP_JSON"
        elif lower.endswith(".gz") and ".bin" in lower:
            category = "GZIP_AVRO_BINARY"
        elif lower.endswith(".gz"):
            category = "OTHER_GZIP"
        else:
            category = "OTHER"

        size = os.path.getsize(path)
        counts[category] += 1

        inventory.append({
            "file_name": filename,
            "file_path": path,
            "size_bytes": size,
            "size_MB": round(size / (1024**2), 3),
            "file_type": category
        })

# Save corrected inventory
inventory_path = os.path.join(
    OUTPUT_PATH, "THEIA_FILE_INVENTORY.csv"
)

with open(inventory_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
    writer.writeheader()
    writer.writerows(inventory)

print("=" * 65)
print("CORRECTED THEIA INVENTORY")
print("=" * 65)

print("Total files:", len(inventory))

for category, count in sorted(counts.items()):
    print(f"{category}: {count}")

# --------------------------------------------------
# 2. INSPECT JSONL RECORD STRUCTURE
# --------------------------------------------------

json_files = sorted(
    glob.glob(os.path.join(THEIA_PATH, "*.json*.gz"))
)

print("\n" + "=" * 65)
print("JSON RECORD INSPECTION")
print("=" * 65)

record_types = Counter()
json_examples = 0

if json_files:
    json_path = json_files[0]

    print("Sample:", os.path.basename(json_path))

    with gzip.open(json_path, "rt", encoding="utf-8") as f:

        for line_number, line in enumerate(f):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
                json_examples += 1

                record_type = record.get("type", "UNKNOWN")
                record_types[record_type] += 1

                if json_examples <= 5:
                    print(f"\nRecord {json_examples}")
                    print("Top-level fields:", list(record.keys()))
                    print("Record type:", record_type)

                    datum = record.get("datum", {})

                    if isinstance(datum, dict):
                        print("Datum types:", list(datum.keys()))

                        for datum_name, datum_value in datum.items():
                            if isinstance(datum_value, dict):
                                print(
                                    "Datum field names:",
                                    list(datum_value.keys())[:40]
                                )

                                if datum_name.endswith(".Event"):
                                    print("EVENT FIELD STRUCTURE:")
                                    print(json.dumps(
                                        datum_value,
                                        indent=2
                                    )[:2500])

                if json_examples >= 100:
                    break

            except Exception as e:
                print("JSON parsing issue at line:", line_number + 1)
                print(repr(e))
                break

    print("\nSampled JSON records:", json_examples)
    print("Sampled record types:", record_types)

else:
    print("No JSON files found.")

# --------------------------------------------------
# 3. INSPECT AVRO SCHEMA AND SAMPLE RECORDS
# --------------------------------------------------

avro_files = sorted(
    glob.glob(os.path.join(THEIA_PATH, "*.bin*.gz"))
)

print("\n" + "=" * 65)
print("AVRO SCHEMA INSPECTION")
print("=" * 65)

if avro_files:

    avro_path = avro_files[0]

    print("Sample:", os.path.basename(avro_path))

    with gzip.open(avro_path, "rb") as f:

        reader = fastavro.reader(f)

        schema = reader.writer_schema

        print("\nAVRO SCHEMA:")
        print(json.dumps(schema, indent=2)[:10000])

        print("\nFIRST 3 AVRO RECORDS:")

        for i, record in enumerate(reader):
            if i >= 3:
                break

            print(f"\nAVRO RECORD {i+1}")
            print("Record fields:", list(record.keys()))
            print("Record preview:")
            print(str(record)[:2500])

else:
    print("No binary Avro files found.")

print("\n" + "=" * 65)
print("STEP 2.3 INSPECTION COMPLETED")
print("Corrected inventory:", inventory_path)


# In[ ]:


get_ipython().run_line_magic('pip', 'install -q fastavro')


# In[ ]:


import fastavro

print("fastavro installed successfully!")
print("Version:", fastavro.__version__)


# In[ ]:


import os
import gzip
import json
import fastavro
import glob
from collections import Counter

THEIA_PATH = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"

def normalize_uuid(value):
    if isinstance(value, bytes):
        return value.hex().upper()
    if isinstance(value, str):
        return value.replace("-", "").upper()
    return str(value)

def json_fingerprint(record):
    datum = record.get("datum", {})

    event = next(
        (v for k, v in datum.items() if k.endswith(".Event")),
        None
    ) if isinstance(datum, dict) else None

    if event is None:
        return None

    return (
        normalize_uuid(event.get("uuid")),
        event.get("sequence", {}).get("long"),
        event.get("timestampNanos"),
        event.get("type")
    )

def avro_fingerprint(record):
    datum = record.get("datum", {})

    if record.get("type") != "RECORD_EVENT":
        return None

    return (
        normalize_uuid(datum.get("uuid")),
        datum.get("sequence"),
        datum.get("timestampNanos"),
        datum.get("type")
    )

json_files = sorted(
    glob.glob(os.path.join(THEIA_PATH, "*.json*.gz"))
)

print("THEIA JSON / AVRO COMPARISON")
print("=" * 60)

for json_path in json_files:

    avro_path = json_path.replace(".json", ".bin")

    print("\nJSON:", os.path.basename(json_path))

    if not os.path.exists(avro_path):
        print("Matching Avro file not found.")
        continue

    json_records = []
    avro_records = []

    try:
        with gzip.open(json_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                fp = json_fingerprint(record)

                if fp:
                    json_records.append(fp)

                if len(json_records) >= 100:
                    break

        with gzip.open(avro_path, "rb") as f:

            reader = fastavro.reader(
                f,
                handle_unicode_errors="replace"
            )

            for record in reader:

                fp = avro_fingerprint(record)

                if fp:
                    avro_records.append(fp)

                if len(avro_records) >= 100:
                    break

        matches = sum(
            a == b
            for a, b in zip(json_records, avro_records)
        )

        print("JSON events:", len(json_records))
        print("Avro events:", len(avro_records))
        print("Matching positions:", matches)

        if json_records == avro_records:
            print("RESULT: IDENTICAL SAMPLED RECORDS")
        elif matches:
            print("RESULT: PARTIAL OVERLAP")
        else:
            print("RESULT: NO MATCHING POSITIONS")

    except Exception as e:
        print("ERROR:", type(e).__name__, str(e))

print("\nCOMPARISON FINISHED")


# In[ ]:


import os
import glob
import gzip
import csv
import json
import hashlib
import fastavro
from collections import Counter

THEIA_PATH = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
OUTPUT_PATH = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

os.makedirs(OUTPUT_PATH, exist_ok=True)

AUDIT_FILE = os.path.join(
    OUTPUT_PATH, "THEIA_DATASET_AUDIT.csv"
)

CHECKPOINT_FILE = os.path.join(
    OUTPUT_PATH, "THEIA_AUDIT_CHECKPOINT.csv"
)

avro_files = sorted(
    glob.glob(os.path.join(THEIA_PATH, "*.bin*.gz"))
)

print("TOTAL AVRO FILES:", len(avro_files))

FIELDS = [
    "file_name", "size_bytes", "sha256",
    "duplicate_of", "total_records", "event_records",
    "record_types", "event_types",
    "min_timestamp_ns", "max_timestamp_ns",
    "missing_timestamps", "timestamp_order",
    "gzip_status", "parse_status", "error"
]

# Resume completed files
completed = {}

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, newline="") as f:
        for row in csv.DictReader(f):
            if row["parse_status"] == "COMPLETE":
                completed[row["file_name"]] = row

print("Previously completed:", len(completed))

# Existing SHA-256 lookup
hash_lookup = {}

for row in completed.values():
    if row["sha256"]:
        hash_lookup.setdefault(row["sha256"], row["file_name"])

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()

def save_checkpoint(row):
    exists = os.path.exists(CHECKPOINT_FILE)

    with open(CHECKPOINT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)

        if not exists:
            writer.writeheader()

        writer.writerow(row)

for index, path in enumerate(avro_files, 1):

    filename = os.path.basename(path)

    if filename in completed:
        print(f"[{index}/{len(avro_files)}] SKIPPED: {filename}")
        continue

    print(f"\n[{index}/{len(avro_files)}] PROCESSING: {filename}")

    row = {
        "file_name": filename,
        "size_bytes": os.path.getsize(path),
        "sha256": "",
        "duplicate_of": "",
        "total_records": 0,
        "event_records": 0,
        "record_types": "{}",
        "event_types": "{}",
        "min_timestamp_ns": "",
        "max_timestamp_ns": "",
        "missing_timestamps": 0,
        "timestamp_order": "UNKNOWN",
        "gzip_status": "UNKNOWN",
        "parse_status": "ERROR",
        "error": ""
    }

    try:
        # SHA-256 of original compressed file
        digest = sha256_file(path)
        row["sha256"] = digest

        if digest in hash_lookup:
            row["duplicate_of"] = hash_lookup[digest]
        else:
            hash_lookup[digest] = filename

        record_counts = Counter()
        event_counts = Counter()

        total = 0
        events = 0
        missing_ts = 0

        min_ts = None
        max_ts = None

        previous_ts = None
        ordering = True

        # Read entire compressed Avro file
        with gzip.open(path, "rb") as f:

            reader = fastavro.reader(
                f,
                handle_unicode_errors="replace"
            )

            for record in reader:

                total += 1

                record_type = record.get("type", "UNKNOWN")
                record_counts[record_type] += 1

                if record_type != "RECORD_EVENT":
                    continue

                events += 1

                datum = record.get("datum", {})

                event_type = datum.get("type", "UNKNOWN")
                event_counts[event_type] += 1

                timestamp = datum.get("timestampNanos")

                if timestamp is None:
                    missing_ts += 1
                    continue

                timestamp = int(timestamp)

                if min_ts is None or timestamp < min_ts:
                    min_ts = timestamp

                if max_ts is None or timestamp > max_ts:
                    max_ts = timestamp

                if previous_ts is not None and timestamp < previous_ts:
                    ordering = False

                previous_ts = timestamp

        row.update({
            "total_records": total,
            "event_records": events,
            "record_types": json.dumps(dict(record_counts)),
            "event_types": json.dumps(dict(event_counts)),
            "min_timestamp_ns": min_ts,
            "max_timestamp_ns": max_ts,
            "missing_timestamps": missing_ts,
            "timestamp_order": (
                "NONDECREASING" if ordering else "UNSORTED"
            ),
            "gzip_status": "PASSED",
            "parse_status": "COMPLETE"
        })

        print("Records:", total)
        print("Events:", events)
        print("Status: COMPLETE")

    except Exception as e:

        row["parse_status"] = "ERROR"
        row["error"] = repr(e)

        print("ERROR:", repr(e))

    # Save immediately after every file
    save_checkpoint(row)

print("\n" + "=" * 65)
print("FULL AVRO AUDIT PROCESSING FINISHED")
print("Checkpoint:", CHECKPOINT_FILE)


# In[ ]:


import shutil, os

disk = shutil.disk_usage("/content")
print("Free Colab storage:", round(disk.free / 1024**3, 2), "GB")

output_dir = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
print("Output folder exists:", os.path.exists(output_dir))
print("Audit checkpoint exists:",
      os.path.exists(os.path.join(output_dir, "THEIA_AUDIT_CHECKPOINT.csv")))


# In[ ]:


import os
import glob
import pandas as pd

source_dir = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
output_dir = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

# Find all compressed Avro partitions
avro_files = sorted(glob.glob(os.path.join(source_dir, "*.bin*.gz")))

# Read audit checkpoint
checkpoint_path = os.path.join(output_dir, "THEIA_AUDIT_CHECKPOINT.csv")
checkpoint = pd.read_csv(checkpoint_path)

print("Avro partitions found:", len(avro_files))
print("Checkpoint rows:", len(checkpoint))
print("Checkpoint columns:", checkpoint.columns.tolist())
print("\nFirst 3 Avro files:")
for f in avro_files[:3]:
    print(os.path.basename(f))

print("\nCheckpoint preview:")
display(checkpoint.head())


# In[ ]:


import pandas as pd
import os

checkpoint_path = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/THEIA_AUDIT_CHECKPOINT.csv"
df = pd.read_csv(checkpoint_path)

print("Completed partitions:", len(df))
print("Records already audited:", f"{df['total_records'].sum():,}")
print("Events already audited:", f"{df['event_records'].sum():,}")
print("Compressed size of audited files (GB):",
      round(df["size_bytes"].sum() / 1024**3, 2))

print("\nAverage records per partition:",
      f"{int(df['total_records'].mean()):,}")
print("Average events per partition:",
      f"{int(df['event_records'].mean()):,}")


# In[ ]:


import gzip
from fastavro import reader
import os

source_dir = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
first_file = sorted([
    f for f in os.listdir(source_dir)
    if f.endswith(".gz") and ".bin" in f
])[0]

path = os.path.join(source_dir, first_file)

with gzip.open(path, "rb") as f:
    avro_reader = reader(f, handle_unicode_errors="replace")
    for record in avro_reader:
        if record.get("type") == "RECORD_EVENT":
            event = record.get("datum", {})
            print("Event fields:")
            for key, value in event.items():
                print(key, "=>", type(value).__name__)
            break


# In[ ]:


get_ipython().run_line_magic('pip', 'install -q fastavro pyarrow duckdb scikit-learn')


# In[ ]:


import fastavro, pyarrow, duckdb, sklearn
import torch

print("fastavro:", fastavro.__version__)
print("pyarrow:", pyarrow.__version__)
print("duckdb:", duckdb.__version__)
print("sklearn:", sklearn.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not detected")


# In[ ]:


import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU memory (GB):", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ))
else:
    print("GPU not connected yet")


# In[ ]:


import os
from google.colab import drive

if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

source_dir = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
output_dir = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

assert os.path.isdir(source_dir), "THEIA source folder not found"
assert os.path.isdir(output_dir), "Output folder not found"

import glob, shutil

avro_files = sorted(glob.glob(os.path.join(source_dir, "*.bin*.gz")))

print("Avro partitions:", len(avro_files))
print("Output folder:", output_dir)
print("Free Colab disk (GB):", round(
    shutil.disk_usage("/content").free / 1024**3, 2
))
print("GPU:", __import__("torch").cuda.get_device_name(0))


# In[ ]:


import pandas as pd
import os, glob

out = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
src = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"

audit = pd.read_csv(os.path.join(out, "THEIA_AUDIT_CHECKPOINT.csv"))

completed = set(audit.loc[
    audit["parse_status"].eq("COMPLETE"), "file_name"
])

files = sorted(glob.glob(os.path.join(src, "*.bin*.gz")))
remaining = [f for f in files if os.path.basename(f) not in completed]

print("Completed partitions:", len(completed))
print("Remaining partitions:", len(remaining))
print("Remaining compressed size (GB):",
      round(sum(os.path.getsize(f) for f in remaining) / 1024**3, 2))
print("Total audited events:", f"{audit['event_records'].sum():,}")
print("Timestamp range:")
print("Min:", audit["min_timestamp_ns"].min())
print("Max:", audit["max_timestamp_ns"].max())


# In[ ]:


import pandas as pd
import os

out = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
audit = pd.read_csv(os.path.join(out, "THEIA_AUDIT_CHECKPOINT.csv"))

audit["min_timestamp_ns"] = pd.to_numeric(
    audit["min_timestamp_ns"], errors="coerce"
)
audit["max_timestamp_ns"] = pd.to_numeric(
    audit["max_timestamp_ns"], errors="coerce"
)

audit["start_utc"] = pd.to_datetime(
    audit["min_timestamp_ns"], unit="ns", utc=True
)
audit["end_utc"] = pd.to_datetime(
    audit["max_timestamp_ns"], unit="ns", utc=True
)

# Sort BEFORE selecting columns
result = audit.sort_values("min_timestamp_ns")

display(result[
    ["file_name", "event_records", "start_utc", "end_utc"]
].to_string(index=False))

print("\nOverall time range:")
print("Start:", audit["start_utc"].min())
print("End:", audit["end_utc"].max())


# In[ ]:


import pandas as pd
import os

out = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
audit = pd.read_csv(os.path.join(out, "THEIA_AUDIT_CHECKPOINT.csv"))

audit["min_timestamp_ns"] = pd.to_numeric(audit["min_timestamp_ns"])
audit["max_timestamp_ns"] = pd.to_numeric(audit["max_timestamp_ns"])

a = audit.sort_values("min_timestamp_ns").reset_index(drop=True)

a["previous_max"] = a["max_timestamp_ns"].shift(1)
a["gap_seconds"] = (
    a["min_timestamp_ns"] - a["previous_max"]
) / 1e9

print("Partitions checked:", len(a))
print("Overlapping partitions:", (a["gap_seconds"] < 0).sum())
print("Partitions with gaps over 1 second:",
      (a["gap_seconds"] > 1).sum())
print("Largest positive gap (seconds):",
      a["gap_seconds"].max())

print("\nPartition time ranges:")
display(a[["file_name", "gap_seconds"]].to_string(index=False))


# In[ ]:


get_ipython().run_line_magic('pip', 'install -q fastavro pyarrow duckdb scikit-learn')


# In[ ]:


import fastavro
import pyarrow
import duckdb
import sklearn
import torch

print("fastavro:", fastavro.__version__)
print("pyarrow:", pyarrow.__version__)
print("duckdb:", duckdb.__version__)
print("sklearn:", sklearn.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not detected")


# In[ ]:


import gzip
from fastavro import reader
import os, glob

source_dir = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
files = sorted(glob.glob(os.path.join(source_dir, "*.bin*.gz")))

with gzip.open(files[0], "rb") as f:
    r = reader(f, handle_unicode_errors="replace")
    for record in r:
        if record.get("type") == "RECORD_EVENT":
            event = record.get("datum", {})
            print("Event properties:")
            print(event.get("properties"))
            print("\nNames:")
            print(event.get("names"))
            print("\nProgram point:")
            print(event.get("programPoint"))
            break


# In[ ]:


import gzip, os, glob
from fastavro import reader

source_dir = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
files = sorted(glob.glob(os.path.join(source_dir, "*.bin*.gz")))

shown = 0

with gzip.open(files[0], "rb") as f:
    for record in reader(f, handle_unicode_errors="replace"):
        if record.get("type") != "RECORD_EVENT":
            continue

        e = record.get("datum", {})
        print({
            "type": e.get("type"),
            "sequence": e.get("sequence"),
            "threadId": e.get("threadId"),
            "names": e.get("names"),
            "size": e.get("size"),
            "subject_present": e.get("subject") is not None,
            "predicateObject_present": e.get("predicateObject") is not None,
            "predicateObject2_present": e.get("predicateObject2") is not None
        })

        shown += 1
        if shown == 10:
            break


# In[ ]:


import os, json, glob, shutil
import pandas as pd
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

SOURCE_DIR = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
OUTPUT_DIR = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted_features")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "extraction_checkpoints")

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

avro_files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.bin*.gz")))

print("Avro partitions:", len(avro_files))
print("Extraction directory:", EXTRACT_DIR)
print("Checkpoint directory:", CHECKPOINT_DIR)
print("Free disk GB:", round(shutil.disk_usage("/content").free / 1024**3, 2))
print("GPU:", torch.cuda.get_device_name(0))


# In[ ]:


import os
import glob
import json
import gzip
import time
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import reader

SOURCE_DIR = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
OUTPUT_DIR = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted_features")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "extraction_checkpoints")

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR, "extraction_progress.json"
)

BATCH_SIZE = 100_000

files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.bin*.gz")))
assert len(files) == 121, f"Expected 121 partitions, found {len(files)}"

schema = pa.schema([
    ("timestamp_ns", pa.int64()),
    ("sequence", pa.int64()),
    ("thread_id", pa.int64()),
    ("event_type", pa.string()),
    ("operation", pa.string()),
    ("size", pa.int64()),
    ("subject_present", pa.bool_()),
    ("object_present", pa.bool_()),
    ("object2_present", pa.bool_()),
    ("source_partition", pa.int16()),
    ("source_record_index", pa.int64())
])

if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, "r") as f:
        progress = json.load(f)
else:
    progress = {"completed": {}}

completed = progress["completed"]

print("Total partitions:", len(files))
print("Already extracted:", len(completed))
print("Remaining:", len(files) - len(completed))
print("Batch size:", BATCH_SIZE)

def save_progress():
    temp = CHECKPOINT_FILE + ".tmp"
    with open(temp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(temp, CHECKPOINT_FILE)

def safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None

for partition_index, path in enumerate(files):
    filename = os.path.basename(path)
    partition_key = f"{partition_index:03d}_{filename}"
    output_file = os.path.join(
        EXTRACT_DIR, partition_key + ".parquet"
    )
    temp_file = output_file + ".tmp"

    if partition_key in completed and os.path.exists(output_file):
        print(f"SKIP {partition_index + 1}/121: {filename}")
        continue

    if os.path.exists(temp_file):
        os.remove(temp_file)

    start_time = time.time()
    buffers = {field.name: [] for field in schema}
    batch_count = 0
    event_count = 0
    record_index = 0
    min_ts = None
    max_ts = None
    writer = None

    print(f"\nSTART {partition_index + 1}/121: {filename}")

    try:
        with gzip.open(path, "rb") as gz:
            avro_reader = reader(
                gz, handle_unicode_errors="replace"
            )

            for record in avro_reader:
                current_record_index = record_index
                record_index += 1

                if record.get("type") != "RECORD_EVENT":
                    continue

                event = record.get("datum") or {}

                ts = safe_int(event.get("timestampNanos"))
                seq = safe_int(event.get("sequence"))
                thread = safe_int(event.get("threadId"))
                size = safe_int(event.get("size"))

                if ts is not None:
                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    max_ts = ts if max_ts is None else max(max_ts, ts)

                names = event.get("names") or []
                operation = "|".join(
                    str(x) for x in names if x is not None
                ) or "UNKNOWN"

                values = {
                    "timestamp_ns": ts,
                    "sequence": seq,
                    "thread_id": thread,
                    "event_type": str(event.get("type") or "UNKNOWN"),
                    "operation": operation,
                    "size": size,
                    "subject_present": event.get("subject") is not None,
                    "object_present": event.get("predicateObject") is not None,
                    "object2_present": event.get("predicateObject2") is not None,
                    "source_partition": partition_index,
                    "source_record_index": current_record_index
                }

                for key, value in values.items():
                    buffers[key].append(value)

                event_count += 1

                if event_count % BATCH_SIZE == 0:
                    table = pa.Table.from_pydict(
                        buffers, schema=schema
                    )

                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_file,
                            schema,
                            compression="zstd",
                            compression_level=3
                        )

                    writer.write_table(table)
                    buffers = {field.name: [] for field in schema}
                    batch_count += 1

                    if batch_count % 5 == 0:
                        print(
                            f"  {event_count:,} events written"
                        )

            # Write final partial batch
            if buffers["timestamp_ns"]:
                table = pa.Table.from_pydict(
                    buffers, schema=schema
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_file,
                        schema,
                        compression="zstd",
                        compression_level=3
                    )

                writer.write_table(table)
                batch_count += 1

            if writer is not None:
                writer.close()
                writer = None

            # Handle a partition with zero event records
            if not os.path.exists(temp_file):
                empty = pa.Table.from_pydict(
                    {field.name: [] for field in schema},
                    schema=schema
                )
                pq.write_table(empty, temp_file, compression="zstd")

        os.replace(temp_file, output_file)

        progress["completed"][partition_key] = {
            "event_count": event_count,
            "record_count": record_index,
            "min_timestamp_ns": min_ts,
            "max_timestamp_ns": max_ts,
            "parquet_bytes": os.path.getsize(output_file)
        }
        save_progress()

        elapsed = time.time() - start_time
        print(
            f"COMPLETE {partition_index + 1}/121 | "
            f"Events: {event_count:,} | "
            f"Time: {elapsed / 60:.1f} min | "
            f"Output: {os.path.getsize(output_file) / 1024**2:.1f} MB"
        )

        print(
            "Free disk GB:",
            round(shutil.disk_usage("/content").free / 1024**3, 2)
        )

    except Exception as exc:
        if writer is not None:
            writer.close()
        print(f"ERROR in {filename}: {type(exc).__name__}: {exc}")
        print("This partition was NOT marked complete.")
        raise

print("\nEXTRACTION LOOP FINISHED")
print("Completed partitions:", len(progress["completed"]))
print("Checkpoint:", CHECKPOINT_FILE)


# In[ ]:


get_ipython().run_line_magic('pip', 'install -q fastavro pyarrow duckdb scikit-learn')


# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os, json, glob, shutil

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

checkpoint = os.path.join(
    OUT, "extraction_checkpoints", "extraction_progress.json"
)

print("Extraction checkpoint exists:", os.path.exists(checkpoint))

if os.path.exists(checkpoint):
    with open(checkpoint) as f:
        progress = json.load(f)
    print("Completed extraction partitions:",
          len(progress["completed"]))

print("Audit checkpoint exists:",
      os.path.exists(os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")))

print("Free disk GB:",
      round(shutil.disk_usage("/content").free / 1024**3, 2))


# In[ ]:


import fastavro, pyarrow, duckdb
print("All extraction packages ready")


# In[ ]:


import json

p = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/extraction_checkpoints/extraction_progress.json"

with open(p) as f:
    progress = json.load(f)

print("Completed partitions:", len(progress["completed"]))
print("Total extracted events:",
      f"{sum(x['event_count'] for x in progress['completed'].values()):,}")


# In[ ]:


import fastavro, pyarrow, duckdb
print("Extraction packages ready")


# In[ ]:


import json

p = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/extraction_checkpoints/extraction_progress.json"

with open(p) as f:
    progress = json.load(f)

items = progress["completed"]

print("Completed:", len(items), "/ 121")
print("Total events:",
      f"{sum(x['event_count'] for x in items.values()):,}")
print("Total source records:",
      f"{sum(x['record_count'] for x in items.values()):,}")


# In[ ]:


import torch
print("CUDA available:", torch.cuda.is_available())


# In[ ]:


BATCH_SIZE = 100_000


# In[ ]:


import os
import glob
import json
import gzip
import time
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from fastavro import reader

SOURCE_DIR = "/content/drive/MyDrive/SAE_THEIA_SOURCE/theia"
OUTPUT_DIR = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
EXTRACT_DIR = os.path.join(OUTPUT_DIR, "extracted_features")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "extraction_checkpoints")
CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR, "extraction_progress.json"
)

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 100_000

files = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.bin*.gz")))
assert len(files) == 121, f"Expected 121 partitions, found {len(files)}"

assert os.path.exists(CHECKPOINT_FILE), "Extraction checkpoint missing!"

with open(CHECKPOINT_FILE) as f:
    progress = json.load(f)

completed = progress["completed"]

schema = pa.schema([
    ("timestamp_ns", pa.int64()),
    ("sequence", pa.int64()),
    ("thread_id", pa.int64()),
    ("event_type", pa.string()),
    ("operation", pa.string()),
    ("size", pa.int64()),
    ("subject_present", pa.bool_()),
    ("object_present", pa.bool_()),
    ("object2_present", pa.bool_()),
    ("source_partition", pa.int16()),
    ("source_record_index", pa.int64())
])

def safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return None

def save_progress():
    temp = CHECKPOINT_FILE + ".tmp"
    with open(temp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(temp, CHECKPOINT_FILE)

print("Total partitions:", len(files))
print("Already completed:", len(completed))
print("Remaining:", len(files) - len(completed))
print("Free disk GB:",
      round(shutil.disk_usage("/content").free / 1024**3, 2))

for partition_index, path in enumerate(files):
    filename = os.path.basename(path)
    partition_key = f"{partition_index:03d}_{filename}"

    output_file = os.path.join(
        EXTRACT_DIR, partition_key + ".parquet"
    )
    temp_file = output_file + ".tmp"

    if partition_key in completed and os.path.exists(output_file):
        print(f"SKIP {partition_index + 1}/121: {filename}")
        continue

    # Do not trust a checkpoint if its output file is missing.
    if partition_key in completed:
        del completed[partition_key]
        save_progress()

    if os.path.exists(temp_file):
        os.remove(temp_file)

    start_time = time.time()
    buffers = {field.name: [] for field in schema}
    event_count = 0
    record_index = 0
    batch_count = 0
    min_ts = None
    max_ts = None
    writer = None

    print(f"\nSTART {partition_index + 1}/121: {filename}")

    try:
        with gzip.open(path, "rb") as gz:
            avro_reader = reader(
                gz, handle_unicode_errors="replace"
            )

            for record in avro_reader:
                current_record_index = record_index
                record_index += 1

                if record.get("type") != "RECORD_EVENT":
                    continue

                event = record.get("datum") or {}
                ts = safe_int(event.get("timestampNanos"))

                if ts is not None:
                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    max_ts = ts if max_ts is None else max(max_ts, ts)

                names = event.get("names") or []
                operation = "|".join(
                    str(x) for x in names if x is not None
                ) or "UNKNOWN"

                values = {
                    "timestamp_ns": ts,
                    "sequence": safe_int(event.get("sequence")),
                    "thread_id": safe_int(event.get("threadId")),
                    "event_type": str(event.get("type") or "UNKNOWN"),
                    "operation": operation,
                    "size": safe_int(event.get("size")),
                    "subject_present": event.get("subject") is not None,
                    "object_present": event.get("predicateObject") is not None,
                    "object2_present": event.get("predicateObject2") is not None,
                    "source_partition": partition_index,
                    "source_record_index": current_record_index
                }

                for key, value in values.items():
                    buffers[key].append(value)

                event_count += 1

                if event_count % BATCH_SIZE == 0:
                    table = pa.Table.from_pydict(
                        buffers, schema=schema
                    )

                    if writer is None:
                        writer = pq.ParquetWriter(
                            temp_file,
                            schema,
                            compression="zstd",
                            compression_level=3
                        )

                    writer.write_table(table)
                    buffers = {field.name: [] for field in schema}
                    batch_count += 1

                    if batch_count % 5 == 0:
                        print(f"  Written {event_count:,} events")

            # Write final partial batch
            if buffers["timestamp_ns"]:
                table = pa.Table.from_pydict(
                    buffers, schema=schema
                )

                if writer is None:
                    writer = pq.ParquetWriter(
                        temp_file,
                        schema,
                        compression="zstd",
                        compression_level=3
                    )

                writer.write_table(table)

            if writer is not None:
                writer.close()
                writer = None

            if not os.path.exists(temp_file):
                empty = pa.Table.from_pydict(
                    {field.name: [] for field in schema},
                    schema=schema
                )
                pq.write_table(empty, temp_file, compression="zstd")

        os.replace(temp_file, output_file)

        completed[partition_key] = {
            "event_count": event_count,
            "record_count": record_index,
            "min_timestamp_ns": min_ts,
            "max_timestamp_ns": max_ts,
            "parquet_bytes": os.path.getsize(output_file)
        }

        save_progress()

        print(
            f"COMPLETE {partition_index + 1}/121 | "
            f"Events: {event_count:,} | "
            f"Time: {(time.time() - start_time)/60:.1f} min"
        )

        print("Completed partitions:", len(completed), "/ 121")
        print("Free disk GB:",
              round(shutil.disk_usage("/content").free / 1024**3, 2))

    except Exception as exc:
        if writer is not None:
            writer.close()
        print("ERROR:", repr(exc))
        print("Partition not marked complete. You can resume later.")
        raise

print("\nEXTRACTION FINISHED")
print("Completed:", len(completed), "/ 121")
print("Total extracted events:",
      f"{sum(x['event_count'] for x in completed.values()):,}")


# In[ ]:


import pandas as pd
import json
import os
from collections import Counter

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

audit = pd.read_csv(
    os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")
)

record_counts = Counter()

for value in audit["record_types"].dropna():
    try:
        record_counts.update(json.loads(value))
    except Exception:
        pass

print("RECORD TYPES FOUND IN THE FULL DATASET")
print("=" * 55)

for name, count in record_counts.most_common():
    print(f"{name:35s} {count:>18,}")

print("\nTotal record types:", len(record_counts))
print("Total source records:", f"{sum(record_counts.values()):,}")


# In[ ]:


import pandas as pd
import os

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
path = os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")

df = pd.read_csv(path, nrows=5)

print("Columns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df)

print("\nData types:")
print(df.dtypes)


# In[ ]:


import pandas as pd
import os

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
path = os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")

df = pd.read_csv(path)

print("PARTITION AUDIT")
print("=" * 60)
print("Partitions:", len(df))
print("Total source records:", f"{df['total_records'].sum():,}")
print("Total event records:", f"{df['event_records'].sum():,}")
print("Missing timestamps:", f"{df['missing_timestamps'].sum():,}")

print("\nParse status:")
print(df["parse_status"].value_counts(dropna=False))

print("\nGzip status:")
print(df["gzip_status"].value_counts(dropna=False))

print("\nTimestamp order:")
print(df["timestamp_order"].value_counts(dropna=False))

print("\nErrors:", df["error"].notna().sum())

print("\nPartition time ranges:")
display(
    df[[
        "file_name",
        "event_records",
        "min_timestamp_ns",
        "max_timestamp_ns",
        "timestamp_order"
    ]].sort_values("min_timestamp_ns").head(15)
)


# In[ ]:


import pandas as pd
import os

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
path = os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")

df = pd.read_csv(path)

print("PARTITION AUDIT")
print("=" * 60)
print("Partitions:", len(df))
print("Total source records:", f"{df['total_records'].sum():,}")
print("Total event records:", f"{df['event_records'].sum():,}")
print("Missing timestamps:", f"{df['missing_timestamps'].sum():,}")

print("\nParse status:")
print(df["parse_status"].value_counts(dropna=False))

print("\nGzip status:")
print(df["gzip_status"].value_counts(dropna=False))

print("\nTimestamp order:")
print(df["timestamp_order"].value_counts(dropna=False))

print("\nErrors:", df["error"].notna().sum())

print("\nPartition time ranges:")
display(
    df[[
        "file_name",
        "event_records",
        "min_timestamp_ns",
        "max_timestamp_ns",
        "timestamp_order"
    ]].sort_values("min_timestamp_ns").head(15)
)


# In[ ]:


import os
import glob
import pandas as pd

OUT = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"

print("FILES IN OUTPUT FOLDER")
for f in sorted(glob.glob(os.path.join(OUT, "*"))):
    if os.path.isfile(f):
        print(os.path.basename(f), f"{os.path.getsize(f)/1024**3:.2f} GB")

audit_path = os.path.join(OUT, "THEIA_AUDIT_CHECKPOINT.csv")
audit = pd.read_csv(audit_path)

print("\nAUDIT CHECKPOINT")
print("Rows:", len(audit))
print("Unique filenames:", audit["file_name"].nunique())
print("Duplicate filenames:", audit["file_name"].duplicated().sum())
print("Total records:", f"{audit['total_records'].sum():,}")
print("Event records:", f"{audit['event_records'].sum():,}")

print("\nPotential extraction outputs:")
for pattern in ["*.parquet", "*.csv", "*.jsonl", "*.npz", "*.npy"]:
    matches = glob.glob(os.path.join(OUT, "**", pattern), recursive=True)
    print(pattern, ":", len(matches))
    for f in matches[:10]:
        print(" ", f)


# In[ ]:


import os
import glob
import pyarrow.parquet as pq

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
PARQUET_DIR = os.path.join(BASE, "extracted_features")

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))

print("PARQUET FILE COUNT:", len(files))
print("=" * 65)

total_rows = 0
total_bytes = 0

for f in files:
    pf = pq.ParquetFile(f)
    rows = pf.metadata.num_rows
    total_rows += rows
    total_bytes += os.path.getsize(f)

print("Total Parquet rows:", f"{total_rows:,}")
print("Total Parquet size:", f"{total_bytes / 1024**3:.2f} GB")

print("\nFIRST FILE SCHEMA")
pf = pq.ParquetFile(files[0])
print(pf.schema)

print("\nFIRST 5 ROWS")
sample = pf.read_row_group(0).slice(0, 5).to_pandas()
display(sample)

print("\nCOLUMN NAMES")
print(sample.columns.tolist())

print("\nSCHEMA CONSISTENCY")
schemas = {}
for f in files:
    schema = str(pq.ParquetFile(f).schema_arrow)
    schemas[schema] = schemas.get(schema, 0) + 1

print("Different schemas:", len(schemas))
for i, (schema, count) in enumerate(schemas.items(), 1):
    print(f"\nSchema {i}: {count} files")
    print(schema)


# In[ ]:


import os
import glob
import pyarrow.parquet as pq

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
PARQUET_DIR = os.path.join(BASE, "extracted_features")

files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))

global_min = None
global_max = None
rows = 0
missing_stats = 0

for i, path in enumerate(files, 1):
    pf = pq.ParquetFile(path)
    rows += pf.metadata.num_rows

    ts_index = pf.schema_arrow.get_field_index("timestamp_ns")

    for rg in range(pf.metadata.num_row_groups):
        col = pf.metadata.row_group(rg).column(ts_index)
        stats = col.statistics

        if stats is None or not stats.has_min_max:
            missing_stats += 1
            continue

        mn, mx = stats.min, stats.max

        global_min = mn if global_min is None else min(global_min, mn)
        global_max = mx if global_max is None else max(global_max, mx)

    if i % 20 == 0:
        print(f"Checked {i}/{len(files)} files")

print("\nRESULT")
print("Rows:", f"{rows:,}")
print("Global minimum timestamp:", global_min)
print("Global maximum timestamp:", global_max)
print("Row groups missing timestamp statistics:", missing_stats)

if global_min is not None and global_max is not None:
    span = global_max - global_min
    train_end = int(global_min + span * 0.70)
    val_end = int(global_min + span * 0.85)

    print("\nPROPOSED CHRONOLOGICAL BOUNDARIES")
    print("Train:", global_min, "to", train_end)
    print("Validation:", train_end, "to", val_end)
    print("Test:", val_end, "to", global_max)


# In[ ]:


import os
import glob
import csv
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc
import pyarrow.parquet as pq

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
SOURCE = os.path.join(BASE, "extracted_features")
OUTPUT = os.path.join(BASE, "dataset_splits")

TRAIN_END = 1557861974939535872
VAL_END   = 1557996262524545024

BATCH_SIZE = 250_000
SHARD_ROWS = 5_000_000

files = sorted(glob.glob(os.path.join(SOURCE, "*.parquet")))
assert len(files) == 121, f"Expected 121 files, found {len(files)}"

# Prevent accidental overwriting of previous split output.
if os.path.exists(OUTPUT) and os.listdir(OUTPUT):
    raise RuntimeError(
        f"{OUTPUT} already exists and is not empty. "
        "Rename or inspect it before running this cell again."
    )

for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(OUTPUT, split), exist_ok=True)

dataset = ds.dataset(files, format="parquet")
schema = dataset.schema

print("Source schema:")
print(schema)

writers = {"train": None, "validation": None, "test": None}
shard_numbers = {"train": 0, "validation": 0, "test": 0}
shard_row_counts = {"train": 0, "validation": 0, "test": 0}
total_counts = {"train": 0, "validation": 0, "test": 0}
null_timestamps = 0

def write_split(split, table):
    if table.num_rows == 0:
        return

    total_counts[split] += table.num_rows

    offset = 0
    while offset < table.num_rows:
        remaining = SHARD_ROWS - shard_row_counts[split]
        take = min(remaining, table.num_rows - offset)
        piece = table.slice(offset, take)

        if writers[split] is None:
            shard_numbers[split] += 1
            path = os.path.join(
                OUTPUT,
                split,
                f"part-{shard_numbers[split]:05d}.parquet"
            )
            writers[split] = pq.ParquetWriter(
                path,
                schema,
                compression="zstd"
            )

        writers[split].write_table(piece)
        shard_row_counts[split] += take
        offset += take

        if shard_row_counts[split] >= SHARD_ROWS:
            writers[split].close()
            writers[split] = None
            shard_row_counts[split] = 0

scanner = dataset.scanner(batch_size=BATCH_SIZE)

for batch_number, batch in enumerate(scanner.to_batches(), 1):
    table = pa.Table.from_batches([batch])
    ts = table["timestamp_ns"]

    null_timestamps += ts.null_count

    # Ignore null timestamps for splitting; report them for investigation.
    valid = pc.is_valid(ts)
    train_mask = pc.and_(valid, pc.less(ts, TRAIN_END))

    val_mask = pc.and_(
        valid,
        pc.and_(
            pc.greater_equal(ts, TRAIN_END),
            pc.less(ts, VAL_END)
        )
    )

    test_mask = pc.and_(
        valid,
        pc.greater_equal(ts, VAL_END)
    )

    write_split("train", table.filter(train_mask))
    write_split("validation", table.filter(val_mask))
    write_split("test", table.filter(test_mask))

    if batch_number % 100 == 0:
        print(
            f"Batch {batch_number:,} | "
            f"Train {total_counts['train']:,} | "
            f"Validation {total_counts['validation']:,} | "
            f"Test {total_counts['test']:,}"
        )

# Close remaining writers.
for split in writers:
    if writers[split] is not None:
        writers[split].close()
        writers[split] = None

# Record split boundaries and counts.
manifest_path = os.path.join(OUTPUT, "split_manifest.csv")

manifest = [
    {
        "split": "train",
        "timestamp_min_inclusive": "1557235299542826897",
        "timestamp_max_exclusive": str(TRAIN_END),
        "rows_written": total_counts["train"],
        "parquet_shards": shard_numbers["train"]
    },
    {
        "split": "validation",
        "timestamp_min_inclusive": str(TRAIN_END),
        "timestamp_max_exclusive": str(VAL_END),
        "rows_written": total_counts["validation"],
        "parquet_shards": shard_numbers["validation"]
    },
    {
        "split": "test",
        "timestamp_min_inclusive": str(VAL_END),
        "timestamp_max_inclusive": "1558130550109553898",
        "rows_written": total_counts["test"],
        "parquet_shards": shard_numbers["test"]
    }
]

with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print("\nSPLITTING COMPLETE")
print("=" * 55)

for split in ["train", "validation", "test"]:
    print(
        f"{split.upper():12s}: "
        f"{total_counts[split]:,} rows | "
        f"{shard_numbers[split]} Parquet shards"
    )

print("\nTotal rows written:", f"{sum(total_counts.values()):,}")
print("Null timestamps skipped:", f"{null_timestamps:,}")
print("Manifest:", manifest_path)


# In[ ]:


import os
import glob
import pyarrow.parquet as pq

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
OUTPUT = os.path.join(BASE, "dataset_splits")

for split in ["train", "validation", "test"]:
    folder = os.path.join(OUTPUT, split)
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))

    rows = 0
    size = 0

    for f in files:
        pf = pq.ParquetFile(f)
        rows += pf.metadata.num_rows
        size += os.path.getsize(f)

    print(f"\n{split.upper()}")
    print("Parquet files:", len(files))
    print("Rows:", f"{rows:,}")
    print("Size:", f"{size / 1024**3:.2f} GB")

    if files:
        print("First file:", os.path.basename(files[0]))
        print("Last file:", os.path.basename(files[-1]))


# In[ ]:


import pyarrow.parquet as pq
import glob
import os

for split in ["train", "validation", "test"]:
    files = glob.glob(
        os.path.join(OUTPUT, split, "*.parquet")
    )

    print("\n", split.upper())

    if not files:
        print("No Parquet files found")
        continue

    try:
        pf = pq.ParquetFile(files[0])
        print("Readable:", os.path.basename(files[0]))
        print("Rows:", pf.metadata.num_rows)
    except Exception as e:
        print("Could not read:", repr(e))


# In[ ]:


import os
import glob
import time
import csv
import pyarrow.parquet as pq

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
OUTPUT = os.path.join(BASE, "dataset_splits")

manifest = []
failures = []

for split in ["train", "validation", "test"]:
    files = sorted(glob.glob(
        os.path.join(OUTPUT, split, "*.parquet")
    ))

    total_rows = 0
    total_size = 0
    good_files = 0

    for f in files:
        success = False

        for attempt in range(3):
            try:
                pf = pq.ParquetFile(f)
                rows = pf.metadata.num_rows
                size = os.path.getsize(f)

                total_rows += rows
                total_size += size
                good_files += 1
                success = True
                break

            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    failures.append((split, f, repr(e)))

        if not success:
            print("FAILED:", split, os.path.basename(f))

    manifest.append({
        "split": split,
        "parquet_files": len(files),
        "readable_files": good_files,
        "rows": total_rows,
        "size_bytes": total_size
    })

    print(
        f"{split.upper()}: {good_files}/{len(files)} readable | "
        f"{total_rows:,} rows | {total_size/1024**3:.2f} GB"
    )

manifest_path = os.path.join(OUTPUT, "split_manifest.csv")

with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
    writer.writeheader()
    writer.writerows(manifest)

print("\nTOTAL READABLE ROWS:",
      f"{sum(x['rows'] for x in manifest):,}")
print("FAILED FILES:", len(failures))
print("Manifest:", manifest_path)


# In[ ]:


import os
import glob
import pyarrow.parquet as pq
import pyarrow.compute as pc

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/dataset_splits"

for split in ["train", "validation", "test"]:
    files = glob.glob(os.path.join(BASE, split, "*.parquet"))

    min_ts = None
    max_ts = None
    rows = 0

    for f in files:
        table = pq.read_table(
            f,
            columns=["timestamp_ns"]
        )

        stats = pc.min_max(table["timestamp_ns"])
        mn = stats["min"].as_py()
        mx = stats["max"].as_py()

        min_ts = mn if min_ts is None else min(min_ts, mn)
        max_ts = mx if max_ts is None else max(max_ts, mx)
        rows += table.num_rows

    print(f"\n{split.upper()}")
    print("Rows:", f"{rows:,}")
    print("Minimum timestamp:", min_ts)
    print("Maximum timestamp:", max_ts)


# In[ ]:


import os
import glob
import json
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED"
SPLIT_DIR = os.path.join(BASE, "dataset_splits")
FEATURE_DIR = os.path.join(BASE, "numerical_features")

BATCH_SIZE = 250_000
DAY_SECONDS = 86400.0

# Safety: don't overwrite previous work.
if os.path.exists(FEATURE_DIR) and os.listdir(FEATURE_DIR):
    raise RuntimeError(
        f"{FEATURE_DIR} already exists and is not empty. "
        "Inspect it before rerunning."
    )

for split in ["train", "validation", "test"]:
    os.makedirs(os.path.join(FEATURE_DIR, split), exist_ok=True)

# --------------------------------------------------
# PASS 1: Learn categorical vocabularies from TRAIN
# --------------------------------------------------

train_files = sorted(
    glob.glob(os.path.join(SPLIT_DIR, "train", "*.parquet"))
)

event_counts = Counter()
operation_counts = Counter()

size_sum = 0.0
size_sq_sum = 0.0
size_count = 0
negative_sizes = 0
null_sizes = 0

print("PASS 1: Learning training-only preprocessing")

for file_no, path in enumerate(train_files, 1):
    pf = pq.ParquetFile(path)

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["event_type", "operation", "size"]
    ):
        data = batch.to_pydict()

        event_counts.update(
            x for x in data["event_type"] if x is not None
        )
        operation_counts.update(
            x for x in data["operation"] if x is not None
        )

        values = np.asarray(
            [x for x in data["size"] if x is not None],
            dtype=np.float64
        )

        null_sizes += len(data["size"]) - len(values)
        negative_sizes += int(np.sum(values < 0))

        # log1p is valid only for nonnegative sizes.
        values = values[values >= 0]
        values = np.log1p(values)

        size_sum += values.sum(dtype=np.float64)
        size_sq_sum += np.square(values).sum(dtype=np.float64)
        size_count += len(values)

    if file_no % 20 == 0:
        print(f"Scanned {file_no}/{len(train_files)} train files")

event_vocab = sorted(event_counts.keys())
operation_vocab = sorted(operation_counts.keys())

if not event_vocab or not operation_vocab:
    raise RuntimeError("Training categorical vocabulary is empty.")

size_mean = size_sum / max(size_count, 1)
size_var = size_sq_sum / max(size_count, 1) - size_mean**2
size_std = math.sqrt(max(size_var, 1e-12))

print("\nTRAINING PREPROCESSING PARAMETERS")
print("Event categories:", len(event_vocab))
print("Operation categories:", len(operation_vocab))
print("Size mean after log1p:", size_mean)
print("Size standard deviation:", size_std)
print("Negative sizes excluded from size feature:", negative_sizes)
print("Null sizes:", null_sizes)

event_map = {v: i for i, v in enumerate(event_vocab)}
operation_map = {v: i for i, v in enumerate(operation_vocab)}

# --------------------------------------------------
# Save preprocessing configuration
# --------------------------------------------------

config = {
    "event_vocab": event_vocab,
    "operation_vocab": operation_vocab,
    "size_log_mean": size_mean,
    "size_log_std": size_std,
    "feature_definition": {
        "size_scaled": "standardized log1p(size)",
        "event_type": "one-hot",
        "operation": "one-hot",
        "time_sin": "UTC time-of-day sine",
        "time_cos": "UTC time-of-day cosine",
        "subject_present": "binary",
        "object_present": "binary",
        "object2_present": "binary"
    },
    "excluded_from_model": [
        "thread_id",
        "sequence",
        "source_partition",
        "source_record_index",
        "raw_timestamp_ns"
    ]
}

config_path = os.path.join(FEATURE_DIR, "preprocessing_config.json")

with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# --------------------------------------------------
# PASS 2: Transform all splits
# --------------------------------------------------

feature_names = (
    ["size_scaled", "time_sin", "time_cos",
     "subject_present", "object_present", "object2_present"]
    + [f"event_type_{i}" for i in range(len(event_vocab))]
    + [f"operation_{i}" for i in range(len(operation_vocab))]
)

manifest = []

print("\nPASS 2: Transforming all splits")

for split in ["train", "validation", "test"]:

    source_files = sorted(
        glob.glob(os.path.join(SPLIT_DIR, split, "*.parquet"))
    )

    output_folder = os.path.join(FEATURE_DIR, split)
    split_rows = 0

    for file_no, source_path in enumerate(source_files, 1):

        source_name = os.path.basename(source_path)
        output_path = os.path.join(
            output_folder,
            source_name.replace(".parquet", "_features.parquet")
        )

        pf = pq.ParquetFile(source_path)
        writer = None
        file_rows = 0

        for batch in pf.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "timestamp_ns",
                "event_type",
                "operation",
                "size",
                "subject_present",
                "object_present",
                "object2_present"
            ]
        ):
            d = batch.to_pydict()
            n = batch.num_rows

            # Numeric size feature
            raw_size = np.asarray(
                [0 if x is None else max(float(x), 0.0)
                 for x in d["size"]],
                dtype=np.float64
            )

            size_scaled = (
                np.log1p(raw_size) - size_mean
            ) / size_std

            # UTC time-of-day features
            timestamps = np.asarray(
                [0 if x is None else x for x in d["timestamp_ns"]],
                dtype=np.int64
            )

            seconds_in_day = (
                (timestamps // 1_000_000_000) % 86400
            ).astype(np.float64)

            angle = 2.0 * np.pi * seconds_in_day / DAY_SECONDS
            time_sin = np.sin(angle)
            time_cos = np.cos(angle)

            # Binary presence features
            subject = np.asarray(
                [0 if x is None else int(x)
                 for x in d["subject_present"]],
                dtype=np.float32
            )
            obj = np.asarray(
                [0 if x is None else int(x)
                 for x in d["object_present"]],
                dtype=np.float32
            )
            obj2 = np.asarray(
                [0 if x is None else int(x)
                 for x in d["object2_present"]],
                dtype=np.float32
            )

            # One-hot categorical features
            event_matrix = np.zeros(
                (n, len(event_vocab)), dtype=np.float32
            )
            operation_matrix = np.zeros(
                (n, len(operation_vocab)), dtype=np.float32
            )

            for i, value in enumerate(d["event_type"]):
                idx = event_map.get(value)
                if idx is not None:
                    event_matrix[i, idx] = 1.0

            for i, value in enumerate(d["operation"]):
                idx = operation_map.get(value)
                if idx is not None:
                    operation_matrix[i, idx] = 1.0

            arrays = [
                pa.array(size_scaled.astype(np.float32)),
                pa.array(time_sin.astype(np.float32)),
                pa.array(time_cos.astype(np.float32)),
                pa.array(subject),
                pa.array(obj),
                pa.array(obj2)
            ]

            arrays.extend(
                pa.array(event_matrix[:, i])
                for i in range(event_matrix.shape[1])
            )
            arrays.extend(
                pa.array(operation_matrix[:, i])
                for i in range(operation_matrix.shape[1])
            )

            table = pa.Table.from_arrays(
                arrays,
                names=feature_names
            )

            # Keep timestamp for traceability, not as model input.
            table = table.append_column(
                "timestamp_ns",
                pa.array(timestamps)
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    table.schema,
                    compression="zstd"
                )

            writer.write_table(table)
            file_rows += n

        if writer is not None:
            writer.close()

        split_rows += file_rows

        if file_no % 20 == 0:
            print(
                f"{split.upper()}: {file_no}/{len(source_files)} files | "
                f"{split_rows:,} rows"
            )

        manifest.append({
            "split": split,
            "source_file": source_name,
            "output_file": os.path.basename(output_path),
            "rows": file_rows
        })

    print(f"{split.upper()} COMPLETE: {split_rows:,} rows")

# --------------------------------------------------
# Save output manifest
# --------------------------------------------------

import csv

manifest_path = os.path.join(FEATURE_DIR, "feature_manifest.csv")

with open(manifest_path, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=manifest[0].keys()
    )
    writer.writeheader()
    writer.writerows(manifest)

print("\nFEATURE ENGINEERING COMPLETE")
print("Feature count:", len(feature_names))
print("Features:", feature_names)
print("Configuration:", config_path)
print("Manifest:", manifest_path)
print("Output:", FEATURE_DIR)


# In[ ]:





# In[ ]:


import os
import glob
import zipfile
from tqdm.auto import tqdm

BASE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/dataset_splits"

OUTPUT = "/content"

splits = {
    "train": "THEIA_TRAIN.zip",
    "validation": "THEIA_VALIDATION.zip",
    "test": "THEIA_TEST.zip"
}

for split, zip_name in splits.items():
    folder = os.path.join(BASE, split)
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))

    zip_path = os.path.join(OUTPUT, zip_name)

    print(f"\nCreating {zip_name}")
    print("Parquet files:", len(files))

    with zipfile.ZipFile(
        zip_path, "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True
    ) as zf:
        for file in tqdm(files):
            zf.write(file, arcname=os.path.basename(file))

    print(
        "ZIP size:",
        round(os.path.getsize(zip_path) / (1024**3), 2),
        "GB"
    )

print("\nALL THREE ZIP FILES CREATED!")


# In[ ]:


import os
import shutil
from google.colab import files

SOURCE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/dataset_splits"
ZIP_BASE = "/content/THEIA_COMPLETE_DATASET_SPLIT"

print("Creating complete dataset ZIP...")

zip_path = shutil.make_archive(
    ZIP_BASE,
    "zip",
    root_dir="/content/drive/MyDrive/THEIA_SAE_PROCESSED",
    base_dir="dataset_splits"
)

print("ZIP created:", zip_path)
print("ZIP size:", round(os.path.getsize(zip_path) / (1024**3), 2), "GB")

files.download(zip_path)


# In[1]:


get_ipython().system('pip install -U huggingface_hub')


# In[2]:


from huggingface_hub import notebook_login
notebook_login()


# In[3]:


get_ipython().system('hf repos delete-files --help')


# In[4]:


from huggingface_hub import HfApi

api = HfApi()

repo_id = "ISMAIL1216/APT-Detection-Qwen-IsolationForest"

# Delete all contents of both folders
api.delete_folder(
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="CADET"
)

api.delete_folder(
    repo_id=repo_id,
    repo_type="dataset",
    path_in_repo="THEIA"
)

print("Deletion requests completed.")


# In[5]:


get_ipython().system('pip install -U huggingface_hub')


# In[7]:


from huggingface_hub import notebook_login

notebook_login()


# In[8]:


from huggingface_hub import HfApi
import os

api = HfApi()

REPO_ID = "ISMAIL1216/APT-Detection-Qwen-IsolationForest"

LOCAL_FOLDER = (
    "/content/drive/MyDrive/"
    "THEIA_SAE_PROCESSED/dataset_splits"
)

assert os.path.isdir(LOCAL_FOLDER)

api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo="THEIA",
    commit_message="Upload complete THEIA train validation test Parquet splits"
)

print("THEIA upload completed!")


# In[9]:


from google.colab import drive

drive.mount("/content/drive", force_remount=True)


# In[10]:


import os
import glob

SOURCE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/dataset_splits"

for split in ["train", "validation", "test"]:
    folder = os.path.join(SOURCE, split)
    files = glob.glob(os.path.join(folder, "*.parquet"))

    total_size = sum(os.path.getsize(f) for f in files)

    print(f"\n{split.upper()}")
    print("Files:", len(files))
    print("Size:", round(total_size / (1024**3), 2), "GB")

print("\nDrive connection verified.")


# In[11]:


import shutil

total, used, free = shutil.disk_usage("/content")

print("Available Colab storage:", round(free / (1024**3), 2), "GB")


# In[15]:


import os
import shutil

SOURCE = "/content/drive/MyDrive/THEIA_SAE_PROCESSED/dataset_splits"
LOCAL = "/content/THEIA_UPLOAD"

if os.path.exists(LOCAL):
    raise RuntimeError("THEIA_UPLOAD already exists. Inspect it before copying.")

print("Copying dataset to Colab local storage...")

shutil.copytree(SOURCE, LOCAL)

print("\nCOPY COMPLETED")

for split in ["train", "validation", "test"]:
    folder = os.path.join(LOCAL, split)
    files = [f for f in os.listdir(folder) if f.endswith(".parquet")]
    print(split.upper(), ":", len(files), "Parquet files")

print("Manifest exists:", os.path.exists(
    os.path.join(LOCAL, "split_manifest.csv")
))


# In[16]:


import os
import glob

LOCAL = "/content/THEIA_UPLOAD"

for split in ["train", "validation", "test"]:
    folder = os.path.join(LOCAL, split)
    files = glob.glob(os.path.join(folder, "*.parquet")) if os.path.isdir(folder) else []

    print(f"{split.upper()}: {len(files)} files")

print(
    "Manifest exists:",
    os.path.exists(os.path.join(LOCAL, "split_manifest.csv"))
)


# In[18]:


get_ipython().system('pip install -U huggingface_hub')


# In[20]:


import huggingface_hub
import os

print("Version:", huggingface_hub.__version__)
print("Upload folder exists:", os.path.exists("/content/THEIA_UPLOAD"))
print("Train shards:", len(os.listdir("/content/THEIA_UPLOAD/train")))
print("Validation shards:", len(os.listdir("/content/THEIA_UPLOAD/validation")))
print("Test shards:", len(os.listdir("/content/THEIA_UPLOAD/test")))


# In[21]:


from huggingface_hub import notebook_login
notebook_login()


# In[23]:


from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="/content/THEIA_UPLOAD",
    repo_id="ISMAIL1216/APT-Detection-Qwen-IsolationForest",
    repo_type="dataset",
    path_in_repo="THEIA",
    commit_message="Upload THEIA train validation and test Parquet splits"
)

print("THEIA upload completed!")


# In[24]:


from huggingface_hub import notebook_login
notebook_login()


# In[ ]:




