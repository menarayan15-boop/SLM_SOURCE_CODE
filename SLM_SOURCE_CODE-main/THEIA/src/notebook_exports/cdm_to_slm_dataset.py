#!/usr/bin/env python
# coding: utf-8

# # CDM Security Telemetry → SLM Prompt/Completion Dataset
# **Input:** CDM18 provenance JSON (CADETS / THEIA style, JSON-lines or JSON array) + vendor XDR knowledge `.md` files
# **Output:** `prompt`/`completion` JSONL (+ train/val split) and a `.jsonl.zst` compressed copy.
# 
# **Design goal – avoid false positives and false negatives.** Every example is *grounded*: the prompt carries the evidence (host, process table, event log), and the completion may only use facts from it.
# * **False-positive guards:** hard-negative "did X happen?" questions (absent entity, absent process, wrong operation on a real object), questions about fields the schema does not contain (hashes, CVEs, domains, attribution), calibrated triage language (an indicator is *not* a verdict), and an automatic grounding verifier that drops any completion quoting an entity that is not in its prompt or the knowledge base.
# * **False-negative guards:** positive existence/count/list questions with exact event references, indicator surfacing (privilege change, execution, public-IP traffic, sensitive paths, fan-out), and process lineage / network classification computed programmatically.
# * **Prompt diversity:** many paraphrases per task, 3 evidence render styles, optional persona/rule preambles, and light surface noise (case, punctuation, politeness).

# In[ ]:


# ---- Cell 1: setup & configuration ----
import os, re, io, sys, json, glob, gzip, lzma, math, random, hashlib, ipaddress, subprocess, collections, datetime

try:
    import zstandard as zstd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "zstandard"])
    import zstandard as zstd

IN_COLAB = "google.colab" in sys.modules

CONFIG = dict(
    DATA_DIR      = os.environ.get("DATA_DIR", "/content"),      # where the .json and .md files live
    OUTPUT_DIR    = os.environ.get("OUTPUT_DIR", "/content"),
    JSON_GLOB     = "*.json",                                      # CDM input files (JSON-lines or array)
    MD_GLOB       = "*.md",                                        # vendor knowledge files
    OUTPUT_NAME   = "slm_dataset.jsonl",
    WINDOW        = 20,        # events per evidence window (keep small enough for your SLM context)
    STRIDE        = 20,        # == WINDOW → no overlap; use WINDOW//2 for 50% overlap (more, correlated samples)
    MIN_WINDOW    = 6,         # drop trailing windows smaller than this
    QA_PER_WINDOW = 14,        # max log-QA examples per window
    KB_REPEAT     = 2,         # paraphrase passes over the knowledge-base Q&A
    SEED          = 42,
    VAL_FRACTION  = 0.05,      # split by window so one window never leaks into both splits
    INCLUDE_META  = False,     # True → adds task/answerable/source fields (handy for analysis; drop for training)
    ZSTD_LEVEL    = 19,
)
rng_global = random.Random(CONFIG["SEED"])

# Colab convenience: prompt for uploads if nothing is present
if IN_COLAB and not glob.glob(os.path.join(CONFIG["DATA_DIR"], CONFIG["JSON_GLOB"])):
    from google.colab import files
    print("Upload your CDM .json files and the vendor .md files:")
    files.upload()

JSON_FILES = sorted(glob.glob(os.path.join(CONFIG["DATA_DIR"], CONFIG["JSON_GLOB"])))
MD_FILES   = sorted(glob.glob(os.path.join(CONFIG["DATA_DIR"], CONFIG["MD_GLOB"])))
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
print(f"{len(JSON_FILES)} JSON file(s), {len(MD_FILES)} markdown file(s)")
for p in JSON_FILES: print("  json:", os.path.basename(p), f"{os.path.getsize(p)/1e6:.2f} MB")


# In[ ]:


# ---- Cell 2: tolerant streaming JSON reader + CDM helpers ----
def iter_json_records(path):
    """Yield dict records from JSON-lines (CRLF ok), a JSON array, or concatenated pretty-printed objects.
    Streams line-by-line on the fast path so very large files do not need to fit in memory."""
    bad, ok = 0, 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line: break
            if line.strip(): first = line.strip(); f.seek(pos); break
        if not first: return
        if first.startswith("["):                       # one big JSON array
            data = json.load(f)
            for r in data:
                if isinstance(r, dict): yield r
            return
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                r = json.loads(s)
                ok += 1
                if isinstance(r, dict): yield r
            except json.JSONDecodeError:
                bad += 1
                if ok == 0 and bad > 20: break          # not JSON-lines → fall back below
    if ok == 0 and bad > 0:                              # concatenated / pretty-printed objects
        text = open(path, "r", encoding="utf-8", errors="replace").read()
        dec, i, n = json.JSONDecoder(), 0, len(text)
        while i < n:
            while i < n and text[i] in " \t\r\n,": i += 1
            if i >= n: break
            obj, i = dec.raw_decode(text, i)
            if isinstance(obj, dict): yield obj
    elif bad:
        print(f"  note: skipped {bad} unparsable line(s) in {os.path.basename(path)}")

def U(x):
    """Unwrap Avro union wrappers: {'string': v}, {'long': v}, {'com.bbn.tc.schema.avro.cdm18.UUID': v} ..."""
    while isinstance(x, dict) and len(x) == 1:
        k = next(iter(x))
        if k in ("string", "long", "int", "array", "boolean", "double") or k.startswith("com.bbn"):
            x = x[k]
        else:
            break
    return x

def P(props):
    """CDM properties are {'map': {...}} (or null)."""
    if isinstance(props, dict):
        m = props.get("map", props)
        return m if isinstance(m, dict) else {}
    return {}

def cdm_kind(rec):
    d = rec.get("datum") or {}
    if not d: return None, None, rec.get("source")
    k = next(iter(d))
    return k.split(".")[-1], d[k], rec.get("source")

def ts_iso(ns):
    try:
        ns = int(ns)
        return datetime.datetime.fromtimestamp(ns // 10**9, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{(ns % 10**9)//1000:06d}Z"
    except Exception:
        return None

def short(s, n=110):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"

def nz(x): return "n/a" if x is None else x

def basename(p): return str(p).rstrip("/").split("/")[-1] if p else ""


# In[ ]:


# ---- Cell 3: parse the vendor markdown files into a structured knowledge base ----
def norm_ws(s): return re.sub(r"\s+", " ", s).strip()

def parse_kb_file(path):
    raw = open(path, encoding="utf-8", errors="replace").read()
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    t = norm_ws(raw)
    hdr = [p.strip() for p in lines[0].split("|")]
    kb = dict(file=os.path.basename(path), vendor="", product="", feature="", role="", description="", concepts=[],
              themes=[], evidence_quality="", impl_boundary="", worked="", urls=[])
    kb["vendor"]  = hdr[1] if len(hdr) > 1 else ""
    kb["product"] = hdr[2] if len(hdr) > 2 else ""
    kb["feature"] = lines[1] if len(lines) > 1 else os.path.basename(path)
    m = re.search(r"Vendor:\s*([^|]+?)\s*\|", t);               kb["vendor"] = m.group(1).strip() if m else kb["vendor"]
    m = re.search(r"Capability role:\s*(.+?)\s+Full-page", t);  kb["role"] = m.group(1).strip() if m else ""
    m = re.search(r"Original supplied feature description:\s*(.+?)\s+Current official vendor sources", t)
    if not m: m = re.search(r"supplied corpus describes it as follows:\s*(.+?)\s+That source description", t)
    kb["description"] = m.group(1).strip() if m else ""
    m = re.search(r"recurring technical concepts are\s+(.+?)\.\s+These concepts", t)
    if m:
        kb["concepts"] = [re.sub(r"^and\s+", "", c.strip()) for c in m.group(1).split(",") if c.strip()]
    names = re.findall(r"Theme (\d+)\s+\S\s+(.+?)\s+Feature-specific foundation", t)
    purposes = re.findall(r"The purpose of this page is\s+(.+?)\.\s+The important distinction", t)
    kb["themes"] = [dict(n=int(n), title=ti.strip(), purpose=pu.strip()) for (n, ti), pu in zip(names, purposes)]
    m = re.search(r"Evidence quality and limitations\s+(.+?)\s+Analytical and ML reasoning", t);          kb["evidence_quality"] = m.group(1).strip() if m else ""
    m = re.search(r"Implementation and configuration boundary\s+(.+?)\s+Feature-specific technical synthesis", t); kb["impl_boundary"] = m.group(1).strip() if m else ""
    m = re.search(r"Worked training scenario for .+? imagine that\s+(.+?)\s+Training checkpoint", t);    kb["worked"] = ("Imagine that " + m.group(1).strip()) if m else ""
    seen = []
    for u in re.findall(r"https?://[^\s)\]>]+", raw):
        u = u.rstrip(".,;")
        if u not in seen: seen.append(u)
    kb["urls"] = seen
    def clean(x):
        if isinstance(x, str): return norm_ws(x.replace("\ufffd", ""))
        if isinstance(x, list): return [clean(i) for i in x]
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        return x
    return {k: (clean(v) if k != "urls" else v) for k, v in kb.items()}

KB = []
for p in MD_FILES:
    try:
        k = parse_kb_file(p)
        if k["description"] and k["concepts"]: KB.append(k)
        else: print("  skipped (no feature description/concepts found):", os.path.basename(p))
    except Exception as e:
        print("  KB parse error", os.path.basename(p), e)
print(f"Parsed {len(KB)} knowledge-base features from markdown")
for k in KB[:3]:
    print(f"  - {k['vendor']} | {k['feature']} | role={k['role']} | concepts={k['concepts']} | themes={len(k['themes'])} | urls={len(k['urls'])}")
KB_TEXT = "\n".join(json.dumps(k, ensure_ascii=False) for k in KB)   # used by the grounding verifier


# In[ ]:


# ---- Cell 4: pass 1 – index entities (hosts, principals, processes, files, flows) per CDM file ----
def index_cdm_file(path):
    E = dict(host={}, prin={}, sub={}, file={}, net={}, mem={}, other={}, procname={}, objpath={}, source=None)
    vocab = dict(proc=collections.Counter(), obj=collections.Counter(), etype=collections.Counter(), remote=collections.Counter())
    n = 0
    for rec in iter_json_records(path):
        kind, v, src = cdm_kind(rec)
        if kind is None: continue
        n += 1; E["source"] = E["source"] or src
        if kind == "Host":
            ips = [a for i in (v.get("interfaces") or []) for a in (i.get("ipAddresses") or []) if "%" not in a and ":" not in a]
            E["host"][v["uuid"]] = dict(name=v.get("hostName") or "", os=short(v.get("osDetails") or "", 90), ips=ips, type=v.get("hostType"))
        elif kind == "Principal":
            E["prin"][v["uuid"]] = dict(uid=v.get("userId"), user=U(v.get("username")), groups=sorted(set(v.get("groupIds") or [])))
        elif kind == "Subject":
            pr = P(v.get("properties")); cmd = U(v.get("cmdLine"))
            if cmd in ("N/A", ""): cmd = None
            path_ = pr.get("path")
            name = basename(path_) if path_ else (basename(str(cmd).split()[0]) if cmd else None)
            E["sub"][v["uuid"]] = dict(pid=v.get("cid"), ppid=pr.get("ppid"), parent=U(v.get("parentSubject")), prin=v.get("localPrincipal"),
                                       cmd=short(cmd, 150) if cmd else None, name=name, start=v.get("startTimestampNanos"))
        elif kind == "FileObject":
            E["file"][v["uuid"]] = dict(path=P((v.get("baseObject") or {}).get("properties")).get("filename"), ftype=v.get("type"))
        elif kind == "NetFlowObject":
            E["net"][v["uuid"]] = dict(l=v.get("localAddress"), lp=v.get("localPort"), r=v.get("remoteAddress"), rp=v.get("remotePort"))
        elif kind == "MemoryObject":
            E["mem"][v["uuid"]] = dict(addr=v.get("memoryAddress"), size=U(v.get("size")))
        elif kind in ("SrcSinkObject", "UnnamedPipeObject", "RegistryKeyObject"):
            E["other"][v["uuid"]] = dict(kind=kind, type=v.get("type"))
        elif kind == "Event":
            pr = P(v.get("properties")); su = U(v.get("subject")); ou = U(v.get("predicateObject")); op = U(v.get("predicateObjectPath"))
            if pr.get("exec") and su not in E["procname"]: E["procname"][su] = pr["exec"]
            if op and op != "<unknown>" and ou and ou not in E["objpath"]: E["objpath"][ou] = op
            vocab["etype"][v.get("type")] += 1
    # vocab of names/objects for hard negatives
    for u, s in E["sub"].items():
        nm = E["procname"].get(u) or s["name"]
        if nm: vocab["proc"][nm] += 1
    for u, f in E["file"].items():
        pth = E["objpath"].get(u) or f["path"]
        if pth and pth != "<unknown>": vocab["obj"][pth] += 1
    for u, p in E["objpath"].items(): vocab["obj"][p] += 1
    for u, nf in E["net"].items():
        if nf["r"]: vocab["remote"][f"{nf['r']}:{nf['rp']}"] += 1
    E["n_records"] = n
    return E, vocab


# In[ ]:


# ---- Cell 5: pass 2 – resolve events into normalized dicts and slice into windows ----
SENSITIVE = [
    (r"^/etc/(passwd|shadow|master\.passwd|pwd\.db|spwd\.db|sudoers|group)$", "account / credential database"),
    (r"^/dev/k?mem$",                                                       "kernel memory device"),
    (r"(^|/)\.ssh/",                                                        "SSH key material"),
    (r"^/etc/ssl/|(^|/)cert\.pem$",                                         "TLS trust store"),
    (r"^/etc/(hosts|resolv\.conf|nsswitch\.conf)$",                         "name-resolution configuration"),
    (r"^/proc/\d+/",                                                        "process pseudo-file"),
]
HIGH_SENS = {"account / credential database", "kernel memory device", "SSH key material"}
WELL_KNOWN_PORTS = {22: "SSH", 25: "SMTP", 53: "DNS", 80: "HTTP", 123: "NTP", 143: "IMAP", 443: "HTTPS", 445: "SMB", 3306: "MySQL", 5432: "PostgreSQL"}
NET_EVENTS = {"EVENT_CONNECT", "EVENT_ACCEPT", "EVENT_BIND", "EVENT_SENDTO", "EVENT_RECVFROM", "EVENT_SENDMSG", "EVENT_RECVMSG", "EVENT_LISTEN"}
SPAWN_EVENTS = {"EVENT_FORK", "EVENT_CLONE"}
DESTRUCTIVE = {"EVENT_UNLINK", "EVENT_TRUNCATE", "EVENT_RENAME", "EVENT_LINK"}
PROC_TAMPER = {"EVENT_MODIFY_PROCESS", "EVENT_SIGNAL", "EVENT_MPROTECT", "EVENT_LOADLIBRARY"}

def sens_cat(path):
    if not path: return None
    for rx, cat in SENSITIVE:
        if re.search(rx, path): return cat
    return None

def classify_ip(addr):
    try:
        ip = ipaddress.ip_address(str(addr))
    except ValueError:
        return "not a valid IP literal (unresolved placeholder)"
    if ip.is_loopback: return "loopback"
    if ip.is_link_local: return "link-local"
    if ip.is_private: return "private (RFC 1918)"
    if ip.is_multicast: return "multicast"
    if ip.is_unspecified: return "unspecified"
    return "public (globally routable)"

def fmt_flow(nf):
    return f"{nf['l']}:{nf['lp']}->{nf['r']}:{nf['rp']}"

def normalize_event(v, E):
    pr = P(v.get("properties")); su = U(v.get("subject")); ou = U(v.get("predicateObject")); op = U(v.get("predicateObjectPath"))
    sub = E["sub"].get(su, {})
    pname = E["procname"].get(su) or sub.get("name") or pr.get("exec") or "unknown-process"
    prin = E["prin"].get(sub.get("prin"), {})
    ev = dict(ns=v.get("timestampNanos") or 0, type=v.get("type"), syscall=U(v.get("name")), proc=pname, pid=sub.get("pid"),
              ppid=sub.get("ppid") or pr.get("ppid"), uid=prin.get("uid"), user=prin.get("user"),
              size=U(v.get("size")), ret=pr.get("return_value", pr.get("rc")), obj=None, kind="none", netflow=None, target_pid=None, obj_name=None)
    if ou in E["sub"]:
        t = E["sub"][ou]; ev.update(kind="process", target_pid=t["pid"], obj_name=E["procname"].get(ou) or t["name"] or "unknown-process")
        ev["obj"] = f"pid {t['pid']} {ev['obj_name']}"
    elif ou in E["net"]:
        nf = E["net"][ou]; ev.update(kind="netflow", netflow=nf, obj=fmt_flow(nf))
    elif ou in E["mem"]:
        m = E["mem"][ou]; ev.update(kind="memory", obj=f"memory@{m['addr']}")
    elif ou in E["file"] or op:
        pth = op or E["objpath"].get(ou) or (E["file"].get(ou) or {}).get("path")
        ev.update(kind="file", obj=short(pth, 120) if pth else "<unnamed file object>")
    elif ou in E["other"]:
        ev.update(kind="ipc", obj=E["other"][ou]["kind"] + (":" + str(E["other"][ou]["type"]) if E["other"][ou]["type"] else ""))
    elif ou:
        ev.update(kind="unresolved", obj="<unresolved object>")
    ev["iso"] = ts_iso(ev["ns"])
    ev["sens"] = sens_cat(ev["obj"]) if ev["kind"] == "file" else None
    ev["unknown_obj"] = ev["obj"] in ("<unknown>", "<unresolved object>", "<unnamed file object>")
    return ev

def iter_windows(path, E):
    buf, widx = [], 0
    W, S = CONFIG["WINDOW"], CONFIG["STRIDE"]
    for rec in iter_json_records(path):
        kind, v, src = cdm_kind(rec)
        if kind != "Event": continue
        buf.append(normalize_event(v, E))
        if len(buf) >= W:
            win = sorted(buf[:W], key=lambda e: e["ns"])
            yield widx, win; widx += 1
            buf = buf[S:]
    if len(buf) >= CONFIG["MIN_WINDOW"] and (widx == 0 or S < W or len(buf) < W):
        yield widx, sorted(buf, key=lambda e: e["ns"])


# In[ ]:


# ---- Cell 6: window facts, evidence rendering, triage rules ----
def window_facts(win, E):
    F = dict(n=len(win), t0=win[0]["iso"], t1=win[-1]["iso"], span=(win[-1]["ns"] - win[0]["ns"]) / 1e9)
    F["types"] = collections.Counter(e["type"] for e in win)
    F["procs"] = collections.OrderedDict()                # (name,pid) → count
    for e in win: F["procs"][(e["proc"], e["pid"])] = F["procs"].get((e["proc"], e["pid"]), 0) + 1
    F["pairs"] = collections.defaultdict(list)           # (proc, type, obj) → [event numbers]
    F["proc_obj"] = collections.defaultdict(lambda: collections.defaultdict(list))   # proc → obj → [(n,type)]
    F["spawns"], F["execs"], F["privs"], F["flows"], F["sens"], F["destr"], F["tamper"], F["unk"] = [], [], [], [], [], [], [], []
    for i, e in enumerate(win, 1):
        e["n"] = i
        F["pairs"][(e["proc"], e["type"], e["obj"])].append(i)
        F["proc_obj"][e["proc"]][e["obj"]].append((i, e["type"]))
        if e["type"] in SPAWN_EVENTS and e["kind"] == "process": F["spawns"].append(e)
        if e["type"] == "EVENT_EXECUTE": F["execs"].append(e)
        if e["type"] == "EVENT_CHANGE_PRINCIPAL": F["privs"].append(e)
        if e["kind"] == "netflow" and e["type"] in NET_EVENTS: F["flows"].append(e)
        if e["sens"]: F["sens"].append(e)
        if e["type"] in DESTRUCTIVE: F["destr"].append(e)
        if e["type"] in PROC_TAMPER: F["tamper"].append(e)
        if e["unknown_obj"]: F["unk"].append(e)
    F["objs"] = collections.Counter(e["obj"] for e in win if e["obj"] and e["kind"] in ("file", "ipc", "memory"))
    F["flow_set"] = collections.OrderedDict()
    for e in F["flows"]:
        F["flow_set"].setdefault(e["obj"], []).append(e["n"])
    F["public_flows"] = [e for e in F["flows"] if classify_ip(e["netflow"]["r"]).startswith("public")]
    F["has_size"] = any(e["size"] is not None for e in win)
    # triage indicators
    ind, score = [], 0
    if F["privs"]:        ind.append(("privilege/identity change", [e["n"] for e in F["privs"]], f"`{F['privs'][0]['type']}` by `{F['privs'][0]['proc']}`")); score += 2
    if F["public_flows"]: ind.append(("traffic to a publicly routable address", [e["n"] for e in F["public_flows"]], f"`{F['public_flows'][0]['obj']}`")); score += 2
    hs = [e for e in F["sens"] if e["sens"] in HIGH_SENS]
    if hs:                ind.append(("access to a sensitive-category path", [e["n"] for e in hs], f"`{hs[0]['obj']}` ({hs[0]['sens']})")); score += 2
    if F["destr"]:        ind.append(("file deletion/truncation/rename", [e["n"] for e in F["destr"]], f"`{F['destr'][0]['type']}` on `{F['destr'][0]['obj']}`")); score += 1
    if F["execs"]:        ind.append(("program execution", [e["n"] for e in F["execs"]], f"`{F['execs'][0]['obj']}`")); score += 1
    if F["tamper"]:       ind.append(("process-state modification or signalling", [e["n"] for e in F["tamper"]], f"`{F['tamper'][0]['type']}`")); score += 1
    if len(F["spawns"]) >= 4: ind.append(("process fan-out (≥4 fork/clone)", [e["n"] for e in F["spawns"]], f"{len(F['spawns'])} spawn events")); score += 1
    F["indicators"] = ind
    F["tier"] = "no rule-based indicator" if score == 0 else ("low-priority review" if score <= 2 else "elevated review")
    return F

def proc_table(win, E):
    rows, seen = [], set()
    for e in win:
        key = (e["pid"], e["proc"])
        if key in seen: continue
        seen.add(key)
        user = f"{e['user']}(uid {e['uid']})" if e["user"] else (f"uid {e['uid']}" if e["uid"] is not None else "unresolved")
        rows.append(dict(pid=e["pid"], name=e["proc"], ppid=e["ppid"], user=user))
    return rows

def render_evidence(win, E, host, src, style):
    hn = host.get("name") or "hostname not recorded"
    head = [f"Source: {src}", f"Host: {hn}" + (f" | OS: {host['os']}" if host.get("os") else "") + (f" | IPs: {', '.join(host['ips'])}" if host.get("ips") else ""),
            f"Window: {len(win)} events, {win[0]['iso']} to {win[-1]['iso']} (UTC)"]
    pt = proc_table(win, E)
    if style == "json":
        head.append("Processes: " + json.dumps(pt, ensure_ascii=False))
        lines = [json.dumps(dict(n=e["n"], t=e["iso"], proc=e["proc"], pid=e["pid"], type=e["type"], obj=e["obj"], size=e["size"], ret=e["ret"]), ensure_ascii=False) for e in win]
    else:
        head.append("Processes:\n" + "\n".join(f"  pid={r['pid']} name={r['name']} ppid={r['ppid']} user={r['user']}" for r in pt))
        if style == "kv":
            lines = [f"event={e['n']} time={e['iso']} proc={e['proc']} pid={e['pid']} type={e['type']} object={e['obj']} size={nz(e['size'])} ret={nz(e['ret'])}" for e in win]
        else:
            lines = [f"#{e['n']} {e['iso']} {e['proc']}(pid {e['pid']}) {e['type']} -> {e['obj']} size={nz(e['size'])} ret={nz(e['ret'])}" for e in win]
    return "\n".join(head) + "\nEvents:\n" + "\n".join(lines)


# In[ ]:


# ---- Cell 7: prompt-diversity machinery ----
PERSONAS = ["You are a security operations analyst assistant.", "You are an endpoint-telemetry analyst helping with incident triage.",
            "You are a threat-hunting copilot that reasons only from supplied evidence.", "You are a SOC tier-2 investigator's assistant.",
            "You are a digital-forensics assistant reviewing system provenance logs.", "Act as a careful incident responder."]
RULES = ["Use only the evidence below; if it does not show something, say it is not present instead of guessing.",
         "Stay strictly within the records provided and state clearly when something cannot be determined from them.",
         "Do not invent events, entities or intent. Separate what is observed from what is inferred.",
         "Answer from the log excerpt only. Cite event numbers where possible.",
         "Be precise and conservative: an observation is not automatically malicious."]

IMPERATIVE = re.compile(r"^(List|Count|Describe|Show|Summarize|Give|Provide|Check|Explain|Report|Trace|Identify|Flag|Enumerate|Define|Briefly|Walk|Which|What)\b")
def perturb(q, r):
    x = r.random()
    if x < 0.08 and "`" not in q: q = q.lower()      # never lowercase quoted entities
    elif x < 0.14: q = q.rstrip("?.")
    elif x < 0.24 and re.match(r"^(List|Count|Describe|Show|Summarize|Give|Provide|Check|Explain|Report|Trace|Identify|Flag|Enumerate|Define|Walk)\b", q): q = "Please " + q[0].lower() + q[1:]
    elif x < 0.30: q = q + " Be concise."
    elif x < 0.36: q = q + " Cite event numbers."
    return q

KB_RULES = ["Answer from the vendor knowledge corpus only; do not invent product claims.", "Ground the answer in the supplied corpus and say so when it does not cover the point.",
            "Be factual and conservative about product capabilities; capability, configuration and outcome are different propositions."]
def make_prompt(evidence, q, r, kb=False):
    parts = []
    if r.random() < 0.7: parts.append(r.choice(PERSONAS) + " " + r.choice(KB_RULES if kb else RULES))
    if evidence: parts.append(evidence)
    parts.append(r.choice(["Question: ", "", "Q: ", "Task: "]) + perturb(q, r))
    return "\n\n".join(parts)

def evrefs(nums, cap=6):
    """Compact event references: [1,2,3,4,9] -> '#1-#4, #9'."""
    nums = sorted(set(nums)); runs, start = [], None
    for i, n in enumerate(nums):
        if start is None: start = prev = n
        elif n == prev + 1: prev = n
        else: runs.append((start, prev)); start = prev = n
    if start is not None: runs.append((start, prev))
    txt = [f"#{a}" if a == b else (f"#{a}, #{b}" if b == a + 1 else f"#{a}-#{b}") for a, b in runs]
    return ", ".join(txt[:cap]) + (f" (+{len(txt)-cap} more ranges)" if len(txt) > cap else "")

def plural(n, w): return f"{n} {w}" + ("" if n == 1 else "s")


# In[ ]:


# ---- Cell 8: log-QA generators (each returns a list of (task, question, answer, answerable)) ----
def tick(x): return f"`{x}`"

def g_summary(F, win, V, r):
    top = ", ".join(f"{tick(t)}×{c}" for t, c in F["types"].most_common(4))
    procs = ", ".join(f"{tick(n)} (pid {p})" for (n, p), c in list(F["procs"].items())[:5])
    ind = "; ".join(f"{name} ({evrefs(nums,3)})" for name, nums, _ in F["indicators"]) or "none from the rule set"
    a = (f"The window holds {F['n']} events spanning {F['span']:.3f} s ({F['t0']} to {F['t1']}). Most frequent event types: {top}. "
         f"Active processes: {procs}. Indicators worth a look: {ind}. This is a description of observed telemetry only; the window by itself does not establish malicious intent.")
    Q = ["Summarize what happened in this window of events.", "Give me a short activity summary of these records.", "What is going on in this log excerpt?",
         "Describe the observed behavior without over-interpreting it.", "Provide a triage-style overview of these events."]
    return [("summary", r.choice(Q), a, True)]

def g_count(F, win, V, r):
    out = []
    present = list(F["types"]); absent = [t for t in V["etype"] if t not in F["types"]]
    Q = ["How many {t} events are in this window?", "Count the {t} events.", "Were there any {t} events here, and if so how many?", "What is the number of {t} records in the excerpt?"]
    if present:
        t = r.choice(present); nums = [e["n"] for e in win if e["type"] == t]
        out.append(("count_present", r.choice(Q).format(t=tick(t)), f"There {'is' if len(nums)==1 else 'are'} {len(nums)} {tick(t)} event{'' if len(nums)==1 else 's'} ({evrefs(nums)}) out of {F['n']}.", True))
    if absent and r.random() < 0.6:
        t = r.choice(absent)
        out.append(("count_absent", r.choice(Q).format(t=tick(t)), f"None. There are 0 {tick(t)} events among the {F['n']} events provided, so nothing in this window supports that activity.", True))
    return out

def g_exists(F, win, V, r):
    out = []
    keys = list(F["pairs"])
    if not keys: return out
    p, t, o = r.choice(keys); nums = F["pairs"][(p, t, o)]
    Q = ["Did {p} perform {t} on {o}?", "Is there an event where {p} did {t} against {o}?", "Check whether {p} issued {t} for {o}.", "Can you confirm that process {p} ran {t} on {o}?"]
    out.append(("exists_positive", r.choice(Q).format(p=tick(p), t=tick(t), o=tick(o)),
                f"Yes. {tick(p)} performed {tick(t)} on {tick(o)} in {evrefs(nums)}; the first occurrence is at {win[nums[0]-1]['iso']}.", True))
    # near-miss: same process+object, different operation never observed
    ops_for_pair = {tt for (pp, tt, oo) in F["pairs"] if pp == p and oo == o}
    alt = [x for x in V["etype"] if x not in ops_for_pair]
    if alt:
        t2 = r.choice(alt)
        seen = ", ".join(f"{tick(tt)} ({evrefs(F['pairs'][(p, tt, o)],3)})" for tt in sorted(ops_for_pair))
        out.append(("exists_near_miss", r.choice(Q).format(p=tick(p), t=tick(t2), o=tick(o)),
                    f"No. There is no {tick(t2)} by {tick(p)} on {tick(o)} in this window. What the log does show for that pair: {seen}.", True))
    # absent object (hard negative from dataset vocabulary)
    neg = r.choice(["object", "process", "both"])
    cand = [x for x in V["obj"] if x not in F["objs"] and x not in {e["obj"] for e in win}]
    if not cand: cand = ["/etc/shadow", "/tmp/.hidden/payload.bin", "/root/.ssh/authorized_keys"]
    o2 = r.choice(cand)
    if neg in ("object", "both"): out.append(("exists_negative_object", r.choice(Q).format(p=tick(p), t=tick(t), o=tick(o2)),
                f"No. {tick(o2)} does not appear anywhere in these {F['n']} events, so there is no evidence that {tick(p)} touched it here. Absence from this window does not prove it never happened elsewhere.", True))
    # absent process
    pc = [x for x in V["proc"] if x not in {n for (n, _) in F["procs"]}] or ["powershell.exe", "nc", "mimikatz"]
    p2 = r.choice(pc)
    if neg in ("process", "both"): out.append(("exists_negative_process", r.choice(["Did {p} run in this window?", "Is {p} present among the active processes?", "Did the process {p} generate any events here?"]).format(p=tick(p2)),
                f"No. {tick(p2)} is not in the process table and has no events in this window. Active processes are: " + ", ".join(tick(n) for (n, _) in F["procs"]) + ".", True))
    return out

def g_lists(F, win, V, r):
    out = []
    pl = "; ".join(f"{tick(n)} (pid {p}, {plural(c,'event')})" for (n, p), c in F["procs"].items())
    out.append(("list_processes", r.choice(["List the distinct processes active in this window.", "Which processes generated these events?", "Enumerate the processes and how many events each produced."]),
                f"{len(F['procs'])} distinct process{'' if len(F['procs'])==1 else 'es'}: {pl}.", True))
    procs = list(F["proc_obj"])
    if procs:
        p = r.choice(procs); objs = F["proc_obj"][p]
        txt = "; ".join(f"{tick(o)} via " + ", ".join(sorted({tick(t) for _, t in v})) + f" ({evrefs([n for n,_ in v],3)})" for o, v in list(objs.items())[:8])
        out.append(("objects_by_process", r.choice(["Which objects did {p} interact with?", "List what {p} touched in this window.", "What files, sockets or processes did {p} access?"]).format(p=tick(p)),
                    f"{tick(p)} interacted with {len(objs)} object{'' if len(objs)==1 else 's'}: {txt}.", True))
    if F["objs"]:
        top = ", ".join(f"{tick(o)} ({c})" for o, c in F["objs"].most_common(5))
        out.append(("top_objects", r.choice(["Which files or objects were accessed most?", "Show the most frequently touched objects.", "What are the top accessed paths?"]), f"Most-touched objects: {top}.", True))
    return out

def g_network(F, win, V, r):
    Q = ["Is there any network activity in this window?", "Did any process communicate over the network?", "List network endpoints and classify them.", "Did any host contact an external (public) address?"]
    if not F["flow_set"]:
        return [("network_none", r.choice(Q), "No network flow events appear in this window: none of the events reference a socket object, so there is no evidence of network communication here.", True)]
    parts = []
    for fl, nums in F["flow_set"].items():
        nf = next(e["netflow"] for e in F["flows"] if e["obj"] == fl)
        pn = WELL_KNOWN_PORTS.get(nf["rp"]); port = f", remote port {nf['rp']} is commonly associated with {pn}" if pn else ""
        evt = sorted({e["type"] for e in F["flows"] if e["obj"] == fl})
        parts.append(f"{tick(fl)} ({evrefs(nums,3)}; {', '.join(tick(x) for x in evt)}) - remote is {classify_ip(nf['r'])}{port}")
    pub = "At least one flow targets a publicly routable address; verify it against asset inventory and threat intelligence before drawing conclusions." if F["public_flows"] else "No flow in this window targets a publicly routable address."
    return [("network_flows", r.choice(Q), f"{len(F['flow_set'])} distinct flow{'' if len(F['flow_set'])==1 else 's'}: " + "; ".join(parts) + f". {pub}", True)]

def g_lineage(F, win, V, r):
    out = []
    pt = {row["pid"]: row for row in proc_table(win, None) if row["pid"] is not None}
    if F["spawns"]:
        e = r.choice(F["spawns"])
        out.append(("lineage_spawn", r.choice(["Which process spawned pid {c}?", "Who is the parent of pid {c} according to these events?", "Trace where pid {c} came from."]).format(c=e["target_pid"]),
                    f"{tick(e['proc'])} (pid {e['pid']}) created pid {e['target_pid']} ({tick(e['obj_name'])}) via {tick(e['type'])} in {evrefs([e['n']])} at {e['iso']}.", True))
        listing = "; ".join(f"pid {s['pid']} {tick(s['proc'])} → pid {s['target_pid']} {tick(s['obj_name'])} ({evrefs([s['n']])})" for s in F["spawns"][:8])
        out.append(("lineage_list", r.choice(["List every parent→child process creation in this window.", "Show the fork/clone relationships observed."]),
                    f"{plural(len(F['spawns']),'process-creation event')}: {listing}.", True))
    else:
        out.append(("lineage_none", r.choice(["Were any child processes created in this window?", "Show the fork/clone relationships observed."]),
                    "No fork or clone events occur in this window, so no parent→child creation is observed here. Parent pids in the process table come from process records, not from creation events in this excerpt.", True))
    ghost = (max([p for p in pt if isinstance(p, int)] or [1000]) + 7919)
    out.append(("lineage_unknown", f"Which process created pid {ghost}?", f"I cannot tell. pid {ghost} does not appear in the process table or in any event of this window, so there is no evidence about its parent.", False))
    withp = [row for row in pt.values() if row["ppid"] not in (None, "")]
    if withp:
        row = r.choice(withp); par = pt.get(int(row["ppid"])) if str(row["ppid"]).isdigit() else None
        a = (f"The process record lists ppid {row['ppid']} for pid {row['pid']} ({tick(row['name'])}); that parent " + (f"is {tick(par['name'])} in this window." if par else "is not in this window's process table, so its name cannot be given from this evidence."))
        out.append(("lineage_ppid", r.choice(["What is the recorded parent pid of pid {c}, and can you name it?", "Report ppid for pid {c}."]).format(c=row["pid"]), a, True))
    return out

def g_sensitive(F, win, V, r):
    Q = ["Were any sensitive files or devices accessed?", "Did anything touch credential stores, kernel memory or trust stores?", "Flag access to security-relevant paths in this window."]
    if not F["sens"]:
        return [("sensitive_none", r.choice(Q), "No path in this window matches the sensitive-path watchlist (credential databases, kernel memory devices, SSH material, trust stores, name-resolution config). That is a statement about this excerpt only.", True)]
    by = collections.OrderedDict()
    for e in F["sens"]: by.setdefault((e["obj"], e["sens"]), []).append(e)
    txt = "; ".join(f"{tick(o)} ({cat}) by " + ", ".join(sorted({tick(x['proc']) for x in es})) + f" ({evrefs([x['n'] for x in es],3)})" for (o, cat), es in list(by.items())[:6])
    return [("sensitive_access", r.choice(Q), f"Yes, path(s) in watchlist categories were accessed: {txt}. Access alone is not evidence of malicious intent; judge it against the process's normal behavior and a baseline.", True)]

def g_triage(F, win, V, r):
    Q = ["Should this window be escalated?", "Is anything here malicious?", "Give a triage verdict and justify it.", "Are there indicators of compromise in these events?", "How suspicious is this activity?"]
    if not F["indicators"]:
        a = ("I cannot call this malicious. No rule-based indicator fires (no privilege change, execution, public-IP traffic, sensitive-path access, destructive operation, process tampering or fan-out). "
             "That is not a clean bill of health: a short window lacks baselines, command lines and wider context, so absence of indicators only means nothing stands out here.")
    else:
        ev = "; ".join(f"{name} ({evrefs(nums,3)}: {detail})" for name, nums, detail in F["indicators"])
        a = (f"Triage tier: {F['tier']}. Indicators observed: {ev}. These are review pointers, not proof of malice - none of them is labeled ground truth. "
             "To raise or lower confidence, check process command lines, the owning user, a behavioral baseline for these processes, and reputation of any remote endpoint.")
    return [("triage", r.choice(Q), a, True)]

def g_time(F, win, V, r):
    out = []
    first, last = win[0], win[-1]
    out.append(("time_bounds", r.choice(["What are the first and last events and the time span?", "When does this activity start and end?"]),
                f"First: {evrefs([first['n']])} {tick(first['type'])} by {tick(first['proc'])} at {first['iso']}. Last: {evrefs([last['n']])} {tick(last['type'])} by {tick(last['proc'])} at {last['iso']}. Span: {F['span']:.3f} s across {F['n']} events.", True))
    k = r.randint(1, F["n"] - 1); a_, b_ = win[k - 1], win[k]
    out.append(("time_next", r.choice(["What happened immediately after event #{k}?", "Which event follows #{k}?"]).format(k=k),
                f"Event #{k+1} follows: {tick(b_['type'])} by {tick(b_['proc'])} on {tick(b_['obj'])} at {b_['iso']}.", True))
    out.append(("event_lookup_oob", f"Describe event #{F['n'] + r.randint(1, 9)}.", f"There is no such event: this window contains only {F['n']} events (#1-#{F['n']}).", False))
    e = r.choice(win)
    out.append(("event_lookup", r.choice(["Describe event #{k} in detail.", "What are the fields of event #{k}?"]).format(k=e["n"]),
                f"Event #{e['n']}: {tick(e['type'])} by {tick(e['proc'])} (pid {e['pid']}) on {tick(e['obj'])} at {e['iso']}; size={nz(e['size'])}, return={nz(e['ret'])}.", True))
    return out

def g_identity(F, win, V, r):
    rows = proc_table(win, None); out = []
    if rows:
        row = r.choice(rows)
        if row["user"] == "unresolved":
            a = f"The owning user of {tick(row['name'])} (pid {row['pid']}) is unresolved in this evidence, so I will not guess it."
        else:
            a = f"{tick(row['name'])} (pid {row['pid']}) runs as {row['user']}."
        out.append(("identity", r.choice(["Which user account does pid {c} run as?", "Who owns pid {c}?"]).format(c=row["pid"]), a, row["user"] != "unresolved"))
    chg = F["privs"]
    out.append(("priv_change", r.choice(["Was there any privilege or identity change?", "Did any process change its UID/GID?"]),
                (f"Yes: {plural(len(chg),'change-principal event')} ({evrefs([e['n'] for e in chg])}) by " + ", ".join(sorted({tick(e['proc']) for e in chg})) + ". Such changes are routine for some daemons and setuid tools; confirm the old and new identity before escalating.") if chg else
                "No. No change-principal event appears in this window, so no privilege or identity change is observed here.", True))
    return out

def g_unanswerable(F, win, V, r):
    """Questions about fields the CDM schema/window lacks. Prevents hallucinated hashes/CVEs/attribution."""
    bank = [("file hash", "What is the SHA-256 of the file touched by event #{k}?", "The records carry no file hash for that object, so I cannot provide one."),
            ("CVE", "Which CVE was exploited in this window?", "No vulnerability or CVE information is present in these records, so I cannot name one; naming a CVE would be speculation."),
            ("domain", "Which domain names were contacted?", "These records contain socket addresses only, with no DNS names, so I cannot say which domains were contacted."),
            ("attacker", "Who is the attacker behind this activity?", "Attribution cannot be made from this excerpt: it has no threat-actor labels or intelligence context, and no event is labeled malicious."),
            ("registry", "Which registry keys were modified?", "No registry events exist in this telemetry; the operating-system objects here are files, sockets, memory regions and processes."),
            ("payload", "What data was sent over the network?", "Payload content is not recorded; at most an event size is present, so the transferred content cannot be read from this evidence."),
            ("ground-truth", "Is this window part of the labeled attack?", "The window has no ground-truth label, so I cannot say whether it belongs to a labeled attack; I can only describe the observed behavior and indicators.")]
    items = r.sample(bank, 2); out = []
    for name, q, a in items:
        k = r.randint(1, F["n"])
        out.append(("unanswerable_" + name.replace(" ", "_"), q.format(k=k), a, False))
    return out

def g_evidence_quality(F, win, V, r):
    Q = ["Which events lack an identifiable object, and why does that matter?", "Are there evidence-quality gaps in these records?"]
    if F["unk"]:
        a = (f"{plural(len(F['unk']),'event')} ({evrefs([e['n'] for e in F['unk']])}) reference an object whose path or identity is unresolved (shown as {tick(F['unk'][0]['obj'])}). "
             "Treat conclusions about those objects as low confidence, because missing entity resolution can merge or hide unrelated files.")
    else:
        a = "Every event in this window resolves to a named object (file path, socket, process or memory region), so object identity is not a limiting factor here. Other limits remain: no command-line baseline and a very short time horizon."
    return [("evidence_quality", r.choice(Q), a, True)]

def g_kb_hybrid(F, win, V, r, KB):
    """Which of a capability's concepts does THIS window's telemetry actually supply? Present AND absent."""
    if not KB: return []
    k = r.choice(KB); lines = []; sat = 0
    for c in k["concepts"]:
        cl = c.lower()
        if re.search(r"\bprocess\b|lineage|ancestry|parent|process chains|root entity", cl):
            ok = bool(F["spawns"] or F["execs"]); txt = (f"{len(F['spawns'])} spawn and {len(F['execs'])} execute event(s), {len(F['procs'])} process(es)" if ok else "no spawn/execute events; only the process table")
        elif re.search(r"file", cl):
            n = sum(1 for e in win if e["kind"] == "file"); ok = n > 0; txt = f"{n} file event(s)" if ok else "no file events"
        elif re.search(r"network|socket|firewall", cl):
            ok = bool(F["flows"]); txt = f"{len(F['flows'])} flow event(s)" if ok else "no network flow events"
        elif re.search(r"system-call|privilege|kernel|memory", cl):
            n = sum(1 for e in win if e["syscall"]); ok = n > 0 or bool(F["privs"]); txt = f"{n} syscall-named event(s), {len(F['privs'])} privilege change(s)" if ok else "no syscall detail"
        elif re.search(r"command line", cl):
            ok = False; txt = "no command-line field in the events (the process table shows names and pids only)"
        elif re.search(r"time|sequence|temporal", cl):
            ok = True; txt = f"{F['n']} timestamped events over {F['span']:.3f} s"
        else:
            ok = False; txt = "no signal in these endpoint events maps to it (it depends on other data sources or on platform behavior, not on this window)"
        sat += ok; lines.append(f"{c}: {'supported' if ok else 'not supported'} - {txt}")
    q = r.choice(["Which of {f}'s core concepts ({v}) can this telemetry window feed, and which are missing?", "Could {f} from {v} be fully exercised with only this window? Check each concept: {cs}."]).format(f=tick(k["feature"]), v=k["vendor"], cs=", ".join(k["concepts"]))
    a = f"Concept-by-concept for {tick(k['feature'])}: " + "; ".join(lines) + f". {sat} of {len(k['concepts'])} concepts have supporting evidence here; the rest need additional telemetry."
    return [("kb_hybrid", q, a, True)]

LOG_GENERATORS = [g_summary, g_count, g_exists, g_lists, g_network, g_lineage, g_sensitive, g_triage, g_time, g_identity, g_unanswerable, g_evidence_quality]


# In[ ]:


# ---- Cell 9: knowledge-base Q&A from the markdown files (paraphrased, contrastive) ----
def strip_prefix(d): return re.sub(r"^The source corpus describes\s+", "", d).rstrip(".")

def kb_examples(KB, r):
    ex = []
    if not KB: return ex
    all_concepts = {c: k for k in KB for c in k["concepts"]}
    for k in KB:
        f, v, pr, d = k["feature"], k["vendor"], k["product"], strip_prefix(k["description"])
        ex.append(("kb_definition", r.choice([f"What is {f} in {pr}?", f"Explain {f} ({v}).", f"Define {tick(f)} and its purpose.", f"Briefly describe {f}."]),
                   f"According to the supplied corpus, {tick(f)} ({v}, {pr}) is described as {d}. Its primary role is {k['role']}.", True))
        ex.append(("kb_vendor", r.choice([f"Which vendor provides {f}?", f"Who makes {f}, and in which product line?"]), f"{tick(f)} belongs to {v}'s {pr} in the supplied corpus.", True))
        ex.append(("kb_role", r.choice([f"What is the primary capability role of {f}?", f"What category of capability is {f}?"]), f"Its primary role is {k['role']}.", True))
        ex.append(("kb_concepts", r.choice([f"Which technical concepts are central to {f}?", f"List the recurring concepts for {f}."]), f"The recurring concepts for {tick(f)} are: " + "; ".join(k["concepts"]) + ".", True))
        ex.append(("kb_reverse", r.choice(["Which feature is described as follows: {d}?", "Identify the capability: {d}."]).format(d=d),
                   f"That matches {tick(f)} from {v} ({pr}).", True))
        for th in r.sample(k["themes"], min(2, len(k["themes"]))):
            ex.append(("kb_theme", r.choice([f"What does the '{th['title']}' section cover for {f}?", f"In the {f} chapter, what is the scope of '{th['title']}'?"]),
                       f"For {tick(f)}, the section {tick(th['title'])} covers {th['purpose']}.", True))
        if k["evidence_quality"]:
            ex.append(("kb_evidence_quality", r.choice([f"What evidence-quality limitations apply to {f}?", f"How can evidence problems mislead {f}?"]), k["evidence_quality"], True))
        if k["impl_boundary"]:
            ex.append(("kb_boundary", r.choice([f"How should implementation detail for {f} be described, and what is the capability vs configuration distinction?", f"Is a supported capability the same as an enabled one for {f}?"]), k["impl_boundary"], True))
        if k["worked"]:
            ex.append(("kb_worked", r.choice([f"Walk through a worked evidence-to-response scenario for {f}.", f"Give a T1/T2/T3 reasoning example for {f}."]), k["worked"], True))
        if k["urls"]:
            ex.append(("kb_sources", r.choice([f"Which official sources were checked for {f}?", f"Where can I verify {f} against vendor documentation?"]),
                       "The corpus lists these vendor sources: " + "; ".join(k["urls"]) + ". Retain the retrieval date and URL for production training, because product documentation changes over time.", True))
        # contrastive: concept belongs to another feature → don't over-claim
        others = [c for c, ok in all_concepts.items() if ok is not k and c not in k["concepts"]]
        if others:
            c = r.choice(others); ok = all_concepts[c]
            ex.append(("kb_contrast_negative", r.choice([f"Does {f} center on {c}?", f"Is {c} one of the concepts of {f}?"]),
                       f"The supplied corpus does not list {tick(c)} among the concepts of {tick(f)}; it associates it with {tick(ok['feature'])} ({ok['vendor']}). "
                       "This only reflects what the corpus states; it does not prove the product lacks related functionality.", True))
        c = r.choice(k["concepts"])
        ex.append(("kb_contrast_positive", r.choice([f"Is {c} one of the concepts of {f}?", f"Does {f} involve {c}?"]), f"Yes. {tick(c)} is one of the recurring concepts the corpus lists for {tick(f)}.", True))
    # vendor roll-ups
    byv = collections.defaultdict(list)
    for k in KB: byv[k["vendor"]].append(k)
    for v, ks in byv.items():
        ex.append(("kb_vendor_features", r.choice([f"Which {v} features does the corpus cover?", f"List the {v} capabilities in the knowledge base."]),
                   f"The corpus covers {len(ks)} {v} feature(s): " + "; ".join(f"{tick(k['feature'])} ({k['role']})" for k in ks) + ".", True))
    # unanswerable KB probes (pricing, version, licensing) → refuse to invent
    k = r.choice(KB)
    for q in [f"What is the price of {k['feature']}?", f"Which software version introduced {k['feature']}?", f"Does the deployed tenant currently have {k['feature']} enabled?"]:
        ex.append(("kb_unanswerable", q, "The supplied corpus does not state that. It describes what the capability does, not pricing, release versions or whether a particular tenant has it enabled; capability and configuration are separate propositions, so I will not guess.", False))
    return ex


# In[ ]:


# ---- Cell 10: build the dataset (index → windows → QA → KB → verify → dedupe → split → write) ----
TICK_RE = re.compile(r"`([^`]+)`")
def grounded(prompt, completion, task):
    for tok in TICK_RE.findall(completion):
        if tok in prompt: continue
        if task.startswith("kb_") and tok in KB_TEXT: continue
        return False
    return True

examples, stats = [], collections.Counter()
for path in JSON_FILES:
    print("Indexing", os.path.basename(path))
    E, V = index_cdm_file(path)
    host = next(iter(E["host"].values()), {})
    src = E["source"] or os.path.basename(path)
    print(f"  {E['n_records']} records | {len(E['sub'])} processes | {len(E['file'])} file objs | {len(E['net'])} flows | source={src}")
    for widx, win in iter_windows(path, E):
        r = random.Random(f"{CONFIG['SEED']}|{os.path.basename(path)}|{widx}")
        F = window_facts(win, E)
        style = r.choice(["log", "kv", "json"])
        evidence = render_evidence(win, E, host, src, style)
        cands = []
        for g in LOG_GENERATORS: cands += [(g.__name__,) + x for x in g(F, win, V, r)]
        cands += [("g_kb_hybrid",) + x for x in g_kb_hybrid(F, win, V, r, KB)]
        must = {"g_summary", "g_triage", "g_exists", "g_network", "g_unanswerable", "g_count"}
        chosen = [c for c in cands if c[0] in must]
        rest = [c for c in cands if c[0] not in must]; r.shuffle(rest)
        chosen += rest[: max(0, CONFIG["QA_PER_WINDOW"] - len(chosen))]
        for gname, task, q, a, answerable in chosen:
            p = make_prompt(evidence, q, r)
            examples.append(dict(prompt=p, completion=a, task=task, answerable=answerable, source=os.path.basename(path), window=f"{os.path.basename(path)}#{widx}"))

for rep in range(CONFIG["KB_REPEAT"]):
    r = random.Random(f"{CONFIG['SEED']}|kb|{rep}")
    for task, q, a, answerable in kb_examples(KB, r):
        pr = make_prompt("", q, r, kb=True)
        examples.append(dict(prompt=pr, completion=a, task=task, answerable=answerable, source="knowledge-base", window=f"kb#{rep}"))

# verify grounding, drop conflicts / duplicates
kept, seen_p, seen_pc = [], {}, set()
for ex in examples:
    if not grounded(ex["prompt"], ex["completion"], ex["task"]): stats["dropped_ungrounded"] += 1; continue
    h = hashlib.sha1((ex["prompt"] + "\u0001" + ex["completion"]).encode()).hexdigest()
    if h in seen_pc: stats["dropped_duplicate"] += 1; continue
    seen_pc.add(h)
    if ex["prompt"] in seen_p and seen_p[ex["prompt"]] != ex["completion"]: stats["dropped_conflicting_prompt"] += 1; continue
    seen_p[ex["prompt"]] = ex["completion"]; kept.append(ex)
rng_global.shuffle(kept)
print(f"\nGenerated {len(examples)} → kept {len(kept)} | {dict(stats)}")

def is_val(ex):
    return int(hashlib.md5(ex["window"].encode()).hexdigest(), 16) % 10000 < CONFIG["VAL_FRACTION"] * 10000

def row(ex):
    d = dict(prompt=ex["prompt"], completion=ex["completion"])
    if CONFIG["INCLUDE_META"]: d.update(task=ex["task"], answerable=ex["answerable"], source=ex["source"])
    return json.dumps(d, ensure_ascii=False)

OUT = os.path.join(CONFIG["OUTPUT_DIR"], CONFIG["OUTPUT_NAME"])
TRAIN, VAL = OUT.replace(".jsonl", "_train.jsonl"), OUT.replace(".jsonl", "_val.jsonl")
with open(OUT, "w", encoding="utf-8") as fa, open(TRAIN, "w", encoding="utf-8") as ft, open(VAL, "w", encoding="utf-8") as fv:
    for ex in kept:
        line = row(ex) + "\n"; fa.write(line); (fv if is_val(ex) else ft).write(line)
for p in (OUT, TRAIN, VAL): print(f"wrote {p}  ({sum(1 for _ in open(p, encoding='utf-8'))} rows, {os.path.getsize(p)/1e6:.2f} MB)")


# In[ ]:


# ---- Cell 11: quality report - balance, diversity, grounding ----
by_task = collections.Counter(e["task"] for e in kept)
ans = collections.Counter(e["answerable"] for e in kept)
print("Rows:", len(kept), "| answerable:", ans[True], "| unanswerable/negative-evidence:", ans[False])
neg_tasks = [t for t in by_task if "negative" in t or "none" in t or "absent" in t or "unanswerable" in t or "near_miss" in t or t.endswith("_oob")]
print(f"Explicit negative / absence / unanswerable rows: {sum(by_task[t] for t in neg_tasks)} ({100*sum(by_task[t] for t in neg_tasks)/max(1,len(kept)):.1f}%)")
print("\nTask distribution:")
for t, c in by_task.most_common(): print(f"  {t:28s} {c}")
lens = sorted(len(e['prompt']) for e in kept)
print(f"\nPrompt chars: min={lens[0]} median={lens[len(lens)//2]} p95={lens[int(len(lens)*.95)]} max={lens[-1]}")
uq = len({e['prompt'].split('\n\n')[-1] for e in kept})
print("Distinct question strings:", uq)
print("\n--- 3 random examples ---")
for e in random.Random(7).sample(kept, 3):
    print(json.dumps(dict(prompt=e["prompt"][-900:], completion=e["completion"]), ensure_ascii=False, indent=1)[:1600]); print()


# In[ ]:


# ---- Cell 12: compress the JSONL (zstd, with gzip/xz fallback) and verify round-trip ----
def sha256_stream(fileobj, chunk=1 << 20):
    h = hashlib.sha256()
    while True:
        b = fileobj.read(chunk)
        if not b: break
        h.update(b)
    return h.hexdigest()

def compress_zst(src, dst=None, level=CONFIG["ZSTD_LEVEL"]):
    dst = dst or src + ".zst"
    cctx = zstd.ZstdCompressor(level=level, threads=-1, write_content_size=True)
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        cctx.copy_stream(fi, fo)                                    # streaming: constant memory even for GB files
    with open(src, "rb") as a, zstd.ZstdDecompressor().stream_reader(open(dst, "rb")) as b:
        ok = sha256_stream(a) == sha256_stream(b)                    # round-trip integrity check
    so, sc = os.path.getsize(src), os.path.getsize(dst)
    print(f"{os.path.basename(src)} → {os.path.basename(dst)}: {so/1e6:.2f} MB → {sc/1e6:.2f} MB (ratio {so/max(sc,1):.1f}x) | round-trip {'OK' if ok else 'FAILED'}")
    return dst

def compress_fallback(src):
    """If zstandard is unavailable: xz (best ratio) or gzip (fastest, universal)."""
    dst = src + ".xz"
    with open(src, "rb") as fi, lzma.open(dst, "wb", preset=6) as fo:
        while True:
            b = fi.read(1 << 20)
            if not b: break
            fo.write(b)
    print("fallback written:", dst); return dst

compressed = []
for p in (OUT, TRAIN, VAL):
    try: compressed.append(compress_zst(p))
    except Exception as e:
        print("zstd failed:", e); compressed.append(compress_fallback(p))

def read_jsonl_zst(path):
    """Stream-read a .jsonl.zst back into dicts (works with huge files)."""
    with open(path, "rb") as fh, zstd.ZstdDecompressor().stream_reader(fh) as sr:
        for line in io.TextIOWrapper(sr, encoding="utf-8"):
            if line.strip(): yield json.loads(line)

first = next(read_jsonl_zst(compressed[0]))
print("\nRead-back check - keys:", list(first), "| prompt starts:", first["prompt"][:90].replace("\n", " "), "...")
# Hugging Face: datasets.load_dataset("json", data_files=compressed[1]) reads .jsonl.zst directly.


# In[ ]:


# ---- Cell 13 (Colab): download results ----
if IN_COLAB:
    from google.colab import files
    for p in compressed: files.download(p)
else:
    print("Not in Colab. Files are in:", CONFIG["OUTPUT_DIR"])

