/**
 * Conversation store + exportable recording — the web counterpart of
 * src/conversation_log.py (ConversationLog).
 *
 * One store serves two distinct concerns:
 *  - the CHAT PANEL renders every entry (it is the live communication surface,
 *    always on, across all three modes);
 *  - the EXPORT includes only entries added while recording was on (`recorded`
 *    flag) — the REC button is consent to keep a transcript, not a gate on
 *    talking.
 *
 * The singleton lives at module scope, so mode switches never touch it.
 *
 * Export format is byte-compatible with the desktop app's files (same
 * filenames, same TXT layout, same CSV columns) so downstream consumers can't
 * tell which app produced a log. Source labels stay the desktop's Spanish
 * constants ("Senas"/"Oyente") for that same compatibility; the UI displays
 * its own English labels.
 */

export const SOURCE_SIGN = "Senas";      // export label — desktop parity
export const SOURCE_HEARING = "Oyente";  // export label — desktop parity

class Conversation {
  constructor() {
    this.entries = [];    // { when: Date, source, text, recorded: bool }
    this.recording = false;
    this._listeners = [];
  }

  /** subscribe(fn(event, payload)) — events: "add" (entry), "clear", "recording". */
  subscribe(fn) {
    this._listeners.push(fn);
  }

  _emit(event, payload) {
    for (const fn of this._listeners) fn(event, payload);
  }

  addSign(text) { this._add(SOURCE_SIGN, text); }
  addHearing(text) { this._add(SOURCE_HEARING, text); }

  _add(source, text) {
    text = String(text ?? "").trim();
    if (!text) return;
    const entry = { when: new Date(), source, text, recorded: this.recording };
    this.entries.push(entry);
    this._emit("add", entry);
  }

  setRecording(on) {
    this.recording = !!on;
    this._emit("recording", this.recording);
  }

  recordedEntries() {
    return this.entries.filter((e) => e.recorded);
  }

  clear() {
    this.entries.length = 0;
    this._emit("clear");
  }
}

export const conversation = new Conversation();

// ---- Export formatting (pure, unit-tested by the gate page) --------------

const pad2 = (n) => String(n).padStart(2, "0");

/** "YYYYMMDD_HHMMSS" in LOCAL time — Python's datetime.now().strftime. */
export function formatStamp(d) {
  return `${d.getFullYear()}${pad2(d.getMonth() + 1)}${pad2(d.getDate())}` +
    `_${pad2(d.getHours())}${pad2(d.getMinutes())}${pad2(d.getSeconds())}`;
}

const clockTime = (d) => `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;

/** TXT layout copied line-for-line from ConversationLog.export. */
export function formatTxt(entries, stamp) {
  let out = `Conversation log - ${stamp}\n${"=".repeat(40)}\n\n`;
  for (const e of entries) {
    out += `[${clockTime(e.when)}] ${e.source}: ${e.text}\n`;
  }
  return out;
}

/** RFC-4180 minimal quoting — matches Python csv.writer's QUOTE_MINIMAL. */
function csvField(value) {
  const s = String(value);
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

/** CSV: timestamp,source,text with LOCAL ISO seconds (Python isoformat). */
export function formatCsv(entries) {
  let out = "timestamp,source,text\r\n";
  for (const e of entries) {
    const d = e.when;
    const iso = `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` +
      `T${clockTime(d)}`;
    out += `${csvField(iso)},${csvField(e.source)},${csvField(e.text)}\r\n`;
  }
  return out;
}

// ---- Download ------------------------------------------------------------

export function downloadBlob(filename, text, mime) {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Give the click a beat to grab the blob before revoking.
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

/**
 * Download the recorded transcript as conversation_{stamp}.txt + .csv.
 * Returns the number of exported entries (0 = nothing recorded, no files).
 * The second download is staggered — some browsers drop the second of two
 * same-tick downloads. TXT (the human-readable one) goes first.
 */
export function exportLog() {
  const entries = conversation.recordedEntries();
  if (entries.length === 0) return 0;
  const stamp = formatStamp(new Date());
  downloadBlob(`conversation_${stamp}.txt`, formatTxt(entries, stamp),
               "text/plain;charset=utf-8");
  setTimeout(() => {
    downloadBlob(`conversation_${stamp}.csv`, formatCsv(entries),
                 "text/csv;charset=utf-8");
  }, 150);
  return entries.length;
}
