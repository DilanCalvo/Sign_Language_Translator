/**
 * Tiny localStorage preference helper. All persisted settings share the
 * "asl-web." namespace (the camera picker in pipeline.js established it).
 * Values are JSON-encoded; every access is wrapped so a blocked storage
 * (Safari private mode) degrades to in-memory defaults instead of throwing.
 */

const PREFIX = "asl-web.";

export function prefGet(key, fallback) {
  try {
    const raw = localStorage.getItem(PREFIX + key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export function prefSet(key, value) {
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(value));
  } catch {
    /* storage unavailable — setting lives for this session only */
  }
}
