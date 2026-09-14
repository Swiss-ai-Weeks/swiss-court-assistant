const LANG: Record<string, string> = { de: "German", fr: "French", it: "Italian", rm: "Romansh", en: "English" };

export const langName = (code: string) => LANG[code] ?? code.toUpperCase();

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(+d) ? iso : d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" });
}

export function relativeTime(iso: string): string {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 7 * 86400) return `${Math.floor(s / 86400)} d ago`;
  return formatDate(iso);
}

/** "E. 4.1, 4.1.1" — the Erwägungen (reasoning paragraphs) a passage covers. */
export const erwLabel = (e: string[]) => (e.length ? `E. ${e.join(", ")}` : "");
