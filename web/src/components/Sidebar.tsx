import { relativeTime } from "../format";

/** Both lists this app keeps — conversations and matters — are the same shape. */
export interface Entry {
  id: string;
  title: string;
  updatedAt: string;
}

interface Props {
  items: Entry[];
  label: string;
  newLabel: string;
  emptyLabel: string;
  activeId: string | null;
  open: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({ items, label, newLabel, emptyLabel, activeId, open, onNew, onSelect,
  onDelete }: Props) {
  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label={label}>
      <div className="sidebar-head">
        <button className="btn-outline" onClick={onNew}>
          + {newLabel}
        </button>
      </div>
      <div className="sidebar-label">{label}</div>
      {items.length === 0 ? (
        <p className="history-empty">{emptyLabel}</p>
      ) : (
        <ul className="history">
          {items.map((c) => (
            <li key={c.id} className={`history-item${c.id === activeId ? " active" : ""}`}>
              <button className="open" onClick={() => onSelect(c.id)} aria-current={c.id === activeId}>
                <span className="title">{c.title}</span>
                <span className="when">{relativeTime(c.updatedAt)}</span>
              </button>
              <button className="del" onClick={() => onDelete(c.id)} aria-label={`Delete “${c.title}”`} title="Delete">
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </aside>
  );
}
