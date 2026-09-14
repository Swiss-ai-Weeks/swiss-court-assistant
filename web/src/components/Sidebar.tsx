import type { ConversationSummary } from "../api";
import { relativeTime } from "../format";

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  open: boolean;
  onNew: () => void;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function Sidebar({ conversations, activeId, open, onNew, onSelect, onDelete }: Props) {
  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Conversation history">
      <div className="sidebar-head">
        <button className="btn-outline" onClick={onNew}>
          + New conversation
        </button>
      </div>
      <div className="sidebar-label">History</div>
      {conversations.length === 0 ? (
        <p className="history-empty">No conversations yet.</p>
      ) : (
        <ul className="history">
          {conversations.map((c) => (
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
