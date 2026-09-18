export default function Logo({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <polygon points="16,4 12,8 20,8" fill="currentColor" />
      <rect x="15" y="8" width="2" height="13" fill="currentColor" />
      <line x1="6" y1="8" x2="26" y2="8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <line x1="6" y1="8" x2="6" y2="15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <line x1="26" y1="8" x2="26" y2="15" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M2 15 Q6 20 10 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M22 15 Q26 20 30 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <rect x="11" y="21" width="10" height="8" rx="1.2" fill="#D52B1E" />
      <rect x="15" y="22.3" width="2" height="5.4" fill="#fff" />
      <rect x="12.8" y="24" width="6.4" height="2" fill="#fff" />
    </svg>
  );
}
