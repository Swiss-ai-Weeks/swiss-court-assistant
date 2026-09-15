const EXAMPLES = [
  { lang: "English", q: "When can a tenant challenge a landlord's termination of the lease?" },
  { lang: "Français", q: "Un licenciement notifié pendant la grossesse peut-il être qualifié d'abusif ?" },
  { lang: "Deutsch", q: "Wann ist eine Kündigung des Mietverhältnisses treuwidrig?" },
  { lang: "Italiano", q: "Quando è abusivo il licenziamento di una lavoratrice?" },
];

export default function Welcome({ onAsk }: { onAsk: (q: string) => void }) {
  return (
    <div className="welcome">
      <p className="eyebrow">Swiss case law · Federal and cantonal courts</p>
      <h1>Ask a question. Get answers you can trace to the decision.</h1>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e.q} className="example" onClick={() => onAsk(e.q)}>
            <span className="corner-square" />
            <span className="lang">{e.lang}</span>
            <span className="q">{e.q}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
