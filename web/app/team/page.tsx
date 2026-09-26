import { LINKS, TEAM } from "@/content/team";

export default function Team() {
  return (
    <>
      <h1>Team TruMinds</h1>
      <p className="lead">Three people, one submission. Who did what, and where to find our work.</p>
      <div className="grid cols-3">
        {TEAM.map((m) => (
          <div key={m.name} className="card">
            <h3>{m.name}</h3>
            <div className="pill" style={{ marginBottom: 10 }}>{m.role}</div>
            <ul style={{ paddingLeft: 18, margin: "0 0 10px" }}>{m.did.map((d) => <li key={d} style={{ color: "var(--text-2)" }}>{d}</li>)}</ul>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 14 }}>
              {m.github && <a href={m.github}>GitHub</a>}
              {m.linkedin && <a href={m.linkedin}>LinkedIn</a>}
              {m.portfolio && <a href={m.portfolio}>Portfolio</a>}
            </div>
            {m.proud && m.proud.length > 0 && (
              <>
                <h3 style={{ marginTop: 14, fontSize: 14 }}>Projects we are proud of</h3>
                {m.proud.map((p) => <p key={p.title} style={{ fontSize: 14 }}><b style={{ color: "var(--text)" }}>{p.url ? <a href={p.url}>{p.title}</a> : p.title}</b>: {p.text}</p>)}
              </>
            )}
          </div>
        ))}
      </div>
      <h2>Links</h2>
      <div className="card">
        <table className="tbl"><tbody>
          <tr><td>Repository</td><td><a href={LINKS.repo}>{LINKS.repo}</a></td></tr>
          <tr><td>Weights</td><td><a href={LINKS.weights}>{LINKS.weights}</a></td></tr>
          <tr><td>Predictions on the samples</td><td><a href={LINKS.predictions}>{LINKS.predictions}</a></td></tr>
        </tbody></table>
      </div>
    </>
  );
}
