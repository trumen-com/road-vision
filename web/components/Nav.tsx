"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const PAGES = [
  ["/", "Approach"],
  ["/results/", "Results"],
  ["/eda/", "EDA"],
  ["/demo/", "Live demo"],
  ["/report/", "Report"],
  ["/team/", "Team"],
];

export default function Nav() {
  const path = usePathname();
  const [theme, setTheme] = useState<string | null>(null);
  useEffect(() => {
    try {
      const t = localStorage.getItem("theme");
      if (t) { document.documentElement.dataset.theme = t; setTheme(t); }
    } catch { /* storage unavailable */ }
  }, []);
  const toggle = () => {
    const dark = theme ? theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    setTheme(next);
    try { localStorage.setItem("theme", next); } catch { /* ignore */ }
  };
  return (
    <nav className="nav">
      <div className="wrap">
        <Link href="/" className="brand">Tru<span>men</span></Link>
        {PAGES.map(([href, label]) => (
          <Link key={href} href={href} className={`link ${path === href || (href !== "/" && path?.startsWith(href)) ? "active" : ""}`}>
            {label}
          </Link>
        ))}
        <span className="spacer" />
        <button className="theme-btn" onClick={toggle} aria-label="Toggle colour theme">◐</button>
      </div>
    </nav>
  );
}
