import type { Metadata, Viewport } from "next";
import "./globals.css";
import Nav from "@/components/Nav";
import { LINKS } from "@/content/team";

export const metadata: Metadata = {
  title: "Trumen · Traffic Event Detection",
  description: "Traffic events and accident anticipation from a fixed road camera — WIUT Hackathon 2026, CV track.",
};

export const viewport: Viewport = { width: "device-width", initialScale: 1 };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="wrap">{children}</main>
        <footer>
          <div className="wrap">
            Trumen · WIUT Hackathon 2026, Computer Vision track ·{" "}
            <a href={LINKS.repo}>Repository</a> · <a href={LINKS.weights}>Weights</a> ·{" "}
            <a href={LINKS.predictions}>predictions_samples.json</a>
          </div>
        </footer>
      </body>
    </html>
  );
}
