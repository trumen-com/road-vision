// Wraps app.html into a standalone index.html for static hosting (GitHub Pages). The Claude artifact publishes app.html as is.
import { readFileSync, writeFileSync } from "node:fs";
const body = readFileSync(new URL("./app.html", import.meta.url), "utf8");
const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style></head><body>\n${body}\n</body></html>\n`;
writeFileSync(new URL("./index.html", import.meta.url), html);
console.log("index.html", html.length, "bytes");
