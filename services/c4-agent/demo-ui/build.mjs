// Builds index.html (and dist/index.html): ONE self-contained file (React + app + data inlined).
// Open it by double-clicking - no server, no internet, no GPU needed.
//   node build.mjs          (needs esbuild + react + react-dom; see README)
import { build } from "esbuild";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";

const res = await build({
  entryPoints: ["src/main.jsx"], bundle: true, minify: true, write: false, format: "iife",
  jsx: "automatic", target: "es2018", loader: { ".json": "json" }, outdir: "out", define: { "process.env.NODE_ENV": '"production"' },
});
const js = res.outputFiles.find((f) => f.path.endsWith(".js")).text;
const css = res.outputFiles.find((f) => f.path.endsWith(".css")).text;
const html = readFileSync("src/template.html", "utf8")
  .replace("<!--STYLE-->", `<style>${css}</style>`)
  .replace("<!--SCRIPT-->", `<script>${js.replace(/<\/script/g, "<\\/script")}</script>`);
mkdirSync("dist", { recursive: true });
writeFileSync("dist/index.html", html);
writeFileSync("index.html", html); // same app at the folder root, so either file opens it
console.log(`index.html and dist/index.html written (${Math.round(html.length / 1024)} KB)`);
