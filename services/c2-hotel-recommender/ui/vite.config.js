import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built page goes to ../src/ui/app and the Python service serves it at http://localhost:8002/app.
// "npm run dev" opens a live-reloading copy on http://localhost:5173 that sends API calls to the service on 8002.
const api = "http://localhost:8002";
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: { outDir: "../src/ui/app", emptyOutDir: true, chunkSizeWarningLimit: 800 },
  server: { proxy: { "/recommendations": api, "/hotels": api, "/health": api } },
});
