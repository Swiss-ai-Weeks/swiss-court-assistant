import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` proxies the API to the FastAPI server (python -m swiss_court_assistant.server).
export default defineConfig({
  // relative asset URLs, so the build also works behind a path prefix (e.g. /coder/proxy/8090/)
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: { "/api": { target: "http://localhost:8090", changeOrigin: true } },
  },
});
