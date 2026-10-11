import { defineConfig } from "vite";
// BASE_PATH is "/road-vision/" on GitHub Pages; "/" locally.
export default defineConfig({
  base: process.env.BASE_PATH ?? "/",
  build: { target: "es2022", chunkSizeWarningLimit: 4000 },
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
