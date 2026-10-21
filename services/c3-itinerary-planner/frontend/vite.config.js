import { defineConfig } from 'vite';

export default defineConfig({
  esbuild: { jsx: 'automatic' },
  server: {
    host: '127.0.0.1',
    proxy: {
      '/api': {
        // IPv4 loopback avoids localhost resolving to ::1 on Windows.
        target: 'http://127.0.0.1:8003',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
        // Optimization can take longer than a normal web request.
        proxyTimeout: 300000,
        timeout: 300000,
      },
    },
  },
});
