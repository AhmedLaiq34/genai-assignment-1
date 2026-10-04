import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    // Downloaded verification images may be locked briefly on Windows.
    watch: { ignored: ['**/verification/**', '**/.npm-cache/**', '**/mock/**'] },
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
});
