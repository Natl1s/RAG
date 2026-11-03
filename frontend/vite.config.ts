import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Разрешаем dev-серверу читать общую базу знаний, лежащую уровнем выше.
    fs: { allow: ['..'] },
    proxy: {
      // Проксируем запросы к FastAPI-заглушке (режим "API").
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
