import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // '/api' prefix is optional — we call full URL in App.jsx
      // This proxy helps avoid CORS when calling localhost:8000 directly
    },
  },
})
