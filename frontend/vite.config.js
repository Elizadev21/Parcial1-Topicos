import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vite'

// En desarrollo, la API y su documentación se redirigen a la API local
// (API_URL permite cambiar el puerto).
const destino = { target: process.env.API_URL || 'http://127.0.0.1:8000', changeOrigin: true }
const proxy = { '/api': destino, '/docs': destino, '/redoc': destino, '/openapi.json': destino }
export default defineConfig({ plugins: [tailwindcss()], server: { proxy }, preview: { proxy } })
