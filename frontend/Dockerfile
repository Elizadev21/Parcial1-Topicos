# 1) Compila la app React.
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# 2) Sirve los archivos estáticos con nginx y redirige /api al backend.
FROM nginx:1.27-alpine
# BACKEND_URL: dirección de la API (sin barra final). En docker compose es el
# servicio "api"; en la nube, la URL pública del backend.
ENV PORT=8080 \
    BACKEND_URL=http://api:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
