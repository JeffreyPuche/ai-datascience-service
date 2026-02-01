import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import webbrowser
from threading import Timer

app = FastAPI(title="Credit UI Server")

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Definir la ruta de la carpeta static
static_dir = os.path.join(current_dir, "static")

# Servir archivos estáticos (CSS, JS, HTML) desde la carpeta static
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


def open_browser():
    webbrowser.open_new("http://127.0.0.1:3000")


if __name__ == "__main__":
    print("--- Servidor de Interfaz de Usuario Iniciado ---")
    print("Asegúrate de que la API principal esté corriendo en el puerto 8000")
    print("Abriendo Panel de Control en: http://127.0.0.1:3000")

    # Abrir el navegador después de 1.5 segundos
    Timer(1.5, open_browser).start()

    # Correr el servidor en el puerto 3000
    uvicorn.run(app, host="127.0.0.1", port=3000, log_level="info")

# 1. Correr la api con el siguiente comando:
# uv run uvicorn server.api:app --reload --port 8000
# 2. Correr el servidor de la ui con el siguiente comando:
# uv run uvicorn examples.client_web.main:app --reload --port 3000


# para matar procesos en un puero
# netstat -ano | findstr :8000
# taskkill /F /PID 12345

# Borrar la caché del navegador en el puerto si hay conflictos
