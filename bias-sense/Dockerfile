# Dockerfile.nb
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# 1) Instala las dependencias de la API Naive-Bayes
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt

# 2) Copia el código de la API
COPY api ./api
COPY bias_sense ./bias_sense

# 3) Expón el puerto que usará Cloud Run
EXPOSE 8080

# 4) Comando de arranque: Uvicorn sobre el endpoint de Naive-Bayes
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]
