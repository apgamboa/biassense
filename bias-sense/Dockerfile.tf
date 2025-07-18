# Dockerfile.tf
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# 1) Instala las mismas dependencias de la API
COPY requirements_api.txt ./
RUN pip install --no-cache-dir -r requirements_api.txt

# 2) Copia el código de la API Transformer
COPY api ./api
COPY bias_sense ./bias_sense

# 3) Expón el puerto que usará este servicio
EXPOSE 8081

# 4) Lanza Uvicorn apuntando al endpoint Transformer
CMD ["uvicorn", "api.api_transformer:app", "--host", "0.0.0.0", "--port", "8081"]

