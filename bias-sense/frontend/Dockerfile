FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Instala solo deps de UI
COPY requirements-ui.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements-ui.txt

# Copia la app completa
COPY . .

EXPOSE 8501

# Arranca Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

