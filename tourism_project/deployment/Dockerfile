FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py model.pkl feature_columns.pkl ./

RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
