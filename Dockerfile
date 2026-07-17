FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV WISDOM_DB_PATH=/data/wisdom.db
ENV APP_FILE=annotator_app.py
VOLUME /data
EXPOSE 8501
CMD streamlit run $APP_FILE --server.headless true --server.port 8501 --server.address 0.0.0.0
