FROM python:3.9-slim

WORKDIR /app

# Install dependencies one by one
RUN pip install --upgrade pip && \
    pip install flask==2.3.3 && \
    pip install python-dotenv==1.0.0 && \
    pip install numpy==1.24.3 && \
    pip install langchain-core==0.2.6 && \
    pip install langchain && \
    pip install langchain-groq && \
    pip install langchain-chroma && \
    pip install langchain-google-genai && \
    pip install langchain-community && \
    pip install chromadb && \
    pip install transformers==4.35.2 && \
    pip install sentence-transformers && \
    pip install google-generativeai

# Copy application files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]