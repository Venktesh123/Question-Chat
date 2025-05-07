from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Import modules for each functionality
from question_generator import (
    initialize_vector_db as qg_initialize_vector_db,
    generate_questions_api
)
from civi_bot import prepare_vectorstore, chat_api, get_status

# Load environment variables
load_dotenv()

# Check required API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Create Flask app
app = Flask(__name__)

# Initialize databases on startup
print("Initializing Question Generator database...")
qg_initialize_vector_db()

# Register question generator routes
@app.route('/', methods=['GET'])
def api_status():
    """Root endpoint for API status"""
    return jsonify({
        "status": "online",
        "message": "Merged API for Course Outcome Question Generator and CiviBot",
        "services": {
            "question_generator": {
                "status": "online",
                "endpoints": {
                    "/generate-questions": "POST - Generate questions based on course outcomes and Bloom's levels"
                }
            },
            "civi_bot": {
                "status": "online",
                "endpoints": {
                    "/api/chat": "POST - Chat with CiviBot about civil engineering topics",
                    "/api/status": "GET - Check CiviBot status"
                }
            }
        },
        "api_keys": {
            "google_api_key": "present" if GOOGLE_API_KEY else "missing",
            "groq_api_key": "present" if GROQ_API_KEY else "missing"
        }
    })

@app.route('/generate-questions', methods=['POST'])
def generate_questions_endpoint():
    """Endpoint for question generation based on course outcomes"""
    return generate_questions_api(request)

# Register CiviBot routes
@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Endpoint for chatting with CiviBot"""
    return chat_api(request)

@app.route('/api/status', methods=['GET'])
def status_endpoint():
    """Endpoint to check CiviBot status"""
    return get_status()

# Main entry point
if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the Flask app
    print(f"Starting merged API server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)