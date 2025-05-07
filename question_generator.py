import os
import numpy as np
from sentence_transformers import SentenceTransformer
from flask import request, jsonify
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Global variables for vector data and API key
chunks = None
embeddings = None
embed_model = None
# Use the Groq API key that's working for chat
GROQ_API_KEY = ""

def setup_dependencies():
    """Setup dependencies for question generation"""
    global embed_model
    
    # Load sentence transformer model only - no Google API
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {str(e)}")

# Load Files
def load_file(file_path):
    """Load a text file with proper error handling"""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return ""

# Chunking the transcript
def chunk_text(text, chunk_size=500):
    """Split text into chunks of specified size"""
    if not text:
        return ["Sample text for empty transcript"]
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Semantic search using NumPy
def semantic_search(query_text, chunks, embeddings, top_k=1):
    """Find the most relevant chunks for a query"""
    global embed_model
    
    # Encode the query
    query_embedding = embed_model.encode([query_text])[0]
    
    # Calculate L2 distances
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    
    # Get indices of top_k smallest distances
    top_indices = np.argsort(distances)[:top_k]
    
    # Get corresponding chunks
    retrieved_chunks = [chunks[i] for i in top_indices]
    
    return retrieved_chunks

# Question Generation - Using ONLY Groq, NOT Gemini
def generate_questions(retrieved_content, co_text, bloom_level):
    """Generate questions using Groq API"""
    prompt = f"""
    You are a Question Generator Agent.
    Course Outcome (CO): {co_text}
    Bloom's Taxonomy Level: {bloom_level}
    
    Based on the content below, generate multiple questions:
    - Two Objective Type Questions
    - Two Short Answer Type Questions
    
    Content:
    {retrieved_content}
    
    Only output the questions in the following format:
    Objective Questions:
    1. <question 1>
    2. <question 2>
    
    Short Answer Questions:
    1. <question 1>
    2. <question 2>
    """
    
    try:
        # Create Groq LLM instance
        llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=GROQ_API_KEY,
            temperature=0.5,
            max_tokens=2048
        )
        
        # Generate questions using Groq
        print("Generating questions using Groq API...")
        response = llm.invoke([HumanMessage(content=prompt)])
        output = response.content.strip()
        print("Questions generated successfully")
        return output
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating questions with Groq: {error_msg}")
        return f"Error generating questions: {error_msg}"

# Parse generated questions into structured format
def parse_questions(questions_text):
    """Parse the generated questions into a structured format"""
    objective_questions = []
    subjective_questions = []
    
    if "Objective Questions:" in questions_text and "Short Answer Questions:" in questions_text:
        parts = questions_text.split("Short Answer Questions:")
        obj_part = parts[0].replace("Objective Questions:", "").strip()
        subj_part = parts[1].strip()
        
        # Extract objective questions
        for line in obj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix (e.g., "1. ", "2. ")
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                objective_questions.append(question)
        
        # Extract subjective questions
        for line in subj_part.split("\n"):
            if line.strip() and any(c.isdigit() for c in line[:2]):
                question = line.strip()
                # Remove the number prefix
                if ". " in question[:3]:
                    question = question[question.find(". ")+2:]
                subjective_questions.append(question)
    
    return {"objective": objective_questions, "subjective": subjective_questions}

# Initialize vector database
def initialize_vector_db():
    """Initialize the vector database for question generation"""
    global chunks, embeddings, embed_model
    
    # Setup dependencies first
    setup_dependencies()
    
    try:
        print("Building question generator vector database... please wait")
        transcript = load_file("cleaned_transcript.txt")
        if transcript:
            chunks = chunk_text(transcript)
            embeddings = embed_model.encode(chunks)
            print(f"Question generator vector database built with {len(chunks)} chunks")
        else:
            print("Warning: transcript file empty or not found, using sample data")
            chunks = ["Sample content for initialization"]
            embeddings = embed_model.encode(chunks)
    except Exception as e:
        print(f"Error initializing question generator vector database: {str(e)}")
        chunks = ["Sample content for error case"]
        embeddings = embed_model.encode(chunks)

# API handler function
def generate_questions_api(request):
    """Handler function for the generate-questions endpoint"""
    global chunks, embeddings
    
    if chunks is None or embeddings is None:
        initialize_vector_db()
    
    # Get request data
    data = request.get_json()
    
    if not data or 'course_outcome' not in data or 'bloom_level' not in data:
        return jsonify({
            "error": "Missing required parameters. Please provide 'course_outcome' and 'bloom_level'."
        }), 400
    
    selected_co = data['course_outcome']
    selected_bloom = data['bloom_level']
    
    try:
        # Generate questions using Groq
        best_chunk = semantic_search(selected_co, chunks, embeddings, top_k=1)[0]
        questions_text = generate_questions(best_chunk, selected_co, selected_bloom)
        
        # Check for errors
        if questions_text.startswith("Error generating"):
            return jsonify({
                "error": questions_text
            }), 500
        
        # Parse the questions into the requested structure
        questions_dict = parse_questions(questions_text)
        
        # Return the generated questions
        return jsonify({
            "course_outcome": selected_co,
            "bloom_level": selected_bloom,
            "questions": questions_dict,
            "raw_text": questions_text
        })
    
    except Exception as e:
        error_msg = str(e)
        return jsonify({
            "error": error_msg
        }), 500