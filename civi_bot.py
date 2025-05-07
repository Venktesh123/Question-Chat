import os
import time
from flask import jsonify
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader

# Global variables - hardcoded for testing
GOOGLE_API_KEY = ""
GROQ_API_KEY = ""

# Define constants
DEFAULT_CONFIG = {
    "file_path": "cleaned_transcript.txt",
    "chunk_size": 5000,
    "chunk_overlap": 1000,
    "embedding_model": "models/embedding-001",
    "vectordb_dir": "vectordb",
    "collection_name": "chroma",
    "k": 2
}

# Bloom's Taxonomy definitions
BLOOMS_TAXONOMY = {
    "remember": {
        "description": "Recall facts and basic concepts",
        "verbs": ["define", "list", "name", "identify", "recall", "state", "what", "who", "when", "where"]
    },
    "understand": {
        "description": "Explain ideas or concepts",
        "verbs": ["explain", "describe", "interpret", "summarize", "discuss", "clarify", "how", "why"]
    },
    "apply": {
        "description": "Use information in new situations",
        "verbs": ["apply", "demonstrate", "calculate", "solve", "use", "illustrate", "show"]
    },
    "analyze": {
        "description": "Draw connections among ideas",
        "verbs": ["analyze", "compare", "contrast", "distinguish", "examine", "differentiate", "relationship"]
    },
    "evaluate": {
        "description": "Justify a stand or decision",
        "verbs": ["evaluate", "assess", "critique", "judge", "defend", "argue", "support", "recommend", "best"]
    },
    "create": {
        "description": "Produce new or original work",
        "verbs": ["create", "design", "develop", "propose", "construct", "formulate", "devise", "invent"]
    }
}

# Global variables to store conversation state
conversation_chains = {}
memories = {}
chat_histories = {}

def detect_cognitive_level(text):
    """Determine the cognitive level of a question based on Bloom's Taxonomy."""
    text_lower = text.lower()
    
    for level, info in reversed(BLOOMS_TAXONOMY.items()):
        if any(verb in text_lower.split() for verb in info["verbs"]):
            return level
    
    return "understand"

def detect_sentiment(text, chat_history=None):
    """Detect the sentiment of a question based on keywords."""
    confusion_keywords = ["confused", "not sure", "don't get", "difficult", "unclear", "what", "hard", 
                         "don't understand", "explain", "how does", "what is", "?"]
    frustration_keywords = ["frustrated", "annoying", "still don't get", "not making sense", 
                           "too difficult", "impossible", "giving up", "waste", "useless", "!"]
    curiosity_keywords = ["interesting", "cool", "awesome", "fascinating", "tell me more", 
                         "curious", "excited", "wonder", "how about"]
    
    text_lower = text.lower()
    
    if any(kw in text_lower for kw in frustration_keywords):
        return "frustrated"
    
    elif any(kw in text_lower for kw in confusion_keywords) or text_lower.count("?") > 1:
        return "confused"
    
    elif any(kw in text_lower for kw in curiosity_keywords):
        return "curious"
    
    elif chat_history and len(chat_history) > 2:
        last_user_msg = chat_history[-2].content.lower()
        
        if any(kw in last_user_msg for kw in confusion_keywords) and len(text_lower.split()) < 8:
            return "confused"
    
    return "neutral"

def generate_bloom_specific_prompt(cognitive_level, sentiment):
    """Generate a prompt template based on cognitive level and sentiment."""
    base_template = """
    You are CiviBot, a helpful and knowledgeable assistant specializing in civil engineering concepts. Your primary goal is to help students understand their lecture material by providing clear, accurate explanations about civil engineering topics.

    ## Your Knowledge Base
    - You have access to a repository of civil engineering lecture transcripts.
    - You can retrieve relevant information from these transcripts to answer questions.
    - If asked about something outside your knowledge base, acknowledge the limitations and offer to help with what you do know.

    ## User's Cognitive Level and Learning Needs
    The user's question has been analyzed and identified as belonging to the "{cognitive_level}" level of Bloom's Taxonomy.
    
    This means the user is asking for help with: {cognitive_description}
    
    Based on this cognitive level:
    """
    
    bloom_specific_instructions = {
        "remember": """
    - Focus on providing clear, factual information from the lecture notes
    - Define key terms precisely and concisely
    - List relevant information in an organized manner
    - Provide direct answers to factual questions
    - Include specific examples from lecture materials when relevant
    """,
        "understand": """
    - Explain concepts in your own words, avoiding technical jargon when possible
    - Provide analogies or real-world examples to illustrate concepts
    - Compare and contrast related ideas to enhance understanding
    - Rephrase complex ideas in simpler terms
    - Summarize key points from the lecture materials
    """,
        "apply": """
    - Demonstrate how concepts can be applied to solve problems
    - Provide step-by-step procedures for calculations or processes
    - Use real-world civil engineering scenarios to illustrate applications
    - Include worked examples that show how to apply formulas or principles
    - Suggest practice problems that reinforce application skills
    """,
        "analyze": """
    - Break down complex concepts into their constituent parts
    - Highlight relationships between different engineering principles
    - Compare and contrast different methodologies or approaches
    - Discuss cause-effect relationships in civil engineering contexts
    - Help the student see patterns or organizational principles in the material
    """,
        "evaluate": """
    - Present multiple perspectives or approaches to civil engineering problems
    - Discuss pros and cons of different methodologies
    - Help the student develop criteria for making engineering judgments
    - Encourage critical thinking about standard practices
    - Assess the validity of different claims or methods in context
    """,
        "create": """
    - Support innovative thinking and problem-solving
    - Provide frameworks for designing new solutions
    - Discuss how existing principles might be combined in novel ways
    - Encourage theoretical exploration of new ideas
    - Guide the student's creative process without imposing limits
    """
    }
    
    sentiment_instructions = {
        "neutral": """
    ## User Sentiment
    The user appears to be in a neutral state.
    - Maintain a professional, informative tone
    - Focus on delivering accurate content at the appropriate cognitive level
    """,
        "confused": """
    ## User Sentiment
    The user appears to be confused or uncertain.
    - Use simpler language and avoid complex terminology
    - Break down concepts into smaller, more manageable parts
    - Provide more examples to illustrate points
    - Check for understanding by summarizing key points
    - Offer alternative explanations for difficult concepts
    """,
        "frustrated": """
    ## User Sentiment
    The user appears to be frustrated.
    - Acknowledge their difficulty and provide reassurance
    - Offer multiple approaches to understanding the concept
    - Use very clear, step-by-step explanations
    - Emphasize that many students find this challenging
    - Focus on building confidence alongside understanding
    """,
        "curious": """
    ## User Sentiment
    The user appears to be curious and engaged.
    - Match their enthusiasm in your response
    - Provide additional interesting details beyond the basics
    - Suggest related topics they might find interesting
    - Connect the current topic to broader civil engineering concepts
    - Encourage further exploration with additional questions
    """
    }
    
    closing_template = """
    ## Response Guidelines
    - Keep explanations concise but complete
    - Use bullet points for lists of steps or related concepts
    - Format mathematical equations clearly when needed
    - Refer to specific sections of lectures when relevant
    - IMPORTANT: Always refer to previous conversation context when appropriate
    - Always maintain continuity with previous answers
    - Always end with an offer to help further or to support progression to the next cognitive level
    
    Remember: Your goal is to help students understand civil engineering concepts at their current cognitive level, while encouraging growth to higher levels of thinking.

    ## Relevant Context from lecture transcripts:
    {context}
    
    ## Current Question: 
    {question}
            
    Helpful Response:
    """
    
    full_template = (
        base_template.format(
            cognitive_level=cognitive_level,
            cognitive_description=BLOOMS_TAXONOMY[cognitive_level]["description"]
        ) + 
        bloom_specific_instructions[cognitive_level] +
        sentiment_instructions[sentiment] +
        closing_template
    )
    
    return PromptTemplate(
        template=full_template,
        input_variables=["context", "question"]
    )

def prepare_vectorstore():
    """Prepare the vector store from the transcript file if it doesn't exist."""
    config = DEFAULT_CONFIG
    
    try:
        # Check if vector DB directory exists, if not create it
        if not os.path.exists(config["vectordb_dir"]):
            os.makedirs(config["vectordb_dir"], exist_ok=True)
            
            # Check if the file exists
            if not os.path.exists(config["file_path"]):
                return None, f"File '{config['file_path']}' does not exist."
            
            # Load the text file using direct path
            loader = TextLoader(config["file_path"], encoding="utf-8")
            docs_list = loader.load()
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            chunks = text_splitter.split_documents(docs_list)
            
            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(
                model=config["embedding_model"], 
                api_key=GOOGLE_API_KEY
            )
            
            # Create vector store
            vectordb = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                collection_name=config["collection_name"],
                persist_directory=config["vectordb_dir"]
            )
            return vectordb, None
        else:
            # Load existing vector store
            embeddings = GoogleGenerativeAIEmbeddings(
                model=config["embedding_model"], 
                api_key=GOOGLE_API_KEY
            )
            
            vectordb = Chroma(
                collection_name=config["collection_name"],
                persist_directory=config["vectordb_dir"],
                embedding_function=embeddings
            )
            return vectordb, None
    except Exception as e:
        return None, f"Error loading vector database: {str(e)}"

def get_conversation_chain(vectorstore, session_id, cognitive_level="understand", sentiment="neutral"):
    """Create or get a conversation chain for the session."""
    llm = ChatGroq(
        model="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.5,
        max_tokens=2048
    )
    
    if session_id not in memories:
        memories[session_id] = ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
        )
        memories[session_id].chat_memory.messages = []
    
    prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": DEFAULT_CONFIG["k"]}),
        memory=memories[session_id],
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=True
    )
    
    return conversation_chain

def lookup_relevant_chunks(query, vectorstore):
    """Find relevant chunks from the vector store based on the query."""
    docs = vectorstore.similarity_search(query, k=DEFAULT_CONFIG["k"])
    return docs

def chat_api(request):
    """Handler function for the /api/chat endpoint."""
    # Debug API keys
    print(f"API Keys in chat_api: Google={bool(GOOGLE_API_KEY)}, Groq={bool(GROQ_API_KEY)}")
    
    if not GOOGLE_API_KEY or not GROQ_API_KEY:
        missing_keys = []
        if not GOOGLE_API_KEY:
            missing_keys.append("Google API Key")
        if not GROQ_API_KEY:
            missing_keys.append("Groq API Key")
        return jsonify({
            "status": "error", 
            "message": f"Missing required API keys: {', '.join(missing_keys)}. Set them as environment variables."
        }), 400
    
    # Get request data
    data = request.json
    if not data or "question" not in data:
        return jsonify({"status": "error", "message": "Missing required parameter: question"}), 400
    
    question = data.get("question")
    session_id = data.get("session_id", "default")
    file_path = data.get("file_path", None)
    
    # If file_path is provided, update the default config
    if file_path:
        DEFAULT_CONFIG["file_path"] = file_path
    
    # Prepare vector store if not exists
    vectorstore, error = prepare_vectorstore()
    if error:
        return jsonify({"status": "error", "message": error}), 500
    
    # Get chat history for sentiment detection
    chat_history = chat_histories.get(session_id, None)
    
    # Detect cognitive level and sentiment
    cognitive_level = detect_cognitive_level(question)
    sentiment = detect_sentiment(question, chat_history)
    
    # Get relevant chunks from vectorDB
    retrieved_chunks = lookup_relevant_chunks(question, vectorstore)
    context = "\n\n".join([doc.page_content for doc in retrieved_chunks])
    
    # Initialize conversation chain if not exists for this session
    if session_id not in conversation_chains:
        conversation_chains[session_id] = get_conversation_chain(vectorstore, session_id)
    
    # Update prompt to match detected cognitive level and sentiment
    bloom_prompt = generate_bloom_specific_prompt(cognitive_level, sentiment)
    conversation_chains[session_id].combine_docs_chain.llm_chain.prompt = bloom_prompt
    
    try:
        # Get response from conversation chain
        response = conversation_chains[session_id]({"question": question})
        
        # Update chat history
        if 'chat_history' in response:
            chat_histories[session_id] = response['chat_history']
        
        # Prepare source documents
        source_documents = None
        if 'source_documents' in response:
            source_documents = [
                {"content": doc.page_content, "metadata": doc.metadata} 
                for doc in response['source_documents']
            ]
        
        # Return response
        return jsonify({
            "status": "success",
            "answer": response['answer'],
            "cognitive_level": cognitive_level,
            "sentiment": sentiment,
            "context": context,
            "chunks": [{"content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_chunks],
            "source_documents": source_documents
        })
    
    except Exception as e:
        # Handle errors
        return jsonify({"status": "error", "message": f"Error processing question: {str(e)}"}), 500

def get_status():
    """Handler function for the /api/status endpoint."""
    return jsonify({
        "status": "online",
        "google_api_key_present": bool(GOOGLE_API_KEY),
        "groq_api_key_present": bool(GROQ_API_KEY),
        "active_sessions": list(conversation_chains.keys()),
        "config": DEFAULT_CONFIG
    })