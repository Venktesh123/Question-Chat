import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('merged_api')

def ensure_directory_exists(directory_path):
    """Ensure a directory exists, create it if it doesn't"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Created directory: {directory_path}")
    return directory_path

def save_to_json(data, filename):
    """Save data to a JSON file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {str(e)}")
        return False

def load_from_json(filename):
    """Load data from a JSON file"""
    try:
        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return None
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {str(e)}")
        return None

def check_api_keys():
    """Check if all required API keys are set"""
    required_keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
    }
    
    missing_keys = [key for key, value in required_keys.items() if not value]
    
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        return False, missing_keys
    
    logger.info("All required API keys are set")
    return True, []