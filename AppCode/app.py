import os
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialization ---
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
GCP_PROJECT_ID = "ai-serverless-assistant"
GCP_LOCATION = os.environ.get('GCP_LOCATION', 'us-central1')

logger.info(f"Initializing with GCP_PROJECT_ID: {GCP_PROJECT_ID}")
logger.info(f"Using GCP_LOCATION: {GCP_LOCATION}")

# Initialize services lazily
db = None
model = None

def init_services():
    global db, model
    try:
        from google.cloud import firestore
        import vertexai
        from vertexai.generative_models import GenerativeModel
        
        logger.info("Initializing Vertex AI...")
        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        
        logger.info("Initializing Firestore...")
        db = firestore.Client()
        
        logger.info("Initializing Gemini model...")
        # Try different model names based on supported models
        model_names = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro"]
        model = None
        
        for model_name in model_names:
            try:
                logger.info(f"Trying model: {model_name}")
                model = GenerativeModel(model_name)
                logger.info(f"Successfully initialized model: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {str(e)}")
                continue
        
        if model is None:
            raise Exception("No Gemini model could be initialized")
        
        logger.info("Successfully initialized all services")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        return False

# --- Helper Function ---
def get_or_create_user_data(user_id):
    if db is None:
        return {"preferences": "User is a tech enthusiast.", "conversation_history": []}
    
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        if user_doc.exists:
            return user_doc.to_dict()
        else:
            default_data = {"preferences": "User is a tech enthusiast.", "conversation_history": []}
            user_ref.set(default_data)
            return default_data
    except Exception as e:
        logger.error(f"Error accessing Firestore: {str(e)}")
        return {"preferences": "User is a tech enthusiast.", "conversation_history": []}

# --- Test Route ---
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Service is running", "status": "ok"})

# --- Health Check Route ---
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Try to initialize services if not already done
        if db is None or model is None:
            init_services()
        
        firestore_status = "connected" if db is not None else "not_initialized"
        model_status = "initialized" if model is not None else "not_initialized"
        
        # Actually test the model if it exists
        if model is not None:
            try:
                # Try a simple test generation
                test_response = model.generate_content("Hello")
                model_status = "working"
            except Exception as e:
                model_status = f"error: {str(e)[:50]}"
        
        if db is not None:
            # Test Firestore connection
            test_doc = db.collection('health_check').document('test').get()
        
        return jsonify({
            "status": "healthy",
            "gcp_project_id": GCP_PROJECT_ID,
            "gcp_location": GCP_LOCATION,
            "firestore": firestore_status,
            "vertex_ai": model_status
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "gcp_project_id": GCP_PROJECT_ID,
            "gcp_location": GCP_LOCATION
        }), 500

# --- API Route ---
@app.route('/ask', methods=['POST'])
def ask_assistant():
    try:
        data = request.get_json()
        if not data or 'question' not in data or 'user_id' not in data:
            return jsonify({"error": "Invalid request."}), 400

        user_id, question = data['user_id'], data['question']
        
        # Initialize services if not already done
        if db is None or model is None:
            if not init_services():
                return jsonify({"error": "AI services not available. Please try again later."}), 503
        
        user_data = get_or_create_user_data(user_id)
        preferences = user_data.get('preferences', '')
        history = "\n".join(user_data.get('conversation_history', []))
        
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        prompt = f"""You are a helpful personal assistant. Current date: {current_date}. User's preferences: {preferences}. Conversation history: {history}. Based on all this context, answer the user's question: "{question}" """
        
        logger.info(f"Generating content for user {user_id} with question: {question[:50]}...")
        response = model.generate_content(prompt)
        assistant_response = response.text
        logger.info(f"Successfully generated response: {assistant_response[:50]}...")
        
        current_history = user_data.get('conversation_history', [])
        current_history.append(f"- User: '{question}'\n- You: '{assistant_response}'")
        if len(current_history) > 5:
            current_history = current_history[-5:]
        
        if db is not None:
            try:
                user_ref = db.collection('users').document(user_id)
                user_ref.update({'conversation_history': current_history})
                logger.info(f"Updated conversation history for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to update conversation history: {str(e)}")
        
        return jsonify({"response": assistant_response})
    except Exception as e:
        logger.error(f"Error in ask_assistant: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
