from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import os

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reference concepts dictionary
CONCEPT_REFERENCES = {
    "Machine Learning": """Machine learning is a subset of artificial intelligence 
    that enables systems to learn and improve from experience without being explicitly 
    programmed. It focuses on developing computer programs that can access data and 
    use it to learn for themselves.""",
    
    "Cloud Computing": """Cloud computing is the delivery of computing services 
    including servers, storage, databases, networking, software, analytics and 
    intelligence over the internet to offer faster innovation, flexible resources 
    and economies of scale.""",
    
    "Deep Learning": """Deep learning is a subset of machine learning that uses 
    neural networks with many layers to learn representations of data with multiple 
    levels of abstraction, enabling machines to automatically learn features.""",
    
    "Artificial Intelligence": """Artificial intelligence is the simulation of 
    human intelligence processes by machines, especially computer systems, including 
    learning, reasoning, problem solving, perception and language understanding.""",
    
    "Neural Networks": """Neural networks are computing systems inspired by biological 
    neural networks in animal brains, consisting of interconnected nodes that process 
    information using connectionist approaches to computation.""",
    
    "Data Science": """Data science is an interdisciplinary field that uses scientific 
    methods, processes, algorithms and systems to extract knowledge and insights 
    from structured and unstructured data."""
}

def get_similarity_score(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(float(similarity) * 100, 2)

def evaluate_concept(transcribed_text: str, topic: str) -> dict:
    """Evaluate how well the transcribed text explains the concept"""
    if topic not in CONCEPT_REFERENCES:
        return {
            "success": False,
            "error": f"Topic '{topic}' not found in reference database"
        }
    
    reference = CONCEPT_REFERENCES[topic]
    score = get_similarity_score(transcribed_text, reference)
    
    if score >= 70:
        level = "Strong Understanding"
        emoji = "🟢"
    elif score >= 45:
        level = "Moderate Understanding"
        emoji = "🟡"
    else:
        level = "Poor Understanding"
        emoji = "🔴"
    
    return {
        "success": True,
        "topic": topic,
        "similarity_score": score,
        "understanding_level": level,
        "emoji": emoji,
        "reference_text": reference
    }

def auto_detect_topic(transcribed_text: str, api_key: str) -> str:
    """Use Gemini to auto detect what concept was explained"""
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        topics = list(CONCEPT_REFERENCES.keys())
        prompt = f"""Given this spoken explanation: "{transcribed_text}"
        
Which topic from this list best matches what was explained?
Topics: {', '.join(topics)}

Reply with ONLY the topic name, nothing else."""
        
        response = gemini.generate_content(prompt)
        detected = response.text.strip()
        
        if detected in CONCEPT_REFERENCES:
            return detected
        return topics[0]
    
    except Exception:
        return "Machine Learning"

def get_gemini_feedback(transcribed_text: str, topic: str, 
                        score: float, api_key: str) -> str:
    """Get personalized AI feedback using Gemini"""
    try:
        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""A student explained the concept of "{topic}" and got a 
similarity score of {score}%.

Their explanation: "{transcribed_text}"

Give 3-4 lines of personalized, encouraging feedback on:
1. What they explained well
2. What key points they missed
3. How to improve

Keep it simple, friendly and helpful."""
        
        response = gemini.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        return f"Could not generate AI feedback: {str(e)}"
