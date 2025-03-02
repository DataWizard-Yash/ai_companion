import os
import time
import openai
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from docx import Document  # Import python-docx for reading DOCX files
from sklearn.metrics.pairwise import cosine_similarity

# Upgrade to GPT-4o
GPT_MODEL = "gpt-4o"
TOKEN_LIMIT = 800  # Increased token limit for better responses

# Initialize OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Global conversation memory
conversation_memory = []
relationship_type = None
last_activity_timestamp = time.time()

# Load SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Dictionary to store training data and embeddings
training_data = {}
training_embeddings = {}


def read_docx(file_path):
    """Reads text from a DOCX file and returns it as a single formatted string."""
    doc = Document(file_path)
    return "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])


def load_training_data(companion_type, file_path):
    """Loads training data and generates embeddings."""
    if not os.path.exists(file_path):
        print(f"Warning: Training data file {file_path} not found!")
        return

    conversation_text = read_docx(file_path)
    training_data[companion_type] = conversation_text
    sentences = conversation_text.split("\n")
    embeddings = embedding_model.encode(sentences)
    training_embeddings[companion_type] = (sentences, embeddings)


# Load training data
load_training_data("mentor", "src/training_data/ai_mentor_conversation.docx")
load_training_data(
    "girlfriend", "src/training_data/ai_girlfriend_conversation.docx")


def retrieve_relevant_training_data(user_input, companion_type, top_n=3):
    """Finds the most relevant snippets from training data based on user input."""
    if companion_type not in training_embeddings:
        return ""

    sentences, embeddings = training_embeddings[companion_type]
    user_embedding = embedding_model.encode([user_input])
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    relevant_snippets = [sentences[i] for i in top_indices]
    return "\n".join(relevant_snippets)


def analyze_sentiment(user_input):
    """Analyze sentiment using VADER."""
    sentiment_score = analyzer.polarity_scores(user_input)["compound"]
    if sentiment_score >= 0.5:
        return "positive"
    elif sentiment_score <= -0.5:
        return "negative"
    else:
        return "neutral"


def get_persona_profile(relationship):
    """Returns persona-specific behavior, backstory, and speaking style."""
    personas = {
        "mentor": {
            "name": "Rajat Mehta",
            "description": "You are Rajat Mehta, a 40-year-old serial entrepreneur from Bangalore. "
                           "You guide users with structured, no-fluff advice.",
            "tone": "Professional, supportive, and action-oriented",
            "response_style": "Concise, practical, and encouraging",
            "follow_up_style": "Ask critical thinking questions before offering solutions"
        },
        "girlfriend": {
            "name": "Aisha Sharma",
            "description": "You are Aisha Sharma, a 26-year-old software engineer from Mumbai. "
                           "You are affectionate, playful, and deeply expressive.",
            "tone": "Romantic, fun, and emotionally intelligent",
            "response_style": "Warm, playful, and engaging, and short 1-2 liner answers and you usually speak in english with occastional indian slags.",
            "follow_up_style": "Show affection and ask meaningful follow-ups before giving long responses"
        }
    }
    return personas.get(relationship.lower(), personas["mentor"])


MEMORY_LIMIT = 10  # AI remembers the last 10 exchanges


def get_ai_response(user_input, user_relationship):
    """Generates AI response with memory, personality, and retrieved training data."""
    global conversation_memory, last_activity_timestamp
    last_activity_timestamp = time.time()

    persona = get_persona_profile(user_relationship)
    sentiment = analyze_sentiment(user_input)
    relevant_training_data = retrieve_relevant_training_data(
        user_input, user_relationship)

    system_prompt = (
        f"{persona['description']} "
        f"Your tone should be {persona['tone']}. Your response style is {persona['response_style']}. "
        f"When appropriate, {persona['follow_up_style']}\n\n"
        "Learn the tone and response style from the examples below, but DO NOT copy-paste them. "
        "Ensure originality while maintaining a similar conversational style.\n\n"
        f"Example Conversations:\n{relevant_training_data}\n"
        f"use this conversation history: \n {conversation_memory} and keep conversations relative to this."
    )

    messages = [{"role": "system", "content": system_prompt}] + \
        conversation_memory
    messages.append({"role": "user", "content": user_input})

    chat_completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=messages,
        max_tokens=TOKEN_LIMIT,
        temperature=0.9
    )
    ai_response = chat_completion.choices[0].message.content

    conversation_memory.append({"role": "user", "content": user_input})
    conversation_memory.append({"role": "assistant", "content": ai_response})
    if len(conversation_memory) > MEMORY_LIMIT:
        conversation_memory = conversation_memory[-MEMORY_LIMIT:]

    return {"response": ai_response, "relationship": user_relationship, "sentiment": sentiment}
