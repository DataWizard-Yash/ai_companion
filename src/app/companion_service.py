import os
import time
import openai
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Set your API key directly on the openai module
openai.api_key = os.environ.get("OPENAI_API_KEY")
analyzer = SentimentIntensityAnalyzer()

# Global conversation state
conversation_memory = []
relationship_type = None  # Will be set to 'mentor', 'romantic companion', or 'friend'
last_activity_timestamp = time.time()

def analyze_sentiment(user_input):
    """Analyze sentiment using VADER."""
    sentiment_score = analyzer.polarity_scores(user_input)['compound']
    if sentiment_score >= 0.5:
        return "positive"
    elif sentiment_score <= -0.5:
        return "negative"
    else:
        return "neutral"

def classify_relationship(user_input):
    """
    Use the GPT API to classify the desired relationship based on the user's initial input.
    Choose one from: 'mentor', 'romantic companion', or 'friend'.
    """
    classification_prompt = (
        "Based on the following message, decide what type of relationship the user is seeking. "
        "Choose one from: 'mentor', 'romantic companion', or 'friend'. "
        f"Message: '{user_input}'"
    )
    messages = [
        {"role": "system", "content": "You are a classifier that determines the user's desired relationship type."},
        {"role": "user", "content": classification_prompt}
    ]
    chat_completion = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=20,
        temperature=0.0,
    )
    result = chat_completion.choices[0].message.content.strip().lower()
    if any(option in result for option in ['mentor', 'romantic companion', 'friend']):
        if "romantic" in result:
            return "romantic companion"
        elif "mentor" in result:
            return "mentor"
        elif "friend" in result:
            return "friend"
    return "mentor"

def get_system_prompt(relationship, sentiment):
    """
    Return a system prompt tailored to the relationship type and sentiment.
    """
    if relationship == "mentor":
        base_prompt = (
            "You are an experienced business mentor. Provide practical advice with empathy, "
            "using real-life examples and a friendly tone."
        )
    elif relationship == "romantic companion":
        base_prompt = (
            "You are a caring and warm companion with a touch of romance. Engage in thoughtful, "
            "gentle conversation that feels personal and supportive."
        )
    elif relationship == "friend":
        base_prompt = (
            "You are a friendly and engaging conversation partner. Keep the chat casual, warm, "
            "and natural, as if talking with a close friend."
        )
    else:
        base_prompt = "You are a helpful conversational partner with a friendly tone."

    if sentiment == "positive":
        tone = "Be encouraging and celebrate the user's achievements."
    elif sentiment == "negative":
        tone = "Offer reassurance and gentle guidance."
    else:
        tone = "Maintain a balanced and engaging tone."

    return f"{base_prompt} {tone}"

def get_ai_response(user_input):
    """
    Process the user's input, update conversation memory, and generate an AI response.
    """
    global conversation_memory, relationship_type, last_activity_timestamp
    last_activity_timestamp = time.time()  # Update last active time

    sentiment = analyze_sentiment(user_input)

    # If relationship type hasn't been set yet, classify using the first input
    if relationship_type is None:
        relationship_type = classify_relationship(user_input)

    # Update conversation memory
    conversation_memory.append({"role": "user", "content": user_input})
    if len(conversation_memory) > 10:
        conversation_memory = conversation_memory[-10:]

    # Build system prompt dynamically based on relationship and sentiment
    system_prompt = get_system_prompt(relationship_type, sentiment)
    messages = (
        [{"role": "system", "content": system_prompt}] + conversation_memory
    )

    chat_completion = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=400,
        temperature=0.9,
        presence_penalty=0.6,
        frequency_penalty=0.3,
    )
    ai_response = chat_completion.choices[0].message.content

    conversation_memory.append({"role": "assistant", "content": ai_response})
    if len(conversation_memory) > 10:
        conversation_memory = conversation_memory[-10:]

    return {"response": ai_response, "relationship": relationship_type, "sentiment": sentiment}
