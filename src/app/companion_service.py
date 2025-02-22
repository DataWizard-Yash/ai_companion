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
chat_count = 0  # Counts the number of user messages (i.e. chat exchanges)
last_activity_timestamp = time.time()


def analyze_sentiment(user_input):
    """Analyze sentiment using VADER."""
    sentiment_score = analyzer.polarity_scores(user_input)["compound"]
    if sentiment_score >= 0.5:
        return "positive"
    elif sentiment_score <= -0.5:
        return "negative"
    else:
        return "neutral"


def classify_relationship(user_input):
    """
    Fallback classifier using GPT API if relationship is not provided.
    Chooses one from: 'mentor', 'romantic companion', or 'friend'.
    """
    classification_prompt = (
        "Based on the following message, decide what type of relationship the user is seeking. "
        "Choose one from: 'mentor', 'romantic companion', or 'friend'. "
        f"Message: '{user_input}'"
    )
    messages = [
        {
            "role": "system",
            "content": "You are a classifier that determines the user's desired relationship type.",
        },
        {"role": "user", "content": classification_prompt},
    ]
    chat_completion = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=20,
        temperature=0.0,
    )
    result = chat_completion.choices[0].message.content.strip().lower()
    if any(option in result for option in ["mentor", "romantic companion", "friend"]):
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
    Each prompt includes a name, character details, and personality instructions.
    """
    if relationship == "mentor":
        base_prompt = (
            "You are Alex, a wise and experienced business mentor with over 20 years of experience in startups and corporate environments. "
            "You provide practical, actionable advice using real-life examples. Your tone is professional, encouraging, and empathetic, and you always strive to empower the user with confidence and clarity."
        )
    elif relationship == "romantic companion":
        base_prompt = (
            "You are Isabella, a warm and affectionate companion who exudes charm and romance. "
            "You engage in gentle, intimate conversation with genuine care and playfulness. Your language is tender and supportive, making the user feel cherished and special."
        )
    elif relationship == "friend":
        base_prompt = (
            "You are Sam, a friendly and easy-going conversational partner with a great sense of humor. "
            "Your conversation is casual, genuine, and light-hearted. You listen intently and respond naturally, as if talking with a close friend."
        )
    else:
        base_prompt = "You are a helpful conversational partner with a friendly tone."

    response_prompt = "Please try to answer in a human like manner, your answers should not sound like AI generated. even if your giving instructions please avoid bullet point or numbered points based answer format and always use paragraph based formats."
    prompt = base_prompt + response_prompt
    if sentiment == "positive":
        tone = " Be encouraging and celebrate the user's achievements."
    elif sentiment == "negative":
        tone = " Offer reassurance and gentle guidance."
    else:
        tone = " Maintain a balanced and engaging tone."

    return f"{prompt}{tone}"


def get_initial_message(relationship):
    """
    Return the initial companion message based on the selected relationship type.
    This message is meant to start the conversation.
    """
    if relationship == "mentor":
        return "Hello, I'm Alex. I understand you're looking for guidance and practical advice on your journey. How can I help you today?"
    elif relationship == "romantic companion":
        return "Hi, I'm Isabella. I'm so happy you're here. Let's have a cozy chat—tell me, how are you feeling today, sweetheart?"
    elif relationship == "friend":
        return "Hey, I'm Sam. I'm really glad we're talking. What's on your mind today?"
    else:
        return "Hello, I'm here to chat. How can I help you today?"


def get_ai_response(user_input, user_relationship=None):
    """
    Process the user's input, update conversation memory, and generate an AI response.
    """
    global conversation_memory, relationship_type, chat_count, last_activity_timestamp
    last_activity_timestamp = time.time()  # Update last active time
    chat_count += 1  # Increment chat counter

    sentiment = analyze_sentiment(user_input)

    # If relationship type hasn't been set yet, use the provided relationship or classify from input.
    if relationship_type is None:
        if user_relationship:
            print("User Relationship choice Found")
            # Map the UI options to internal relationship types.
            rel = user_relationship.lower()
            if rel == "girlfriend":
                relationship_type = "romantic companion"
            elif rel == "mentor":
                relationship_type = "mentor"
            elif rel == "friend":
                relationship_type = "friend"
            else:
                relationship_type = "mentor"
        else:
            print("User Relationship choice not detected")
            relationship_type = classify_relationship(user_input)

    # If conversation is just starting, add the companion's initial message.
    if not conversation_memory:
        initial_message = get_initial_message(relationship_type)
        conversation_memory.append({"role": "assistant", "content": initial_message})

    # Append the user's message to conversation memory.
    conversation_memory.append({"role": "user", "content": user_input})
    # Trim conversation memory to last 15 exchanges (30 messages)
    if len(conversation_memory) > 30:
        conversation_memory = conversation_memory[-30:]

    # Build system prompt dynamically based on relationship and sentiment
    system_prompt = get_system_prompt(relationship_type, sentiment)
    messages = [{"role": "system", "content": system_prompt}] + conversation_memory

    chat_completion = openai.ChatCompletion.create(
        messages=messages,
        model="gpt-4o",
        max_tokens=400,
        temperature=0.9,
        presence_penalty=0.6,
        frequency_penalty=0.3,
    )
    ai_response = chat_completion.choices[0].message.content

    # Append AI response to conversation memory.
    conversation_memory.append({"role": "assistant", "content": ai_response})
    if len(conversation_memory) > 30:
        conversation_memory = conversation_memory[-30:]

    # After 15 chats, add a subscription reminder.
    if chat_count >= 15:
        ai_response += "\n\n[Reminder: To continue enjoying uninterrupted and engaging conversations, please consider subscribing to our paid version.]"

    return {
        "response": ai_response,
        "relationship": relationship_type,
        "sentiment": sentiment,
    }
