import os

# Ensure .env is loaded (if not already done globally)
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

def get_agronomist_response(disease_name: str, disease_info: dict, user_message: str, chat_history: list) -> str:
    """
    Get a response from the AI Agronomist Chatbot.
    If GROQ_API_KEY is available, it uses Groq Cloud (Llama 3). 
    Otherwise, it returns a simulated helpful mock response.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    
    if api_key:
        try:
            from groq import Groq
            client = Groq(api_key=api_key)

            # Construct system prompt
            system_prompt = f"""You are a professional Agronomist AI Assistant.
            The user recently scanned a crop leaf and the system detected: {disease_name}.
            Severity is: {disease_info.get('severity', 'Unknown')}.
            Standard advice given: {', '.join(disease_info.get('treatment', []))}.
            Description: {disease_info.get('description', '')}

            Help the user with any follow-up questions they have about treating or identifying this disease. Give concise, actionable, and safe agricultural advice. Keep responses under 3 paragraphs."""

            messages = [{"role": "system", "content": system_prompt}]

            # Append history
            for msg in chat_history:
                role = "assistant" if msg["role"] in ["assistant", "model"] else "user"
                messages.append({"role": role, "content": msg["content"]})
            
            # Append new user message
            messages.append({"role": "user", "content": user_message})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=300,
            )

            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error connecting to AI Agronomist API: {str(e)}"

    else:
        # Fallback Mock response for demo purposes if no API key is provided
        lowercase_msg = user_message.lower()
        if "organic" in lowercase_msg:
            return f"Since you're dealing with {disease_name}, an organic approach would involve using copper fungicides (if applicable) or neem oil. Make sure to remove any infected debris immediately and improve airflow around the plants."
        elif "water" in lowercase_msg or "irrigation" in lowercase_msg:
            return f"For {disease_name}, it is critical to avoid overhead watering. Water strictly at the base of the plant early in the morning so the foliage can dry quickly in the sun."
        elif "amount" in lowercase_msg or "how much" in lowercase_msg:
            return "Application amounts depend heavily on the specific brand of fungicide you use. As a general rule, follow the manufacturer's label exactly. Usually, it's about 1-2 tablespoons of concentrate per gallon of water, applied every 7-10 days."
        else:
            return f"That's a great question about {disease_name}. As your AI Agronomist, I recommend monitoring the situation closely. (Note: I am running in mock mode. Add a GROQ_API_KEY to your .env to unlock the full LLM capabilities!)"
