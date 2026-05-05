import re

with open('utils/llm_chatbot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace variables
content = content.replace('GEMINI_API_KEY', 'GROQ_API_KEY')

# Replace logic block
old_logic = """    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            # Construct system prompt
            system_prompt = f\"\"\"You are a professional Agronomist AI Assistant.
            The user recently scanned a crop leaf and the system detected: {disease_name}.
            Severity is: {disease_info.get('severity', 'Unknown')}.
            Standard advice given: {', '.join(disease_info.get('treatment', []))}.
            Description: {disease_info.get('description', '')}

            Help the user with any follow-up questions they have about treating or identifying this disease. Give concise, actionable, and safe agricultural advice. Keep responses under 3 paragraphs.\"\"\"

            model = genai.GenerativeModel('gemini-2.0-flash', system_instruction=system_prompt)

            formatted_history = []
            for msg in chat_history:
                # Gemini uses "user" and "model" roles
                role = "model" if msg["role"] == "assistant" else msg["role"]   
                formatted_history.append({"role": role, "parts": [{"text": msg["content"]}]})

            chat = model.start_chat(history=formatted_history)
            response = chat.send_message(user_message)

            return response.text
        except Exception as e:
            return f"Error connecting to AI Agronomist API: {str(e)}\""""

new_logic = """    if api_key:
        try:
            import groq
            client = groq.Groq(api_key=api_key)

            # Construct system prompt
            system_prompt = f\"\"\"You are a professional Agronomist AI Assistant.
            The user recently scanned a crop leaf and the system detected: {disease_name}.
            Severity is: {disease_info.get('severity', 'Unknown')}.
            Standard advice given: {', '.join(disease_info.get('treatment', []))}.
            Description: {disease_info.get('description', '')}

            Help the user with any follow-up questions they have about treating or identifying this disease. Give concise, actionable, and safe agricultural advice. Keep responses under 3 paragraphs.\"\"\"

            messages = [{"role": "system", "content": system_prompt}]

            # Append history
            for msg in chat_history:
                role = "assistant" if msg["role"] in ["assistant", "model"] else "user"
                messages.append({"role": role, "content": msg["content"]})
            
            # Append new user message
            messages.append({"role": "user", "content": user_message})

            chat_completion = client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=250,
            )

            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error connecting to AI Agronomist API: {str(e)}\""""

content = content.replace(old_logic, new_logic)

with open('utils/llm_chatbot.py', 'w', encoding='utf-8') as f:
    f.write(content)
