import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI API key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")

def generate_response(question1, answer1, question2):
    mes = [
        {            "role": "system",
            "content": "only answer if question is related to the previous chat else answer I don't know"
                       "like asking clarification, explanation, indepth, reason ,summaries etc."
                       "You should answer based on your previous response, do not change your stance, just explain more "

        },
        {
            "role": "user",
            "content": question1
        },
        {
            "role": "assistant",
            "content": answer1
        },
        {
            "role": "user",
            "content": question2 + "(answer I don't know if not related to previous chat)"
        }]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=mes,
        temperature=0.8,
    )
    # Extract and return the assistant's reply
    assistant_reply = response["choices"][0]["message"]["content"]
    return assistant_reply


