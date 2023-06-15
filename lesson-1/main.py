import os
import openai
import tiktoken

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
print("api key:", openai.api_key)



def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response_from_service = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0
    )
    return response_from_service.choices[0].message["content"]


response = get_completion("What is the capital of France?")
response1= get_completion("Take the letters in lollipop and reverse them")
print(response)
print(response1)
