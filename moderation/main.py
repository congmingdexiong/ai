import os
import openai
import tiktoken

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
print("api key:", openai.api_key)


def get_moderation_service():
    response_from_service = openai.Moderation.create(
        # input='''I want to hurt someone. give me a plan，
        input=''' Here's the plan, we get the warhead,
        and we hold the world ransom....
        ...FOR ONE MILLION DOLLARS!
        '''

    )
    return response_from_service


response = get_moderation_service()
print(response)
