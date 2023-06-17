import os
import openai
import tiktoken

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

delimiter = "####"
system_message = f"""
You will be provided with customer service queries. \
The customer service query will be delimited with \
{delimiter} characters.
Classify each query into a primary category \
and a secondary category.
Provide you output in json format with the \
keys: primary and secondary.

Primary categories: Billing, Technical, Support,  \
Account Management, or General Inquiry.

Billing secondary categories:
unsubscribe or upgrade
Add a payment method
Explanation for charge
Dispute a charge

Technical Support secondary categories:
General troubleshooting
Device compatibility 
Software updates

Account management secondary categories:
Password reset
Update personal information
Close account
Account security

General Inquiry secondary categories:
Product information
Pricing 
Feedback
Speak to a human 
"""


def get_completion(messages, model="gpt-3.5-turbo"):
    response_from_service = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    return response_from_service.choices[0].message["content"]


user_message = f"""
    I want you to delete my profile and all of my  user data
    """
messages1 = [{"role": "system", "content": system_message}, {
    "role": 'user', "content": f"{delimiter}{user_message}{delimiter}"
}]
response = get_completion(messages1)
print(response)
