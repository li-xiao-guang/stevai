from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential

import config

client = ChatCompletionsClient(
    endpoint=config.azure_deepseek_endpoint,
    credential=AzureKeyCredential(config.api_key)
)


def ask_gpt(message):
    completion = client.complete(
        model='DeepSeek-V3',
        messages=message
    )
    return completion.choices[0].message.content


prompt = [
    SystemMessage(content="You are a helpful assistant."),
    UserMessage(content="What is the capital of Canada?")
]

userMessages = [
    "Where can I visit?",
    "What is the most famous university?",
    "Is there an animation college?"
]

response = ask_gpt(prompt)
print(response)

for msg in userMessages:
    prompt.append(AssistantMessage(content=response))
    prompt.append(UserMessage(content=msg))
    response = ask_gpt(prompt)
    print(response)
