from openai import AzureOpenAI

import config

client = AzureOpenAI(
    azure_endpoint=config.azure_openai_endpoint,
    api_key=config.api_key,
    api_version='2024-05-01-preview'
)


def ask_gpt(message):
    completion = client.chat.completions.create(
        model='gpt-35-turbo',
        messages=message
    )
    return completion.choices[0].message.content


prompts = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Canada?"}
    ],
    [
        {"role": "system",
         "content": "You are a large company's CEO, please answer user's questions based your role and in 20 words."},
        {"role": "user", "content": "What is the biggest challenge i your work?"}
    ],
    [
        {"role": "system", "content": "translate user's message into Chinese."},
        {"role": "user", "content": "The best place in the world is ocean."}
    ]
]

for prompt in prompts:
    response = ask_gpt(prompt)
    print(response)
