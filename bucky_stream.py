from promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_7b93c14d8a188035bcaa52e61d9b8c58", enable_tracing=True)
OpenAI = pl.openai.OpenAI
client = OpenAI()


def get_streaming_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    for chunk in response:
        print(chunk.choices[0].delta.content)


if __name__ == "__main__":
    get_streaming_response("Once upon a time, in a land far, far away, there was a")
