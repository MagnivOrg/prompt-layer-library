import asyncio

from promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_7b93c14d8a188035bcaa52e61d9b8c58", enable_tracing=True)


@pl.traceable()
async def beans(qty):
    await asyncio.sleep(0.1)  # Simulating some async operation
    return f"{qty} beans"


@pl.traceable()
async def apples(qty):
    beans_response = await beans(qty)
    await asyncio.sleep(0.1)  # Simulating some async operation
    return f"{beans_response} apples"


@pl.traceable()
async def get_response(input_variables):
    await apples(5)
    # Use asyncio.to_thread to run the synchronous pl.run method in a separate thread
    response = await asyncio.to_thread(
        pl.run,
        prompt_name="ai-poet",
        prompt_release_label="prod",
        input_variables=input_variables,
        metadata={
            "user_id": "123",
        },
    )

    return response


async def main():
    _input_variables = {"topic": "watermelon"}
    result = await get_response(_input_variables)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
