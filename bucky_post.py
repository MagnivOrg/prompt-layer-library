from promptlayer import PromptLayer

pl = PromptLayer(api_key="pl_7b93c14d8a188035bcaa52e61d9b8c58")


@pl.traceable()
def beans(qty):
    import time

    time.sleep(1)
    return f"{qty} beans"


@pl.traceable()
def apples(qty):
    beans_response = beans(qty)
    return f"{beans_response} apples"


@pl.traceable()
def get_response(input_variables):
    apples(5)
    response = pl.run(
        prompt_name="ai-poet",
        prompt_release_label="prod",
        input_variables=input_variables,
        metadata={
            "user_id": "123",
        },
    )

    return response


if __name__ == "__main__":
    _input_variables = {"topic": "watermelon"}
    get_response(_input_variables)
