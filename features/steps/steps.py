from io import StringIO
from unittest.mock import patch

import requests
from behave import *
from behave.api.async_step import async_run_until_complete
from langchain.schema import HumanMessage

ERROR_STR = 'PromptLayer had the following error:'


@patch('sys.stderr', new_callable=StringIO)
@patch('requests.post', wraps=requests.post)
def assert_network_call(post, stderr, callback, **kwargs):
    callback(**kwargs)
    if not kwargs.get('stream'):
        post.assert_called_once()
    print(stderr.getvalue())
    assert ERROR_STR not in stderr.getvalue()


@patch('sys.stderr', new_callable=StringIO)
@patch('requests.post', wraps=requests.post)
async def assert_async_network_call(post, stderr, callback, **kwargs):
    await callback(**kwargs)
    if not kwargs.get('stream'):
        post.assert_called_once()
    print(stderr.getvalue())
    assert ERROR_STR not in stderr.getvalue()


@given('the human message "hi my name is jared"')
def step_impl(context):
    context.human_message = HumanMessage(content="hi my name is jared")


def langchain_chat_openai(context):
    context.langchain_response = context.langchain_chat_openai(
        [context.human_message])


@when('langchain chat is created')
def step_impl(context):
    assert_network_call(callback=langchain_chat_openai, context=context)


@then('langchain chat is created successfully')
def step_impl(context):
    assert context.langchain_response is not None


@given('the prompt "{prompt}"')
def step_impl(context, prompt):
    context.prompt = prompt


def langchain_openai_chat(context):
    context.langchain_response = context.langchain_openai_chat.generate([
                                                                        context.prompt])


@when('langchain response is generated')
def step_impl(context):
    assert_network_call(callback=langchain_openai_chat, context=context)


@then('langchain response is generated successfully')
def step_impl(context):
    assert context.langchain_response is not None


async def langchain_openai_chat_stream(context):
    context.langchain_response = await context.langchain_openai_chat.agenerate([context.prompt])


@when('langchain async response is generated')
@async_run_until_complete
async def step_impl(context):
    await assert_async_network_call(callback=langchain_openai_chat_stream, context=context)


@given('the messages')
def step_impl(context):
    context.messages = []
    for row in context.table:
        context.messages.append(
            {"role": row.get('role'), "content": row.get('content')})


def openai_chat(context, model):
    context.response = context.openai.ChatCompletion.create(
        model=model,
        messages=context.messages,
    )


@when('openai chat is created with "{model}"')
def step_impl(context, model):
    assert_network_call(callback=openai_chat, context=context, model=model)


@then('openai chat is created successfully')
def step_impl(context):
    assert context.response is not None


def openai_completion(context, model, stream):
    context.response = context.openai.Completion.create(
        model=model,
        stream=stream,
        prompt=context.prompt,
    )


@when('openai completion is created with "{model}" and "{stream}"')
def step_impl(context, model, stream):
    stream = stream == "True"
    assert_network_call(callback=openai_completion, context=context,
                        model=model, stream=stream)


async def openai_async_completion(context, model):
    context.response = await context.openai.Completion.acreate(
        model=model,
        prompt=context.prompt,
    )


@when('openai async completion is created with "{model}"')
@async_run_until_complete
async def step_impl(context, model):
    await assert_async_network_call(callback=openai_async_completion, context=context, model=model)


@then('openai completion is created successfully')
def step_impl(context):
    assert context.response is not None


@then('openai completion is streamed successfully')
def step_impl(context):
    assert context.response is not None
