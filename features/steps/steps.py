from unittest.mock import patch

import requests
from behave import *
from behave.api.async_step import async_run_until_complete
from langchain.schema import HumanMessage


@given('the human message "hi my name is jared"')
def step_impl(context):
    context.human_message = HumanMessage(content="hi my name is jared")


@when('langchain chat is created')
def step_impl(context):
    with patch.object(requests, 'post') as post:
        context.langchain_response = context.langchain_chat_openai(
            [context.human_message])
        post.assert_called_once()


@then('langchain chat is created successfully')
def step_impl(context):
    assert context.langchain_response is not None


@given('the prompt "{prompt}"')
def step_impl(context, prompt):
    context.prompt = prompt


@when('langchain response is generated')
def step_impl(context):
    with patch.object(requests, 'post') as post:
        context.langchain_response = context.langchain_openai_chat.generate([
            context.prompt])
        post.assert_called_once()


@then('langchain response is generated successfully')
def step_impl(context):
    assert context.langchain_response is not None


@when('langchain async response is generated')
@async_run_until_complete
async def step_impl(context):
    with patch.object(requests, 'post') as post:
        context.langchain_response = await context.langchain_openai_chat.agenerate([
            context.prompt])
        post.assert_called_once()


@given('the messages')
def step_impl(context):
    context.messages = []
    for row in context.table:
        context.messages.append(
            {"role": row.get('role'), "content": row.get('content')})


@when('openai chat is created with "{model}"')
def step_impl(context, model):
    with patch.object(requests, 'post') as post:
        context.response = context.openai.ChatCompletion.create(
            model=model,
            messages=context.messages,
        )
        post.assert_called_once()


@then('openai chat is created successfully')
def step_impl(context):
    assert context.response is not None


@when('openai completion is created with "{model}" and "{stream}"')
def step_impl(context, model, stream):
    stream = stream == "True"
    with patch.object(requests, 'post') as post:
        context.response = context.openai.Completion.create(
            model=model,
            stream=stream,
            prompt=context.prompt,
        )
        if not stream:
            post.assert_called_once()


@when('openai async completion is created with "{model}"')
@async_run_until_complete
async def step_impl(context, model):
    with patch.object(requests, 'post') as post:
        context.response = await context.openai.Completion.acreate(
            model=model,
            prompt=context.prompt,
        )
        post.assert_called_once()


@then('openai completion is created successfully')
def step_impl(context):
    assert context.response is not None


@then('openai completion is streamed successfully')
def step_impl(context):
    assert context.response is not None
