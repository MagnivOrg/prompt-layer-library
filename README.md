<div align="center">

# 🍰 PromptLayer

**The first platform built for <span style="background-color: rgb(219, 234, 254);">prompt engineers</span>**

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://docs.promptlayer.com"><img alt="Docs" src="https://custom-icon-badges.herokuapp.com/badge/docs-PL-green.svg?logo=cake&style=for-the-badge"></a>
<a href="https://www.loom.com/share/196c42e43acd4a369d75e9a7374a0850"><img alt="Demo with Loom" src="https://img.shields.io/badge/Demo-loom-552586.svg?logo=loom&style=for-the-badge&labelColor=gray"></a>

---  

<div align="left">

[PromptLayer](https://promptlayer.com/) is the first platform that allows you to track, manage, and share your GPT prompt engineering. PromptLayer acts a middleware between your code and OpenAI’s python library. 

PromptLayer records all your OpenAI API requests, allowing you to search and explore request history in the PromptLayer dashboard.

This repo contains the Python wrapper library for PromptLayer.

## Quickstart ⚡

### Install PromptLayer

```bash
pip install promptlayer
```

### Installing PromptLayer Locally

Use `pip install .` to install locally.

### Using PromptLayer

To get started, create an account by clicking “*Log in*” on [PromptLayer](https://promptlayer.com/). Once logged in, click the button to create an API key and save this in a secure location ([Guide to Using Env Vars](https://towardsdatascience.com/the-quick-guide-to-using-environment-variables-in-python-d4ec9291619e)).

Once you have that all set up, [install PromptLayer using](https://pypi.org/project/promptlayer/) `pip`.

In the Python file where you use OpenAI APIs, add the following. This allows us to keep track of your requests without needing any other code changes.

```python
import promptlayer
promptlayer.api_key = "<YOUR PromptLayer API KEY pl_xxxxxx>"
openai = promptlayer.openai
```

**You can then use `openai` as you would if you had imported it directly.**

<aside>
💡 Your OpenAI API Key is **never** sent to our servers. All OpenAI requests are made locally from your machine, PromptLayer just logs the request.
</aside>

### Adding PromptLayer tags: `pl_tags`

PromptLayer allows you to add tags through the `pl_tags` argument. This allows you to track and group requests in the dashboard. 

*Tags are not required but we recommend them!*

```python
openai.Completion.create(
  engine="text-ada-001", 
  prompt="My name is", 
  pl_tags=["name-guessing", "pipeline-2"]
)
```

After making your first few requests, you should be able to see them in the PromptLayer dashboard!

## Using the REST API

This Python library is a wrapper over PromptLayer's REST API. If you use another language, like Javascript, just interact directly with the API. 

Here is an example request below:

```jsx
import requests
request_response = requests.post(
  "https://api.promptlayer.com/track-request",
  json={
    "function_name": "openai.Completion.create",
    "args": [],
    "kwargs": {"engine": "text-ada-001", "prompt": "My name is"},
    "tags": ["hello", "world"],
    "request_response": {"id": "cmpl-6TEeJCRVlqQSQqhD8CYKd1HdCcFxM", "object": "text_completion", "created": 1672425843, "model": "text-ada-001", "choices": [{"text": " advocacy\"\n\nMy name is advocacy.", "index": 0, "logprobs": None, "finish_reason": "stop"}]},
    "request_start_time": 1673987077.463504,
    "request_end_time": 1673987077.463504,
    "api_key": "pl_<YOUR API KEY>",
  },
)
```

## Contributing

We welcome contributions to our open source project, including new features, infrastructure improvements, and better documentation. For more information or any questions, contact us at [hello@promptlayer.com](mailto:hello@promptlayer.com).
