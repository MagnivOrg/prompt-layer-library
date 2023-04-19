# 🍰 Promptlytics Client

**Open source Prompt Observability built on Frappe Framework.**

### Install Promptlyticscl

```bash
pip install promptlyticscl
```

### Installing Promptlytics Locally

Use `pip install .` to install locally.

### Using Promptlytics

To get started, create an account by clicking “*Log in*” on [Promptlytics](https://promptlytics.com/). Once logged in, click the button to create an API key and save this in a secure location ([Guide to Using Env Vars](https://towardsdatascience.com/the-quick-guide-to-using-environment-variables-in-python-d4ec9291619e)).

Once you have that all set up, [install Promptlytics using](https://pypi.org/project/promptlytics/) `pip`.

In the Python file where you use OpenAI APIs, add the following. This allows us to keep track of your requests without needing any other code changes.

```python
import promptlyticscl promptlytics
promptlytics.api_key = "<YOUR Promptlytics API KEY pl_xxxxxx>"
openai = promptlytics.openai
```

**You can then use `openai` as you would if you had imported it directly.**

<aside>
💡 Your OpenAI API Key is **never** sent to our servers. All OpenAI requests are made locally from your machine, Promptlytics just logs the request.
</aside>

### Adding Promptlytics tags: `pl_tags`

Promptlytics allows you to add tags through the `pl_tags` argument. This allows you to track and group requests in the dashboard. 

*Tags are not required but we recommend them!*

```python
openai.Completion.create(
  engine="text-ada-001", 
  prompt="My name is", 
  pl_tags=["name-guessing", "pipeline-2"]
)
```

After making your first few requests, you should be able to see them in the Promptlytics dashboard!

## Using the REST API

This Python library is a wrapper over Promptlytics's REST API. If you use another language, like Javascript, just interact directly with the API. 

Here is an example request below:

```jsx
import requests
request_response = requests.post(
  "https://API_URL/api/method/promptlytics.api.track",
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

We welcome contributions to our open source project, including new features, infrastructure improvements, and better documentation. For more information or any questions, contact us at [chinmay@hybrowlabs.com](mailto:chinmay@hybrowlabs.com).
