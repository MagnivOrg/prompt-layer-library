from setuptools import find_packages, setup

setup(
    name="promptlayer",
    description="PromptLayer is a package to keep track of your GPT models training",
    author_email="hello@magniv.io",
    url="https://www.magniv.io",
    project_urls={"Documentation": "https://docs.magniv.io",},
    version="0.1.1",
    py_modules=["promptlayer"],
    packages=find_packages(),
    install_requires=["requests", "openai"],
)
