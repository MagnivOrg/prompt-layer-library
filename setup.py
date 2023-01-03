from setuptools import find_packages, setup

setup(
    name="promptlayer",
    description="PromptLayer is a package to keep track of your GPT models training",
    author_email="hello@magniv.io",
    url="https://www.magniv.io",
    project_urls={"Documentation": "https://magniv.notion.site/Prompt-Layer-Docs-db0e6f50cacf4564a6d09824ba17a629",},
    version="0.1.2",
    py_modules=["promptlayer"],
    packages=find_packages(),
    install_requires=["requests", "openai"],
)
