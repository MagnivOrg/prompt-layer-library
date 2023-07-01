import promptlayer
import os
from promptlayer import openai
import promptlayer

promptlayer.api_key = 'pl_da7fff3c8e79d4a5e29399564d01c5a4'
openai.api_key = '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'

#                         # nameRegistry, langchain, tag, version, model
promptlayer.prompts.run("runRegistry", False, ['test123', 'promptlayer'], 2, 'text-ada-001', 'openai')

promptlayer.prompts.run("uhuh", False, ['test123', 'promptlayer'], 2, 'gpt-3.5-turbo', 'openai')

promptlayer.prompts.run("llm_bash", False, ['test123', 'promptlayer'], 1, 'claude-v1-100k', 'anthropic')

