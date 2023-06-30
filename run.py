import promptlayer
import os
# from promptlayer import openai
import promptlayer
import anthropic

promptlayer.api_key = 'pl_da7fff3c8e79d4a5e29399564d01c5a4'
# openai.api_key = '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'

#                         # nameRegistry, langchain, tag, version, model
promptlayer.prompts.run("runRegistry", False, ['funciona?', 'xD'], 2, 'text-ada-001')

promptlayer.prompts.run("uhuh", False, ['funciona?', 'xD'], 2, 'gpt-3.5-turbo')

promptlayer.prompts.run("llm_bash", False, ['funciona?', 'xD'], 1, 'claude-v1-100k', 'anthropic')

