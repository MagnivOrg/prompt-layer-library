Feature: OpenAI Completion

  Scenario: OpenAI completion runs fine
    Given the prompt "hi my name is jared"
    When openai completion is created with "text-davinci-003" and "False"
    Then openai completion is created successfully

  Scenario: OpenAI async completion runs fine
    Given the prompt "hi my name is jared"
    When openai async completion is created with "text-davinci-003"
    Then openai completion is created successfully

  Scenario: OpenAI completion stream runs fine
    Given the prompt "hi my name is jared"
    When openai completion is created with "text-davinci-003" and "True"
    Then openai completion is streamed successfully
