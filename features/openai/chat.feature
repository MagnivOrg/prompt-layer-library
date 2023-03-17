Feature: OpenAI Chat

  Scenario: OpenAI chat runs fine
    Given the messages
      | role      | content                           |
      | system    | You are a helpful assistant.      |
      | user      | Who won the world series in 2020? |
      | assistant | assistant                         |
      | user      | Where was it played?              |
    When openai chat is created with "gpt-3.5-turbo"
    Then openai chat is created successfully
