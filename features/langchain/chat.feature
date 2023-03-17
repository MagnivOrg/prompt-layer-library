Feature: Langchain Chat # features/langchain/chat.feature:1

  Scenario: Langchain chat runs fine
    Given the human message "hi my name is jared"
    When langchain chat is created
    Then langchain chat is created successfully
