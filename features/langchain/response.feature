Feature: Langchain Generation # features/langchain/response.feature:1

  Scenario: Langchain generation runs fine
    Given the prompt "hi my name is jared"
    When langchain response is generated
    Then langchain response is generated successfully

  Scenario: Langchain async generation runs fine
    Given the prompt "hi my name is jared"
    When langchain async response is generated
    Then langchain response is generated successfully
