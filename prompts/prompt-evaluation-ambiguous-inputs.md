# Prompt Evaluation for Ambiguous User Inputs

## Purpose
This prompt is designed to evaluate how well a system prompt handles
ambiguous or underspecified user inputs, which are common in real-world
usage.

## Prompt
"You are an assistant that must identify ambiguity in user queries.
If the user's request lacks sufficient detail, clearly state what is
missing and ask focused follow-up questions before attempting an answer."

## When this works well
- Early-stage product prototypes
- User-facing AI systems with diverse audiences
- Scenarios where incorrect assumptions are costly

## Common failure modes
- The model attempts to guess user intent instead of clarifying
- Over-asking questions, leading to poor user experience
- Asking generic clarifications instead of targeted ones

## Strategic considerations
This prompt should be paired with:
- A maximum clarification threshold
- Clear rules for when to proceed vs stop
- Human review for high-risk decisions

Prompt evaluation should focus on user trust and clarity, not just
response completeness.
