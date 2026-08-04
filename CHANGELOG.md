# Changelog

## Unreleased

- Added official OpenTelemetry auto-instrumentation for OpenAI, Anthropic Messages, Google GenAI in Gemini and Vertex AI modes, and AWS Bedrock through Botocore. Message content capture defaults to `SPAN_ONLY` and can be disabled with `NO_CONTENT`.
- Added public table scorecard APIs under `client.tables.sheets.scorecards`.
- Existing `/score` compatibility types now allow scorecard fallback and piggyback recalculation responses.
- Legacy score configuration deletion is not automatic during migration; `delete_legacy_score` defaults to `False`.
