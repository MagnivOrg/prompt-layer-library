# Changelog

## Unreleased

- Added public table scorecard APIs under `client.tables.sheets.scorecards`.
- Existing `/score` compatibility types now allow scorecard fallback and piggyback recalculation responses.
- Legacy score configuration deletion is not automatic during migration; `delete_legacy_score` defaults to `False`.
