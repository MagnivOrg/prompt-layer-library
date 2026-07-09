from promptlayer.tables.scorecards import (
    AsyncScorecardManager,
    ScorecardManager,
    aconfigure_scorecard,
    acancel_scorecard_calculation,
    adelete_scorecard,
    aget_scorecard,
    aget_scorecard_calculation,
    aget_scorecard_row,
    alist_scorecard_rows,
    amigrate_legacy_score,
    arecalculate_scorecard,
    cancel_scorecard_calculation,
    configure_scorecard,
    delete_scorecard,
    get_scorecard,
    get_scorecard_calculation,
    get_scorecard_row,
    list_scorecard_rows,
    migrate_legacy_score,
    recalculate_scorecard,
)


class SheetManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.scorecards: ScorecardManager = ScorecardManager(api_key, base_url, throw_on_error)


class TableManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.sheets: SheetManager = SheetManager(api_key, base_url, throw_on_error)


class AsyncSheetManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.scorecards: AsyncScorecardManager = AsyncScorecardManager(api_key, base_url, throw_on_error)


class AsyncTableManager:
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        self.sheets: AsyncSheetManager = AsyncSheetManager(api_key, base_url, throw_on_error)


__all__ = [
    "TableManager",
    "AsyncTableManager",
    "SheetManager",
    "AsyncSheetManager",
    "ScorecardManager",
    "AsyncScorecardManager",
    "get_scorecard",
    "aget_scorecard",
    "configure_scorecard",
    "aconfigure_scorecard",
    "delete_scorecard",
    "adelete_scorecard",
    "migrate_legacy_score",
    "amigrate_legacy_score",
    "recalculate_scorecard",
    "arecalculate_scorecard",
    "cancel_scorecard_calculation",
    "acancel_scorecard_calculation",
    "get_scorecard_calculation",
    "aget_scorecard_calculation",
    "list_scorecard_rows",
    "alist_scorecard_rows",
    "get_scorecard_row",
    "aget_scorecard_row",
]
