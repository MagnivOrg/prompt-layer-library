from typing import Any, Dict, Optional, Union

from promptlayer.tables import api as tables_api
from promptlayer.types.table import (
    AddTableRows,
    AddTraceImport,
    BatchRecalculateCells,
    ConfigureSheetScore,
    CreateColumn,
    CreateSheet,
    CreateSheetVersion,
    CreateTable,
    ListTablesParams,
    ResourceId,
    CellResponse,
    ColumnListResponse,
    ColumnResponse,
    SheetListResponse,
    SheetResponse,
    SheetStatusCountsResponse,
    SheetVersionListResponse,
    SheetVersionResponse,
    TableListResponse,
    TableResponse,
    TableScoreResponse,
    UpdateCell,
    UpdateColumn,
    UpdateSheet,
    UpdateTable,
)


class _TableResourceContext:
    """Shared credential/resource ids for nested Table managers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        table_id: Optional[ResourceId] = None,
        sheet_id: Optional[ResourceId] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.throw_on_error = throw_on_error
        self.table_id = table_id
        self.sheet_id = sheet_id


class TableSheetRowsManager(_TableResourceContext):
    def list(self) -> Union[Dict[str, Any], None]:
        return tables_api.list_smart_sheet_rows(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def add(self, body: AddTableRows) -> Union[Dict[str, Any], None]:
        return tables_api.add_smart_sheet_rows(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )


class AsyncTableSheetRowsManager(_TableResourceContext):
    async def list(self) -> Union[Dict[str, Any], None]:
        return await tables_api.alist_smart_sheet_rows(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def add(self, body: AddTableRows) -> Union[Dict[str, Any], None]:
        return await tables_api.aadd_smart_sheet_rows(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )


class TableSheetColumnsManager(_TableResourceContext):

    def list(self) -> Union[ColumnListResponse, None]:
        return tables_api.list_smart_sheet_columns(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def create(self, body: CreateColumn) -> Union[ColumnResponse, None]:
        return tables_api.create_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    def update(
        self,
        column_id: ResourceId,
        body: UpdateColumn,
    ) -> Union[ColumnResponse, None]:
        return tables_api.update_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            column_id,
            body,
        )

    def delete(self, column_id: ResourceId) -> bool:
        return tables_api.delete_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            column_id,
        )


class AsyncTableSheetColumnsManager(_TableResourceContext):

    async def list(self) -> Union[ColumnListResponse, None]:
        return await tables_api.alist_smart_sheet_columns(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def create(self, body: CreateColumn) -> Union[ColumnResponse, None]:
        return await tables_api.acreate_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    async def update(
        self,
        column_id: ResourceId,
        body: UpdateColumn,
    ) -> Union[ColumnResponse, None]:
        return await tables_api.aupdate_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            column_id,
            body,
        )

    async def delete(self, column_id: ResourceId) -> bool:
        return await tables_api.adelete_sheet_column(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            column_id,
        )


class TableSheetCellsManager(_TableResourceContext):

    def get(self, cell_id: ResourceId) -> Union[CellResponse, None]:
        return tables_api.get_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
        )

    def update(self, cell_id: ResourceId, body: UpdateCell) -> Union[CellResponse, None]:
        return tables_api.update_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
            body,
        )

    def recalculate(self, cell_id: ResourceId) -> Union[CellResponse, None]:
        return tables_api.recalculate_smart_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
        )

    def batch_recalculate(self, body: BatchRecalculateCells) -> Union[Dict[str, Any], None]:
        return tables_api.batch_recalculate_smart_sheet_cells(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )


class AsyncTableSheetCellsManager(_TableResourceContext):

    async def get(self, cell_id: ResourceId) -> Union[CellResponse, None]:
        return await tables_api.aget_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
        )

    async def update(self, cell_id: ResourceId, body: UpdateCell) -> Union[CellResponse, None]:
        return await tables_api.aupdate_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
            body,
        )

    async def recalculate(self, cell_id: ResourceId) -> Union[CellResponse, None]:
        return await tables_api.arecalculate_smart_sheet_cell(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            cell_id,
        )

    async def batch_recalculate(self, body: BatchRecalculateCells) -> Union[Dict[str, Any], None]:
        return await tables_api.abatch_recalculate_smart_sheet_cells(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )


class TableSheetVersionsManager(_TableResourceContext):

    def list(self) -> Union[SheetVersionListResponse, None]:
        return tables_api.list_sheet_versions(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def create(self, body: Optional[CreateSheetVersion] = None) -> Union[SheetVersionResponse, None]:
        return tables_api.create_sheet_version(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body or {},
        )

    def get(self, version_id: ResourceId) -> Union[SheetVersionResponse, None]:
        return tables_api.get_sheet_version(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            version_id,
        )

    def score_history(self) -> Union[Dict[str, Any], None]:
        return tables_api.get_sheet_score_history(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class AsyncTableSheetVersionsManager(_TableResourceContext):

    async def list(self) -> Union[SheetVersionListResponse, None]:
        return await tables_api.alist_sheet_versions(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def create(
        self,
        body: Optional[CreateSheetVersion] = None,
    ) -> Union[SheetVersionResponse, None]:
        return await tables_api.acreate_sheet_version(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body or {},
        )

    async def get(self, version_id: ResourceId) -> Union[SheetVersionResponse, None]:
        return await tables_api.aget_sheet_version(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            version_id,
        )

    async def score_history(self) -> Union[Dict[str, Any], None]:
        return await tables_api.aget_sheet_score_history(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class TableSheetScoreManager(_TableResourceContext):

    def get(self) -> Union[TableScoreResponse, None]:
        return tables_api.get_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def configure(self, body: ConfigureSheetScore) -> Union[TableScoreResponse, None]:
        return tables_api.configure_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    def recalculate(self) -> Union[TableScoreResponse, None]:
        return tables_api.recalculate_smart_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class AsyncTableSheetScoreManager(_TableResourceContext):

    async def get(self) -> Union[TableScoreResponse, None]:
        return await tables_api.aget_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def configure(self, body: ConfigureSheetScore) -> Union[TableScoreResponse, None]:
        return await tables_api.aconfigure_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    async def recalculate(self) -> Union[TableScoreResponse, None]:
        return await tables_api.arecalculate_smart_sheet_score(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class TableSheetManager(_TableResourceContext):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        table_id: ResourceId,
        sheet_id: ResourceId,
    ):
        super().__init__(api_key, base_url, throw_on_error, table_id=table_id, sheet_id=sheet_id)
        self.rows = TableSheetRowsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.columns = TableSheetColumnsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.cells = TableSheetCellsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.versions = TableSheetVersionsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.score = TableSheetScoreManager(api_key, base_url, throw_on_error, table_id, sheet_id)

    def get(self) -> Union[SheetResponse, None]:
        return tables_api.get_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def status_counts(self) -> Union[SheetStatusCountsResponse, None]:
        return tables_api.get_sheet_status_counts(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    def update(self, body: UpdateSheet) -> Union[SheetResponse, None]:
        return tables_api.update_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    def delete(self) -> bool:
        return tables_api.delete_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class AsyncTableSheetManager(_TableResourceContext):
    def __init__(
        self,
        api_key: str,
        base_url: str,
        throw_on_error: bool,
        table_id: ResourceId,
        sheet_id: ResourceId,
    ):
        super().__init__(api_key, base_url, throw_on_error, table_id=table_id, sheet_id=sheet_id)
        self.rows = AsyncTableSheetRowsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.columns = AsyncTableSheetColumnsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.cells = AsyncTableSheetCellsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.versions = AsyncTableSheetVersionsManager(api_key, base_url, throw_on_error, table_id, sheet_id)
        self.score = AsyncTableSheetScoreManager(api_key, base_url, throw_on_error, table_id, sheet_id)

    async def get(self) -> Union[SheetResponse, None]:
        return await tables_api.aget_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def status_counts(self) -> Union[SheetStatusCountsResponse, None]:
        return await tables_api.aget_sheet_status_counts(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )

    async def update(self, body: UpdateSheet) -> Union[SheetResponse, None]:
        return await tables_api.aupdate_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
            body,
        )

    async def delete(self) -> bool:
        return await tables_api.adelete_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            self.sheet_id,
        )


class TableSheetsManager(_TableResourceContext):

    def list(self) -> Union[SheetListResponse, None]:
        return tables_api.list_sheets(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
        )

    def create(self, body: Optional[CreateSheet] = None) -> Union[SheetResponse, None]:
        from promptlayer.tables.helpers import with_default_empty_sheet_source

        return tables_api.create_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            with_default_empty_sheet_source(body),
        )

    def for_sheet(self, sheet_id: ResourceId) -> TableSheetManager:
        return TableSheetManager(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            sheet_id,
        )


class AsyncTableSheetsManager(_TableResourceContext):

    async def list(self) -> Union[SheetListResponse, None]:
        return await tables_api.alist_sheets(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
        )

    async def create(self, body: Optional[CreateSheet] = None) -> Union[SheetResponse, None]:
        from promptlayer.tables.helpers import with_default_empty_sheet_source

        return await tables_api.acreate_sheet(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            with_default_empty_sheet_source(body),
        )

    def for_sheet(self, sheet_id: ResourceId) -> AsyncTableSheetManager:
        return AsyncTableSheetManager(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            self.table_id,
            sheet_id,
        )


class TableImportsManager(_TableResourceContext):

    def add_trace(self, body: AddTraceImport) -> Union[Dict[str, Any], None]:
        return tables_api.add_trace_import(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            body,
        )


class AsyncTableImportsManager(_TableResourceContext):

    async def add_trace(self, body: AddTraceImport) -> Union[Dict[str, Any], None]:
        return await tables_api.aadd_trace_import(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            body,
        )


class TableManager(_TableResourceContext):
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        super().__init__(api_key, base_url, throw_on_error)
        self.imports = TableImportsManager(api_key, base_url, throw_on_error)

    def list(self, params: Optional[ListTablesParams] = None) -> Union[TableListResponse, None]:
        return tables_api.list_tables(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            params,
        )

    def create(self, body: CreateTable) -> Union[TableResponse, None]:
        return tables_api.create_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            body,
        )

    def get(self, table_id: ResourceId) -> Union[TableResponse, None]:
        return tables_api.get_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
        )

    def update(self, table_id: ResourceId, body: UpdateTable) -> Union[TableResponse, None]:
        return tables_api.update_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
            body,
        )

    def delete(self, table_id: ResourceId) -> bool:
        return tables_api.delete_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
        )

    def sheets(self, table_id: ResourceId) -> TableSheetsManager:
        return TableSheetsManager(self.api_key, self.base_url, self.throw_on_error, table_id)


class AsyncTableManager(_TableResourceContext):
    def __init__(self, api_key: str, base_url: str, throw_on_error: bool):
        super().__init__(api_key, base_url, throw_on_error)
        self.imports = AsyncTableImportsManager(api_key, base_url, throw_on_error)

    async def list(self, params: Optional[ListTablesParams] = None) -> Union[TableListResponse, None]:
        return await tables_api.alist_tables(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            params,
        )

    async def create(self, body: CreateTable) -> Union[TableResponse, None]:
        return await tables_api.acreate_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            body,
        )

    async def get(self, table_id: ResourceId) -> Union[TableResponse, None]:
        return await tables_api.aget_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
        )

    async def update(self, table_id: ResourceId, body: UpdateTable) -> Union[TableResponse, None]:
        return await tables_api.aupdate_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
            body,
        )

    async def delete(self, table_id: ResourceId) -> bool:
        return await tables_api.adelete_table(
            self.api_key,
            self.base_url,
            self.throw_on_error,
            table_id,
        )

    def sheets(self, table_id: ResourceId) -> AsyncTableSheetsManager:
        return AsyncTableSheetsManager(self.api_key, self.base_url, self.throw_on_error, table_id)
