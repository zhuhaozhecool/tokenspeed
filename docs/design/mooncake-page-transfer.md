# Mooncake page transfer compatibility

`TOKENSPEED_MOONCAKE_PAGE_TRANSFER` explicitly selects `native` (the default
configuration) or `batch`. The engine constructor requires the selection.
Native mode requires Mooncake’s page-write API. Batch mode expands page/field
metadata into bounded batches for the ordinary WRITE API; no tensor payload
is copied through host memory. A partial failure is returned to the caller
without blindly retrying successful writes. IPv6 initialization and bracketed
session addresses follow the upstream implementation.
