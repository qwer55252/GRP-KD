# Historical third-party source

`NeMo/` is the source snapshot included in the research repository at commit `b995ba3`. Its contents are unchanged by this migration. It identifies itself as `nemo_toolkit 2.5.0rc0`; an exact upstream commit was not recorded.

**This snapshot is incomplete:** the ASR `parts` package is absent, so it is not used as an installable dependency. The public setup pins the official `nemo_toolkit[asr]==2.5.0` package instead. Do not add this historical tree to `PYTHONPATH`.

Its license, notices, and upstream attribution remain intact. The research code license does not replace third-party notices.
