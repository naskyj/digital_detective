"""Session-wide safety net: DD_CONFIG_DIR is set to an isolated, throwaway
directory the moment pytest starts collecting this package's tests - BEFORE
any individual test file's own module-level code runs.

Real incident (2026-09-18/19, twice): `dd_lab.api.main` constructs a module-
level `JobManager(Path(_settings.cases_root) / ".dd_jobs.sqlite")`
(`api/main.py`), and `JobManager.__init__` unconditionally marks every
currently 'queued'/'running' job as 'error' - "interrupted, the app was
restarted" - on construction (`api/jobs.py`), since it assumes being
constructed means the app just (re)started. That assumption breaks the
moment ANYTHING ELSE constructs a `JobManager` against the same db file - and
`test_source_labeling.py` had exactly that: a plain module-level
`from dd_lab.api.main import _source_label_for_app`, which pytest imports
during COLLECTION, before any fixture (including one that would set
DD_CONFIG_DIR) gets a chance to run. Since `_settings.cases_root` defaults to
the REAL per-user cases directory when DD_CONFIG_DIR isn't set, every run of
this test suite was silently interrupting the user's real, actually-running
background jobs on their live dd-lab instance - confirmed twice today, hours
apart, with the second one killing a ~2-hour real memory-analysis run.

This is deliberately a plain module-level statement (conftest.py's own
top-level code, not inside a fixture), and deliberately `setdefault` rather
than an unconditional set - pytest imports each directory's conftest.py
before collecting any test file inside it, so this runs before the
collection-time import that caused the incident, and a real environment
override (a CI setup, a deliberate `DD_CONFIG_DIR=...` on the command line)
is still respected rather than clobbered.

Still fix any individual test file's own import to be deferred where you
find one (see test_api.py's `client` fixture for the established pattern) -
this is a backstop for what's missed, not a reason to stop caring."""

from __future__ import annotations

import os
import tempfile

os.environ.setdefault("DD_CONFIG_DIR", tempfile.mkdtemp(prefix="dd_lab_test_config_"))
