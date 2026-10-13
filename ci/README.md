# CI configuration

The workflow files live here rather than in `.github/workflows/` because the
token used to push these branches does not carry the `workflow` OAuth scope, and
GitHub rejects any push that creates or updates a workflow file without it.

Install them with:

```bash
make ci-install     # copies ci/github/*.yml into .github/workflows/
```

or by hand:

```bash
mkdir -p .github/workflows && cp ci/github/*.yml .github/workflows/
```

Then commit that from a session whose credentials have `workflow` scope (a
normal `git push` from a developer machine, or a PAT with the scope enabled).

| File | Purpose |
|---|---|
| `github/core-ci.yml` | Tests `core/` on Python 3.9 and 3.12, verifies the core imports with only numpy+scipy installed, runs the CLI smoke test, lints with ruff. |
