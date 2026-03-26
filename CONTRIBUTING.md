# Contributing To M.A.R.C A1

## Branching Model

This repository follows a lightweight GitFlow-style model:

- `main`
  Stable branch. Only validated, releasable changes should land here.
- `develop`
  Integration branch for ongoing work.
- `feature/<topic>`
  New work that targets `develop`.
- `fix/<topic>`
  Bug fixes that target `develop`.
- `hotfix/<topic>`
  Production-critical fixes that target `main` and should be merged back into `develop`.

## Recommended Workflow

1. Branch from `develop` for normal work.
2. Keep changes focused and easy to review.
3. Run tests locally before pushing:

```bash
python -m pytest -q
```

4. Open a pull request into `develop`.
5. Merge `develop` into `main` only for stable, validated releases.

## Commit Style

Use short, imperative commit messages:

- `Add structured access mode handling`
- `Improve session verification loop`
- `Fix Windows bootstrap path`

## Pull Request Checklist

- The change matches the existing architecture.
- Tests pass locally.
- User-facing behavior is documented when relevant.
- Risky runtime or access-mode changes are called out clearly.
