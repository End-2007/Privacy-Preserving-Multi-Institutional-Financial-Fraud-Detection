# Contributing to FedGuard

Do not commit directly to `main`. Open an Issue, create a branch named
`<owner>/<task>` (for example, `riyanshu/validation-tests`), make focused
commits, and open a Pull Request. A teammate must approve the PR before merge.

Use `feat:`, `fix:`, `test:`, `docs:`, or `chore:` commit prefixes.

Review rotation:

- Anvi reviews Naman's PRs.
- Riyanshu reviews Anvi's PRs.
- Naman reviews Riyanshu's PRs.

Raw/processed data, model binaries, and W&B runs stay out of Git. Run
`pytest -q` before opening a PR. Create GitHub labels `stage-1`, `paysim`,
`ieee-cis`, `shared-layer`, `documentation`, `needs-review`, and `tests`.

