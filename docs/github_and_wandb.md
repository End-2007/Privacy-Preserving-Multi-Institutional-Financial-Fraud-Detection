# GitHub and W&B handoff

Riyanshu-owned repository setup is captured in `.github/` and
`CONTRIBUTING.md`. Pull requests are required, the review rotation is
documented, CI runs on pushes to `main` and pull requests, and generated data,
models, and W&B runs remain ignored.

The shared tracker defaults to offline mode. Sync runs after account access is
ready with:

```powershell
python scripts\sync_wandb_offline.py --project <project-name>
```

The W&B project and teammate invitations still require someone with access to
the team account. Never place API keys in the repository.

