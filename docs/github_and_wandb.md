# GitHub and W&B handoff

Repository contribution rules, pull-request templates, and CI configuration
are in `.github/` and `CONTRIBUTING.md`. Generated data, models, and W&B run
files are excluded from Git.

## W&B status

The team project is [FedGuard Stage 1](https://forge.coreweave.com/wandb/ndugar007-bharati-vidyapeeths-college-of-engineering/fedguard-stage1). It contains four model-summary runs for PaySim and four for IEEE-CIS. These summaries use saved comparison results without retraining. Validation metrics are available for all models; final test metrics are recorded only for the selected LightGBM model on each dataset. Naman confirmed that teammate invitations were handled.

The legacy PaySim offline run files remain locally under `D:\MICRO\wandb`, outside this checkout. Their old W&B entries were removed after the per-model summaries were created. **Do not sync those legacy files unless you intentionally want to restore the old attempts and training-history run.**

For future offline runs saved under this checkout's `wandb/` directory, sync with:

```powershell
python scripts\sync_wandb_offline.py --project fedguard-stage1
```

Never place API keys in the repository. GitHub labels and branch-protection
settings are remote repository settings and are not verified by this local
handoff note.
