# Anvi contribution and Riyanshu handoff status

This note records Anvi's contribution to the shared handoff and the later
status of the account-level and release-level tasks.

## Completed on this branch

- Added validation tests for shared metrics, validation-only threshold choice,
  deterministic stratified splitting, and W&B offline defaults.
- Added a GitHub Actions workflow that installs the declared requirements and
  runs the test suite on pushes to `main` and pull requests.
- Added contribution rules, review rotation, a pull-request template, and an
  issue template.
- Added a documented W&B offline-run sync command.
- Preserved the team decisions: LightGBM is the selected model for both
  datasets, and IEEE-CIS selects class weights rather than SMOTE.

## Open at the time of Anvi's contribution

- Create the W&B project and invite the two teammates using the team account.
- Authenticate and perform the actual W&B offline-run sync.
- Create the GitHub labels in the remote repository and configure branch
  protection/reviewer rules if required by the team.
- Review this branch, make any final edits, and perform the final push/merge
  decision under Riyanshu's own account.

## Status at the Stage 1 handoff

- Naman created the W&B project, handled teammate invitations, and added four
  saved-metric summary runs per dataset. No training was repeated for this.
- The older PaySim W&B entries were removed after the summaries were verified;
  their offline `.wandb` files remain outside the repository.
- GitHub labels and branch protection remain remote settings to verify only if
  the team requires them.
- Naman's final source/documentation commit, push, and separate transfer-bundle
  ZIP remain to be completed.

## Not part of this split

SHAP explanations, Streamlit integration, schema-alignment ownership, and the
common model interface remain separate follow-on or Anvi-owned work, as noted
in `docs/HANDOVER.md`.

