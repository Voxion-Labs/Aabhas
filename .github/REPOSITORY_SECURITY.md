# Repository write policy

The release pipeline uses container isolation for deterministic builds and
publishes artifacts through the GitHub Actions bot after successful `main` CI.
It is not an EDR-evasion mechanism and does not hide workflow activity.

The repository files enforce the following controls:

- `CODEOWNERS` requires owner review for repository changes.
- Build and security workflows have `contents: read` only.
- `release.yml` is the only workflow with `contents: write`, and it runs only
  after the named core CI workflow succeeds on `main`.
- The bot publishes a release artifact; it does not push generated source code
  back to `main`.

To enforce “owner and bot only” at the branch boundary, the repository owner
must enable a GitHub branch ruleset for `main` in Settings → Rules → Rulesets:

1. Require a pull request and require Code Owner approval.
2. Restrict deletions and force pushes.
3. Restrict push actors to the owner and the repository’s approved GitHub
   Actions/bot identity.
4. Keep repository Actions permissions on read-only by default; retain the
   release job’s narrowly scoped `contents: write` permission.

GitHub repository settings are account-level controls and cannot be reliably
enforced by a file committed inside the repository.
