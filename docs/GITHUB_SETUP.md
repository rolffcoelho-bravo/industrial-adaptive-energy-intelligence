# GitHub setup

Recommended repository name:

```text
industrial-adaptive-energy-intelligence
```

Recommended description:

```text
Industrial energy forecasting, peak-risk classification, structural-drift governance, constrained optimization, and decision-grade reporting.
```

Recommended topics:

```text
industrial-ai steel-industry databricks machine-learning mlops energy-forecasting optimization model-monitoring python sql data-engineering generative-ai multi-agent-systems
```

## Public-release sequence

1. Create the repository without a generated README, `.gitignore`, or license because those files already exist locally.
2. Push the foundation to a temporary private repository or local branch.
3. Implement the first real-data baseline and obtain a green CI run.
4. Remove all local credentials and inspect the Git history.
5. Replace `OWNER` in the README badges with the GitHub account name.
6. Make the repository public.
7. Create release `v0.1.0-baseline` only after the real five-page brief is produced.

## Commands

Run from the directory containing this repository:

```bash
git init
git branch -M main
git add .
git commit -m "Establish governed industrial AI repository foundation"
git remote add origin git@github.com:rolffcoelho-bravo/industrial-adaptive-energy-intelligence.git
git push -u origin main
```

If HTTPS authentication is preferred, replace the SSH remote with the HTTPS remote shown by GitHub.

## Repository settings

- Require pull requests before merging to `main` once development begins.
- Require the `quality` CI job to pass.
- Enable secret scanning and push protection when available.
- Disable wiki and discussions initially to keep the review surface focused.
- Keep Issues enabled for visible engineering traceability.
- Do not enable GitHub Pages until the baseline report exists.
