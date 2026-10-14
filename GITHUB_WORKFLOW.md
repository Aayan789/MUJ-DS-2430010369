# GitHub Workflow, Issues and Contribution Log

Follows the Batch F Capstone Guidelines (Steps 1 to 12).

## 1. Submission Checklist
- [ ] Accepted the GitHub Organization invitation; GitHub profile completed
- [ ] **Public** personal repo named `MUJ-DS-<RegNo>` (e.g. `MUJ-DS-229301001`)
- [ ] Instructor added as collaborator (do not remove during the training period)
- [ ] Folders present: `assignments/`, `notebooks/`, `code/`, `resources/`, `presentations/`, `capstone/`
- [ ] `README.md` has Name, Reg No, Branch, Batch F, Project Title, GitHub Username, Training Program
- [ ] Regular, meaningful commits (weekly at minimum)
- [ ] Team capstone repo created; all teammates **and** the instructor added as collaborators
- [ ] Issues created and assigned; work done through branches and pull requests
- [ ] Deliverables in repo: source code, documentation, presentation, screenshots, results, installation guide
- [ ] Final submission: personal repo URL, capstone repo URL, presentation, other required materials
- [ ] Not submitted as a ZIP only

## 2. Suggested Issues (create one per item and assign it)
| # | Title | Label | Suggested scope |
|---|---|---|---|
| 1 | Dataset collection and preprocessing | `data` | loaders for 5 datasets, scaling, train/val/test and PC1 shift splits |
| 2 | Implement MLP baseline | `model` | 2-hidden-layer MLP, parameter matching |
| 3 | Implement FastKAN-style KANLayer | `model` | RBF basis plus SiLU base path |
| 4 | Implement FreqKAN (Fourier features) | `model` | learnable Fourier front end |
| 5 | Training and evaluation pipeline | `model` | AdamW, early checkpointing, error metrics |
| 6 | Noise and distribution-shift experiments | `experiment` | noise levels 0.1, 0.25, 0.5; PC1 shift |
| 7 | Testing and reproducibility | `testing` | quick 2-seed run, seed control, shape checks |
| 8 | Results analysis and plots | `analysis` | results table, robustness plots, interpretation |
| 9 | Documentation | `documentation` | README, model docs, installation guide |
| 10 | Presentation and demo | `presentation` | slides, screenshots, live demo run |
| 11 | Deployment (optional) | `deployment` | Streamlit/Gradio demo to compare models on user-uploaded CSV |
| 12 | Ablation: width/depth confound | `enhancement` | addresses limitation 1 in the model docs |

### Issue template
```
**Description:**
**Acceptance criteria:**
- [ ]
**Assignee:**
**Linked PR:** (use "Closes #N" in the PR)
```

## 3. Branch and Pull Request Workflow
```bash
git checkout main && git pull
git checkout -b feature/<short-name>        # e.g. feature/freqkan-layer
# ...work...
git add <files>
git commit -m "Add FreqKAN Fourier feature layer (#4)"
git push -u origin feature/<short-name>
```
1. Open a pull request into `main` and reference the issue (`Closes #4`).
2. Request review from at least one teammate.
3. Address comments, then merge after approval (prefer *Squash and merge*).
4. Delete the merged branch.

**Branch names:** `feature/`, `fix/`, `docs/`, `experiment/`.
**Commit messages:** imperative, specific, with the issue number.

## 4. Weekly Update Routine (personal repo)
- Push the week's assignment to `assignments/` and notebooks to `notebooks/`.
- Add a short note per week to `resources/weekly_log.md` (what was done, what is next).
- Keep activity continuous. Several small commits are better than one large one.

## 5. Individual Contribution Log
Fill in one row per teammate. Contributions may be coding, documentation, testing, deployment, data collection, frontend/backend, or model development.

| Team member | GitHub username | Area of contribution | Issues | PRs | Commits |
|---|---|---|---|---|---|
| `<name>` | `<username>` | e.g. model development | #3, #4 | #7 | `<n>` |
| `<name>` | `<username>` | e.g. data collection, testing | #1, #7 | #5 | `<n>` |
| `<name>` | `<username>` | e.g. documentation, presentation | #9, #10 | #11 | `<n>` |

## 6. Evaluation Map (30 marks)
| Criterion | Marks | Where to show it |
|---|---|---|
| Project implementation and functionality | 10 | `code/`, `capstone/results/`, working install guide |
| Individual contribution (commits, issues, PRs, documentation) | 8 | Section 5 table, GitHub Insights, issue and PR history |
| Repository organisation and documentation | 4 | folder structure, README, `docs/` |
| Presentation and demonstration | 4 | `presentations/`, live run of the script |
| Innovation and problem solving | 4 | FreqKAN idea, shift evaluation, limitations and ablation plan |
