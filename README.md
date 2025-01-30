# MunjeLumenDS2025
Ovde će se naći sav kod, vizuelizacije, kao i ostale bitne stvari unutar rešenja u Lumen Data Science takmičenju 2025

## Standards of committing and branching on the repository
1. Developers create a feature branch from main.
```bash
git checkout -b name/feature
```
Example:
```bash
git checkout -b vojislav/initial-setup
```
2. Developers commit to this branch in the following manner:
```bash
git commit -m "action: short description"
```
Example:
```bash
git commit -m "add: initial infrastructure for the project"
```
The actions that should generally be used are: `add`, `update`, `fix`.
3. When work is done (everything that's improtant for the feature is committed), they create a pull request.
```bash
git push origin name/feature
```
Example:
```bash
git push origin vojislav/initial-setup
```
4. CI/CD runs status checks (linting, tests, etc).
5. Code is reviewed and approved.
- This is important because the developers need to be up to date with what is being done on the project.
6. The PR is merged using rebase and fast-forward, keeping a linear history. When merging the pull request, you should select the option to `squash and merge`.
- The source branch will be deleted when merged.
7. No merge commits, ensuring a clean Git history.
8. When you are finished, to update the remote repo with your own:
```bash
git checkout main
git fetch --prune
git pull --rebase
```

General adding, committing tips:
- Use `git status` a lot to see what you're working with.
- Use `git tree` to see what branch you're checked out to, so there is no mixup to what branch is being committed to.

## Visual Studio Code setup
I suggest installing the following extensions, and configuring them in the settings:
- Black formatter, then go into VS Code settings > As a Default formatter add Black formatter > Search for Black > To `Black-formatter: Args` add: `--line-length=99`.
- Flake8, then go into VS Code settings > Search for Flake8 > For `Flake8: Import Strategy` put `fromEnvironment`.
- isort, then go into VS Code settings > Search for Flake8 > For `isort: Import Strategy` put `fromEnvironment`.
- Python Extension Pack is good too.
- RainbowCSV for easier viewing of `.csv` files.
- vscode-pdf for easier viewing of `.pdf` files.

## Repository setup

### Virtual environment setup
We will use virtual environments as it is more reliable for testing.
Creating a virtual environment requires a certain version of Python, we'll work with 3.10.

1. To create a virtual environment run the following:
`python3.10 -m venv venv`
2. Then, based on operating system, in the chosen terminal run the following:
- Windows (cmd):
`venv\Scripts\activate`
- Windows (PowerShell):
`venv\Scripts\Activate.ps1`
- Linux/macOS/Windows(Git Bash):
`source venv/bin/activate`

Note: To deactivate a virtual environment, simply run `deactivate` in the terminal.

### Installing dependencies
1. Pre-commit install for local linting (flake8, black, isort):
```bash
pip install pre-commit==2.13
pre-commit install
```

2. For the requirements, run the following command:
```bash
pip install -r requirements.txt
```
