#!/bin/bash
set -e
sudo chown -R vscode:vscode /workspaces/MunjeLumenDS2025/training/
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get update
sudo apt-get install -y git-lfs libgl1
sudo apt install openssh-client
git lfs install
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install black==24.10.0 isort==5.13.2 flake8==7.1.1
pip install pre-commit==2.13.0
pre-commit install
pip install --extra-index-url https://download.pytorch.org/whl/cu126 -r requirements/requirements_train.txt

sudo chmod 700 ~/.ssh && sudo chmod 600 ~/.ssh/*
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts
