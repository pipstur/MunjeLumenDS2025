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
pip install pre-commit==2.13.0
pre-commit install
pip install -r requirements/requirements_train.txt

sudo chmod 700 ~/.ssh && sudo chmod 600 ~/.ssh/*
mkdir -p ~/.ssh
ssh-keyscan github.com >> ~/.ssh/known_hosts
