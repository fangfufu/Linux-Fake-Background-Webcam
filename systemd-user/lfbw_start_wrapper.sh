#!/usr/bin/env bash

VENV_ACTIVATE_BIN="$HOME/.pyenv/versions/3.11.6/envs/lfbw/bin/activate"

. "$VENV_ACTIVATE_BIN"
lfbw -c "$HOME/.config/lfbw.ini"
