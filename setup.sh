#!/bin/bash

cp -r ./setup_files/* .git/hooks/

ENV_NAME=$(head -n 1 env.yml | cut -d' ' -f2)
conda env update -f env.yml --prune || conda env create -f env.yml
conda run -n "$ENV_NAME" pre-commit install
