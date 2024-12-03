# MMSR 2024 - Music Retrieval System

## Dependency Management

[Here](https://python-poetry.org/) is a installation guide for setting up poetry

### After Set-up Poetry

- execute `poetry config virtualenvs.in-project true` to set the config to create the `.venv` folder in the project directory. This helps to better manage virtual environments
- run `poetry install --no-root` in the root directory (where the `pyproject.toml` file is located) to install all dependencies.