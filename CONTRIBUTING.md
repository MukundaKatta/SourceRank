# Contributing to SourceRank

Thank you for your interest in contributing to SourceRank! This document provides guidelines and instructions for contributing.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/MukundaKatta/SourceRank.git
cd SourceRank

# Install in development mode
make dev

# Run tests to verify setup
make test
```

## Running Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run a specific test file
pytest tests/test_core.py -v
```

## Code Quality

```bash
# Lint
make lint

# Format
make format
```

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Write tests for any new functionality.
3. Ensure all tests pass and linting is clean.
4. Update documentation if you change public API.
5. Open a pull request with a clear description of the change.

## Adding a New Scoring Signal

To add a new scoring dimension:

1. Add the weight field to `SignalWeights` in `src/sourcerank/config.py`.
2. Add the signal field to `QualitySignals` in `src/sourcerank/core.py`.
3. Implement a `_score_<signal_name>` method on `SourceRanker` in `src/sourcerank/core.py`.
4. Include the signal in the composite calculation in the `score()` method.
5. Add tests in `tests/test_core.py`.

## Code Style

- Follow PEP 8 conventions (enforced via `ruff`).
- Use type annotations for all public functions.
- Write docstrings in NumPy style.
- Keep functions focused and under 30 lines where possible.

## Reporting Issues

Open a GitHub issue with:
- A clear title and description.
- Steps to reproduce (if applicable).
- Expected vs. actual behaviour.
- Python version and OS.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
