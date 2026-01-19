# Contributing to Climate Rainfall Analysis

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/climate-rainfall-analysis.git
   cd climate-rainfall-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

## Development Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new features

3. **Run tests**
   ```bash
   pytest tests/
   ```

4. **Check code quality**
   ```bash
   # Format code
   black src/
   
   # Lint code
   flake8 src/
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Write docstrings for all functions/classes
- Keep functions focused and small
- Add type hints where appropriate

## Testing

- Write unit tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Commit Message Guidelines

Use clear, descriptive commit messages:

```
Add: New feature description
Fix: Bug fix description
Update: Changes to existing feature
Docs: Documentation updates
Test: Test additions/modifications
```

## Pull Request Process

1. Update README.md with details of changes if needed
2. Update documentation
3. Ensure all tests pass
4. Request review from maintainers

## Questions?

Open an issue or contact the maintainers.

## Code of Conduct

Be respectful, inclusive, and collaborative.
