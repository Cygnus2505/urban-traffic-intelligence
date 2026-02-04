# Contributing to Urban Traffic Intelligence

We welcome contributions to the Urban Traffic Intelligence project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. Check if the issue already exists in the [Issues](https://github.com/Cygnus2505/urban-traffic-intelligence/issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

### Submitting Changes

1. **Fork the Repository**
   ```bash
   git clone https://github.com/Cygnus2505/urban-traffic-intelligence.git
   cd urban-traffic-intelligence
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run syntax check
   python verify_syntax.py
   
   # Run tests
   pytest tests/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add: brief description of changes"
   ```

6. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a Pull Request on GitHub.

## Code Style

### Python
- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for functions and classes
- Keep functions focused and small

### Documentation
- Update README.md for major features
- Add API documentation for new endpoints
- Include code examples

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Testing Guidelines

- Write tests for new features
- Ensure existing tests pass
- Aim for good test coverage
- Test edge cases

## Pull Request Process

1. Update documentation with details of changes
2. Ensure all tests pass
3. Update the README.md if needed
4. Your PR will be reviewed by maintainers
5. Address any feedback
6. Once approved, it will be merged

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone.

### Expected Behavior

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what's best for the community
- Show empathy towards others

### Unacceptable Behavior

- Harassment or discriminatory language
- Trolling or insulting comments
- Public or private harassment
- Publishing others' private information

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
