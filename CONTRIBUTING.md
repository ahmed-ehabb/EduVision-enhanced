# Contributing to EduVision

Thank you for considering contributing to EduVision! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/EduVision-enhanced/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Screenshots (if applicable)
   - Environment details (OS, Python version, GPU, etc.)

### Suggesting Features

1. Check existing feature requests in [Issues](https://github.com/yourusername/EduVision-enhanced/issues)
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Potential implementation approach

### Pull Requests

1. **Fork** the repository
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code standards below
4. **Test thoroughly** - ensure all existing tests pass
5. **Commit** with clear, descriptive messages:
   ```bash
   git commit -m "Add: Brief description of changes"
   ```
6. **Push** to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Create a Pull Request** with:
   - Description of changes
   - Related issue numbers
   - Screenshots (if UI changes)
   - Testing performed

## Development Setup

### Backend Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev

# Run linter
npm run lint

# Run tests
npm test
```

## Code Standards

### Python

- Follow **PEP 8** style guide
- Use **type hints** for function parameters and returns
- Write **docstrings** for all functions and classes
- Maximum line length: **100 characters**
- Use **meaningful variable names**

Example:
```python
def process_lecture(
    audio_file: str,
    textbook_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process lecture audio through AI pipeline.

    Args:
        audio_file: Path to audio/video file
        textbook_path: Optional path to textbook PDF

    Returns:
        Dictionary containing transcript, notes, quiz, etc.
    """
    # Implementation...
```

### JavaScript/React

- Follow **Airbnb JavaScript Style Guide**
- Use **functional components** with hooks
- Use **TypeScript** for new features (if applicable)
- Maximum line length: **100 characters**
- Use **meaningful component/variable names**

Example:
```javascript
/**
 * Upload lecture file with progress tracking
 * @param {File} file - Audio/video file to upload
 * @param {Object} metadata - Lecture metadata
 * @returns {Promise<string>} Job ID for tracking
 */
async function uploadLecture(file, metadata) {
  // Implementation...
}
```

### Database

- Use **snake_case** for table and column names
- Include **foreign key constraints**
- Add **indexes** for frequently queried columns
- Write **migrations** for schema changes

### Commit Messages

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code formatting (no logic changes)
- `refactor:` Code refactoring
- `test:` Adding/updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: Add quiz difficulty levels
fix: Resolve GPU memory leak in ASR module
docs: Update API documentation for /lectures endpoint
```

## Testing Guidelines

### Backend Tests

- Write unit tests for all utility functions
- Write integration tests for API endpoints
- Aim for >80% code coverage
- Mock external dependencies (APIs, database)

```python
def test_generate_quiz():
    """Test quiz generation with sample transcript"""
    result = quiz_generator.generate(sample_transcript)
    assert len(result['questions']) == 5
    assert all('correct_answer' in q for q in result['questions'])
```

### Frontend Tests

- Write unit tests for utility functions
- Write component tests using React Testing Library
- Test user interactions and edge cases

```javascript
test('displays error message on login failure', async () => {
  render(<LoginPage />);
  // Simulate failed login...
  expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
});
```

## Documentation

- Update README.md for user-facing changes
- Update API documentation for endpoint changes
- Add inline comments for complex logic
- Include examples in docstrings

## Performance Considerations

- **Backend**: Profile AI model loading and inference
- **Frontend**: Use React.memo() for expensive components
- **Database**: Add indexes for slow queries
- **API**: Implement pagination for large datasets

## Security Guidelines

- Never commit secrets (API keys, passwords) to repository
- Use environment variables for configuration
- Validate and sanitize all user inputs
- Follow OWASP security best practices
- Report security vulnerabilities privately

## Getting Help

- Check existing [documentation](README.md)
- Search [closed issues](https://github.com/yourusername/EduVision-enhanced/issues?q=is%3Aissue+is%3Aclosed)
- Ask questions in [Discussions](https://github.com/yourusername/EduVision-enhanced/discussions)
- Contact maintainers: ahmed@example.com

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to EduVision! 🎓✨
