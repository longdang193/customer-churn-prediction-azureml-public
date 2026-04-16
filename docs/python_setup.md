# Python 3.9 Environment Setup Guide

This guide explains how to set up a Python 3.9 development environment for this project. Using Python 3.9 ensures compatibility with the Docker image (`python:3.9-slim`) and prevents version mismatches when compiling requirements.

## Why Python 3.9?

- **Version consistency**: Ensures requirements compile correctly for production
- **Prevents conflicts**: Avoids package version issues between dev and Docker environments

## Prerequisites

- System with package manager access (apt, yum, brew, etc.)
- Administrator/sudo access (for system-wide Python installation)

## Installation by Platform

### Windows

1. **Download Python 3.9** from [python.org](https://www.python.org/downloads/release/python-3913/)
2. **Run installer** and check "Add Python to PATH"
3. **Verify installation**:

   ```cmd
   python --version
   # Should output: Python 3.9.x
   ```

### Ubuntu/Debian

1. **Add deadsnakes PPA** (provides multiple Python versions):

   ```bash
   sudo apt-get update
   sudo apt-get install -y software-properties-common
   sudo add-apt-repository -y ppa:deadsnakes/ppa
   sudo apt-get update
   ```

2. **Install Python 3.9**:

   ```bash
   sudo apt-get install -y python3.9 python3.9-venv python3.9-dev
   ```

3. **Verify installation**:

   ```bash
   python3.9 --version
   # Should output: Python 3.9.x
   ```

### macOS

**Option 1: Using Homebrew** (recommended):

```bash
brew install python@3.9
python3.9 --version
```

**Option 2: Using pyenv** (works on macOS and Linux):

```bash
# Install pyenv if not already installed
brew install pyenv  # Linux: see https://github.com/pyenv/pyenv#installation

# Install Python 3.9
pyenv install 3.9.25

# Set as local version (optional)
pyenv local 3.9.25
```

## Setting Up Virtual Environment

Once Python 3.9 is installed, create a virtual environment:

### Linux/macOS

```bash
# Create virtual environment with Python 3.9
python3.9 -m venv venv

# Activate the environment
source venv/bin/activate

# Verify Python version
python --version  # Should show: Python 3.9.x
which python      # Should point to venv/bin/python
```

### Windows

```cmd
# Create virtual environment with Python 3.9
python -m venv venv

# Activate the environment
venv\Scripts\activate

# Verify Python version
python --version  # Should show: Python 3.9.x
```

## Installing pip-tools

With the virtual environment activated:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install pip-tools
pip install pip-tools

# Verify installation
pip-compile --version
```

## Compiling Requirements

Now you can compile requirements with Python 3.9:

```bash
# Make sure venv is activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Compile requirements first (ensures compatibility with Python 3.9)
pip-compile requirements.in -o requirements.txt

# Compile dev-requirements with constraints from requirements.txt
# This ensures shared dependencies (matplotlib, scipy, etc.) use compatible versions
pip-compile dev-requirements.in -o dev-requirements.txt --constraint requirements.txt
```

**Important**: Compiling with Python 3.9 ensures all pinned versions are compatible with the Docker image. If you compile with a different Python version, you may encounter package version conflicts during Docker builds.

## Installing Dependencies

After compiling requirements:

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r dev-requirements.txt
```

## Using the Environment

### Activate the Environment

**Linux/macOS:**

```bash
source venv/bin/activate
```

**Windows:**

```cmd
venv\Scripts\activate
```

### Deactivate the Environment

```bash
deactivate
```

### Verify You're Using Python 3.9

```bash
python --version  # Should show: Python 3.9.x
which python      # Should point to venv/bin/python (or venv\Scripts\python on Windows)
```

## Project Configuration

### .python-version File

The project includes a `.python-version` file that specifies Python 3.9. This file is used by:

- **pyenv**: Automatically switches to Python 3.9 when entering the project directory
- **Documentation**: Clearly indicates the required Python version

If using pyenv, the version will be automatically set when you `cd` into the project directory.

## Troubleshooting

### Python 3.9 Not Found

**Ubuntu/Debian:**

- Ensure deadsnakes PPA is added: `sudo add-apt-repository ppa:deadsnakes/ppa`
- Update package list: `sudo apt-get update`
- Try installing again: `sudo apt-get install python3.9`

**macOS:**

- Check Homebrew: `brew list | grep python`
- Reinstall: `brew reinstall python@3.9`

**Windows:**

- Verify Python is in PATH: `python --version`
- Reinstall Python 3.9 from python.org

### Virtual Environment Issues

**Problem**: `python3.9 -m venv venv` fails

**Solution**: Ensure `python3.9-venv` package is installed (Ubuntu/Debian):

```bash
sudo apt-get install python3.9-venv
```

**Problem**: Wrong Python version in venv

**Solution**: Delete and recreate the virtual environment:

```bash
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
python --version  # Verify it's 3.9.x
```

### Requirements Compilation Issues

**Problem**: Packages require Python >=3.11

**Solution**: This means requirements were compiled with a newer Python version. Recompile with Python 3.9:

```bash
source venv/bin/activate
pip-compile requirements.in -o requirements.txt
```

**Problem**: Docker build fails with version conflicts

**Solution**: Ensure requirements were compiled with Python 3.9, not a newer version. Check the header of `requirements.txt` - it should show "Python 3.9" not "Python 3.12".

## Best Practices

1. **Always use Python 3.9** for this project (matches Dockerfile)
2. **Activate venv** before working on the project
3. **Recompile requirements** if you add new packages to `.in` files
4. **Keep venv in .gitignore** (already configured)
5. **Document Python version** in `.python-version` file (already created)

## Quick Reference

```bash
# Setup (one-time)
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip pip-tools
pip-compile requirements.in -o requirements.txt
pip-compile dev-requirements.in -o dev-requirements.txt
pip install -r requirements.txt

# Daily use
source venv/bin/activate  # Activate environment
# ... work on project ...
deactivate                # Deactivate when done
```
