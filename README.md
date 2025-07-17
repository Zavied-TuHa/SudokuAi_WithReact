# Sudoku AI with React

A Sudoku solver application built with Python (FastAPI) for the backend and React for the frontend. Supports three solving algorithms: Rule-Based, Probabilistic Logic, and Random Forest.

## Features
- Solve Sudoku puzzles using different algorithms.
- Display evaluation metrics (accuracy, solve time, memory usage).
- User-friendly interface with React.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Zavied-TuHa/SudokuAi_WithReact.git
   cd SudokuAi_WithReact
    ```
2. Set up the Python environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3. Install frontend dependencies:
    ```bash
    cd web
    npm install
    npm run build
    ```
4. Run the application:
    ```bash
    cd ..
    python src/main.py
    ```
5. Access the app at http://localhost:8000.

## Project Structure
    src/: Backend Python code (FastAPI, solvers, data processing).
    web/: Frontend React code.
    config/: Configuration files (YAML).
    data/: Data files (excluded from Git).
## Requirements
    Python 3.8+
    Node.js 16+
    8GB RAM (optimized for low-memory systems)