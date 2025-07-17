export const solvePuzzles = async (puzzle, algorithm) => {
  const response = await fetch('/api/solve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ puzzle, algorithm })
  });
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Failed to solve puzzle');
  }
  return response.json();
};