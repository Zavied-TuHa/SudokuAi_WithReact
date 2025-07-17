import { useState, useRef } from 'react';
import SudokuBoard from './SudokuBoard';
import { solvePuzzles } from '../utils/api';

const SolverUI = ({ config }) => {
  const [puzzle, setPuzzle] = useState(Array(81).fill('')); // Input puzzle
  const [algorithm, setAlgorithm] = useState('rule-based');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [stats, setStats] = useState(null);
  const [solvedPuzzle, setSolvedPuzzle] = useState(null);
  const inputRefs = useRef([]);

  const handleInputChange = (index, value) => {
    if (!/^[0-9.]$/.test(value) && value !== '') return;
    const newPuzzle = [...puzzle];
    newPuzzle[index] = value;
    setPuzzle(newPuzzle);
  };

  const handleKeyDown = (e, index) => {
    const row = Math.floor(index / 9);
    const col = index % 9;
    let newIndex = index;

    if (e.key === 'ArrowUp' && row > 0) newIndex = index - 9;
    if (e.key === 'ArrowDown' && row < 8) newIndex = index + 9;
    if (e.key === 'ArrowLeft' && col > 0) newIndex = index - 1;
    if (e.key === 'ArrowRight' && col < 8) newIndex = index + 1;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      handleInputChange(index, '');
      return;
    }

    if (newIndex !== index && inputRefs.current[newIndex]) {
      inputRefs.current[newIndex].focus();
    }
  };

  const handleSolve = async () => {
    setLoading(true);
    setMessage('');
    setMetrics(null);
    setStats(null);
    setSolvedPuzzle(null);

    try {
      const puzzleString = puzzle.map(cell => cell === '' ? '0' : cell).join('').replace(/[^0-9.]/g, '.');
      if (puzzleString.length !== 81) {
        setMessage('Puzzle must be 81 characters long');
        setLoading(false);
        return;
      }
      const response = await solvePuzzles(puzzleString, algorithm);
      setMetrics(response.metrics);
      setStats(response.stats);
      setSolvedPuzzle(response.solved_puzzle);
      setMessage(response.message || 'Puzzle solved successfully!');
    } catch (e) {
      setMessage(`Error: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setPuzzle(Array(81).fill(''));
    setMessage('');
    setMetrics(null);
    setStats(null);
    setSolvedPuzzle(null);
  };

  const handleAlgorithmChange = (e) => {
    setAlgorithm(e.target.value);
    setMessage('');
    setMetrics(null);
    setStats(null);
    setSolvedPuzzle(null);
  };

  return (
    <div className="min-h-screen p-6" style={{ backgroundColor: config.ui?.colors?.background }}>
      <h2 className="text-3xl font-bold mb-6" style={{ color: config.ui?.colors?.text }}>Sudoku Solver</h2>

      <div className="mb-6 bg-white p-4 rounded-lg shadow-md">
        <label className="block mb-2 font-semibold" style={{ color: config.ui?.colors?.text }}>Algorithm</label>
        <select
          className="w-full p-2 border rounded"
          value={algorithm}
          onChange={handleAlgorithmChange}
        >
          <option value="rule-based">Rule-Based</option>
          <option value="prob-logic">Probabilistic Logic</option>
          <option value="random-forest">Random Forest</option>
        </select>
      </div>

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2" style={{ color: config.ui?.colors?.text }}>Input Puzzle</h3>
        <div className="grid grid-cols-9 gap-0 w-96 h-96 mx-auto">
          {puzzle.map((cell, index) => {
            const row = Math.floor(index / 9);
            const col = index % 9;
            const isSubgridBorder = (row + 1) % 3 === 0 && row < 8;
            const isRightSubgridBorder = (col + 1) % 3 === 0 && col < 8;
            return (
              <input
                key={index}
                type="text"
                maxLength="1"
                value={cell}
                onChange={(e) => handleInputChange(index, e.target.value)}
                onKeyDown={(e) => handleKeyDown(e, index)}
                ref={(el) => (inputRefs.current[index] = el)}
                className="flex items-center justify-center border text-center"
                style={{
                  borderColor: config.ui?.colors?.grid_lines,
                  borderBottomWidth: isSubgridBorder ? '2px' : '1px',
                  borderRightWidth: isRightSubgridBorder ? '2px' : '1px',
                  backgroundColor: config.ui?.colors?.background,
                  color: config.ui?.colors?.text,
                  fontSize: '1.5rem',
                  width: '42px',
                  height: '42px'
                }}
              />
            );
          })}
        </div>
      </div>

      <div className="flex gap-4 mb-6">
        <button
          className="px-6 py-3 rounded-lg text-white font-semibold hover:bg-green-600"
          style={{ backgroundColor: config.ui?.colors?.button_bg }}
          onClick={handleSolve}
          disabled={loading}
        >
          Solve
        </button>
        <button
          className="px-6 py-3 rounded-lg text-white font-semibold bg-red-500 hover:bg-red-600"
          onClick={handleClear}
          disabled={loading}
        >
          Clear
        </button>
      </div>

      {message && (
        <p className="text-lg mb-4" style={{ color: config.ui?.colors?.text }}>{message}</p>
      )}

      {loading && (
        <div className="fixed inset-0 flex items-center justify-center" style={{ backgroundColor: config.ui?.colors?.loading_bg }}>
          <p className="text-xl font-semibold" style={{ color: config.ui?.colors?.loading_text }}>Loading...</p>
        </div>
      )}

      {(puzzle.join('') || solvedPuzzle) && (
        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          {puzzle.join('') && (
            <div>
              <h3 className="text-xl font-semibold mb-2" style={{ color: config.ui?.colors?.text }}>Input Puzzle</h3>
              <SudokuBoard puzzle={puzzle.map(cell => cell === '' ? '.' : cell).join('')} config={config} />
            </div>
          )}
          {solvedPuzzle && (
            <div>
              <h3 className="text-xl font-semibold mb-2" style={{ color: config.ui?.colors?.text }}>Solved Puzzle</h3>
              <SudokuBoard puzzle={solvedPuzzle} config={config} />
            </div>
          )}
        </div>
      )}

      {metrics && stats && (
        <div className="mb-6 bg-white p-4 rounded-lg shadow-md">
          <h3 className="text-xl font-semibold mb-2" style={{ color: config.ui?.colors?.text }}>Evaluation Metrics</h3>
          <p style={{ color: config.ui?.colors?.text }}>Accuracy: {(metrics.accuracy * 100).toFixed(2)}%</p>
          <p style={{ color: config.ui?.colors?.text }}>Solve Time: {metrics.solve_time.toFixed(4)}s</p>
          <p style={{ color: config.ui?.colors?.text }}>Memory Usage: {metrics.memory_usage.toFixed(2)} MB</p>
          {stats[algorithm.replace('-', '_')] && (
            <>
              {algorithm === 'rule-based' && (
                <>
                  <p style={{ color: config.ui?.colors?.text }}>Steps Taken: {stats.rule_based.steps_taken}</p>
                  <p style={{ color: config.ui?.colors?.text }}>Rules Triggered: {stats.rule_based.rules_triggered}</p>
                  <p style={{ color: config.ui?.colors?.text }}>Time per Step: {stats.rule_based.time_per_step.toFixed(6)}s</p>
                  <p style={{ color: config.ui?.colors?.text }}>Fail Rate: {(stats.rule_based.fail_rate * 100).toFixed(2)}%</p>
                </>
              )}
              {algorithm === 'prob-logic' && (
                <>
                  <p style={{ color: config.ui?.colors?.text }}>Iterations to Converge: {stats.prob_logic.iterations_to_converge}</p>
                  <p style={{ color: config.ui?.colors?.text }}>Number of Conflicts: {stats.prob_logic.number_of_conflicts}</p>
                  <p style={{ color: config.ui?.colors?.text }}>Probability Confidence: {stats.prob_logic.probability_confidence.toFixed(4)}</p>
                </>
              )}
              {algorithm === 'random-forest' && (
                <>
                  <p style={{ color: config.ui?.colors?.text }}>Inference Time: {stats.rf.inference_time.toFixed(4)}s</p>
                  <p style={{ color: config.ui?.colors?.text }}>Prediction Accuracy: {(stats.rf.prediction_accuracy * 100).toFixed(2)}%</p>
                  <p style={{ color: config.ui?.colors?.text }}>Model Size: {stats.rf.model_size.toFixed(2)} MB</p>
                  <p style={{ color: config.ui?.colors?.text }}>Failed Attempts: {stats.rf.failed_attempts || 0}</p>
                  <p style={{ color: config.ui?.colors?.text }}>Empty Cells Detected: {stats.rf.empty_cells_detected || 0}</p>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default SolverUI;