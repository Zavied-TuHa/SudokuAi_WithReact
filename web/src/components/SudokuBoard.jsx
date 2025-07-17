const SudokuBoard = ({ puzzle, config }) => {
  // Convert string to 9x9 grid if puzzle is a string
  const grid = typeof puzzle === 'string'
    ? Array(9).fill().map((_, i) =>
        puzzle.slice(i * 9, (i + 1) * 9).split('').map(c => c === '.' || c === '0' ? 0 : parseInt(c))
      )
    : puzzle;

  return (
    <div className="flex justify-center">
      <div className="grid grid-cols-9 gap-0 w-96 h-96">
        {grid.flat().map((cell, index) => {
          const row = Math.floor(index / 9);
          const col = index % 9;
          const isSubgridBorder = (row + 1) % 3 === 0 && row < 8;
          const isRightSubgridBorder = (col + 1) % 3 === 0 && col < 8;
          return (
            <div
              key={index}
              className="flex items-center justify-center border"
              style={{
                borderColor: config.ui?.colors?.grid_lines,
                borderBottomWidth: isSubgridBorder ? '2px' : '1px',
                borderRightWidth: isRightSubgridBorder ? '2px' : '1px',
                backgroundColor: cell === 0 ? config.ui?.colors?.background : config.ui?.colors?.highlighted_cell,
                color: config.ui?.colors?.text,
                fontSize: '1.5rem'
              }}
            >
              {cell === 0 ? '' : cell}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default SudokuBoard;