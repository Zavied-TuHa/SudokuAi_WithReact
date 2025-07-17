import { useState, useEffect } from 'react';
import SolverUI from './components/SolverUI';

const App = () => {
  const [config, setConfig] = useState({});

  useEffect(() => {
    setConfig({
      ui: {
        colors: {
          background: '#f0f4f8',
          grid_lines: '#333333',
          highlighted_cell: '#bbdefb',
          text: '#000000',
          button_bg: '#4caf50',
          button_active: '#45a049',
          loading_bg: '#e0e0e0',
          loading_text: '#333333'
        }
      }
    });
  }, []);

  return <SolverUI config={config} />;
};

export default App;