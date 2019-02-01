// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';

import Plot from 'react-plotly.js'

class ExperimentPlots extends React.Component { // eslint-disable-line react/prefer-stateless-function
  render() {
    return (
      <Plot
        data={[
          {
            x: [1, 2, 3],
            y: [2, 6, 3],
            type: 'scatter',
            mode: 'lines+points',
            marker: {color: 'red'},
          },
          {type: 'bar', x: [1, 2, 3], y: [2, 5, 3]},
        ]}
        layout={{width: 320, height: 240, title: 'A Fancy Plot'}}
      />
    )
  }
}

export default ExperimentPlots;
