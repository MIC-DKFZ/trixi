// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import DynamicTable from "../DynamicTable";

class ExperimentLogs extends React.Component { // eslint-disable-line react/prefer-stateless-function
  render() {
    var data = this.props.logs;
    console.log(data)
    return (
      <div>
        <h1>Experiment Logs</h1>
        <DynamicTable key="experiment_logs_dynamic_table" data={data}/>
      </div>
    )
  }
}

export default ExperimentLogs;
