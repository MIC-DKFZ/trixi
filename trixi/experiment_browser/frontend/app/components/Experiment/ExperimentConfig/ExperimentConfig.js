// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import {Table} from 'react-bootstrap';
import DynamicTable from "../DynamicTable";

class ExperimentConfig extends React.Component { // eslint-disable-line react/prefer-stateless-function

  render() {
    var data = this.props.config.configs;
    return (
      <div>
        <h1>Configs</h1>
        <DynamicTable key="experiment_config_dynamic_table" data={data}/>
      </div>
    )
  }
}

export default ExperimentConfig;
