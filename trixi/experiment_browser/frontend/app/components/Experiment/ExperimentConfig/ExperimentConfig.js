// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import {Table} from 'react-bootstrap';

class ExperimentConfig extends React.Component { // eslint-disable-line react/prefer-stateless-function

  get_data() {

    var data = [];
    var dummy_content = {
      "name": ["No experiment available"],
    };
    try {
      data = this.props.config.configs;
      if (typeof data === 'undefined') {
        data = dummy_content
      }
    } catch (e) {
      if (e instanceof TypeError) {
        console.log("not initialized yet...")
        data = dummy_content;
      } else {
        console.log(e)
      }
    }
    return data
  }

  extract_table_content() {
    // finding all keys
    var all_experiment_keys = Object.keys(this.get_data());

    // fill values
    var experiment_values = new Array();

    for (var key_idx in all_experiment_keys) {
      var key = all_experiment_keys[key_idx];
      var experiment_value = "";
      if (typeof this.get_data()[key] !== 'undefined') {
        experiment_value = this.get_data()[key];
      }
      // if key occurs the first time
      if (!(key in experiment_values)) {
        experiment_values[key] = []
      }
      experiment_values[key].push(experiment_value)
    }

    return experiment_values;
  }

  generate_table_header() {
    var output = this.get_data().name.map((experiment_name) => {
      var line = <th key={experiment_name}>{experiment_name}</th>;
      return line;
    });
    return output;


  }

  generate_table_body() {
    var table_body = [];
    var table_content = this.extract_table_content();

    for (var row_key in table_content) {
      var row_data = [];
      var row_content = table_content[row_key];

      row_data.push(<td key={"td_header" + row_key}>{row_key}</td>);
      row_content.map((data, i) => {
        row_data.push(<td key={row_key + "_" + i}>{data}</td>)
      });
      var row = <tr key={"td_header" + row_key}>{row_data}</tr>;
      table_body.push(row)
    }
    return table_body;
  }

  render() {
    return (
      <div>
        <h1>Configs</h1>
        <Table responsive="sm" striped="true">
          <thead>
          <tr>
            <th>#</th>
            {this.generate_table_header()}
          </tr>
          </thead>
          <tbody>
          {this.generate_table_body()}
          </tbody>
        </Table>
      </div>
    )
  }
}

export default ExperimentConfig;
