// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import {Table} from 'react-bootstrap';

//TODO give unique identifier to all elements!
class ExperimentConfig extends React.Component { // eslint-disable-line react/prefer-stateless-function
  generate_dummy_data() {
    var experiments = [
      {
        config: {
          "name": "test_experiment1",
          "description": "This is only to test the experiment browser",
          "time": "20180221-163831",
          "state": "Finished",
          "patch_size": "__tuple__((128, 128, 128))",
          "batch_size": 2,
          "num_epochs": 1000
        },
        results: {
          "train": 0.051016926765441895,
          "val": 0.35155946016311646,
          "dice": 0.22086010873317719,
          "auc": 0.07968243956565857,
          "expval1": 0.000001,
        }
      },
      {
        config: {
          "name": "test_experiment2",
          "description": "This is only to test the experiment browser",
          "time": "20180221-163831",
          "state": "Error",
          "patch_size": "__tuple__((128, 128, 128))",
          "batch_size": 3,
          "num_epochs": 2000,
          "another_entry": "something something"
        },
        results: {
          "train": 0.051016926765441895,
          "val": 0.35155946016311646,
          "dice": 0.22086010873317719,
          "auc": 0.07968243956565857,
          "exp2val": 0.999999999,
        }
      },
      {
        config: {
          "name": "test_experiment3",
          "description": "This is only to test the experiment browser",
          "time": "20180221-163831",
          "state": "Error",
          "patch_size": "__tuple__((128, 128, 128))",
          "batch_size": 3,
          "num_epochs": 2000,
          "another_entry": "something something"
        },
        results: {
          "train": 0.051016926765441895,
          "val": 0.35155946016311646,
          "dice": 0.22086010873317719,
          "auc": 0.07968243956565857,
          "exp3val": 0.999999999,
        }
      }
    ];
    this.data = experiments
  }

  extract_table_content() {
    // finding all keys
    var all_experiment_keys = [];
    this.data.map((experiment) => {
      for (var key in experiment["results"]) {
        all_experiment_keys.push(key)
      }
    });
    var key_unique = [...new Set(all_experiment_keys)];

    // fill values
    var experiment_values = new Array();
    this.data.map((experiment) => {
      for (var i=0;i<key_unique.length;i++) {
        var key = key_unique[i];
        // if value exists, use it
        var experiment_value = "";
        if (typeof experiment["results"][key] !== 'undefined') {
          experiment_value = experiment["results"][key];
        }
        // if key occurs the first time
        if (!(key in experiment_values)) {
            experiment_values[key] = []
        }
        // insert value
        experiment_values[key].push(experiment_value)
      }
    });
    return experiment_values;
  }

  generate_table_header() {
    var output = this.data.map((experiment) => {
      var experiment_name = experiment["config"]["name"];
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

      row_data.push(<td key={"td_header" +row_key}>{row_key}</td>);
      row_content.map((data, i) => {
        row_data.push(<td key={row_key + "_" + i}>{data}</td>)
      });
      var row = <tr key={"td_header" +row_key}>{row_data}</tr>;
      table_body.push(row)
    }
    return table_body;
  }

  render() {
    this.generate_dummy_data();
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
