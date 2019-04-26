// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';

import Plot from 'react-plotly.js'
import dummy_data from "../../../../example_data/results-log.json"

class ExperimentPlots extends React.Component { // eslint-disable-line react/prefer-stateless-function
  generate_dummy_data() {
    this.data = dummy_data
  }

  get_data() {
    console.log(this.props)
    var image_path = [];
    var experiments = [];

    try {
      image_path = this.props.images.img_path.experiments;
      experiments = this.props.images.imgs.experiments;
    } catch (e) {
      if (e instanceof TypeError) {
        console.log("ExperimentImages not initialized yet")
      } else {
        console.log(e)
      }
    }

    return {"image_path": image_path, "experiments": experiments}
  }

  parse_data() {
    var plots = new Array()
    for (var i = 0; i < this.data.length; i++) {
      var entry = this.data[i];
      var key = Object.keys(entry)[0];
      var x = entry[key]["epoch"];
      var y = entry[key]["data"];
      var label = entry[key]["label"];

      // create structure
      if (!(label in plots)) {
        plots[label] = [];
      }
      if (!(key in plots[label])) {
        plots[label][key] = {
          x: [],
          y: [],
          name: key,
          type: 'scatter',
          mode: 'lines+points',
        };
      }

      // fill with values
      plots[label][key]["x"].push(x);
      plots[label][key]["y"].push(y);
    }
    return plots;
  }

  create_plots() {
    var data = this.parse_data();
    var plot_names = Object.keys(data);
    var plots = plot_names.map((plot_name) => {
      var plot_data = data[plot_name];
      var line_names = Object.keys(plot_data);
      var lines = [];
      for (var i=0;i<line_names.length;i++) {
        var line_data = plot_data[line_names[i]];
        lines.push(line_data)
      }

      return (<Plot
        data={lines}
        key={"plot_"+plot_name}
        layout={{title: plot_name}}
      />)

    });
    return plots;
  }

  render() {
    this.generate_dummy_data();
    return (
      <div>
        <h1>Plots</h1>
        {this.create_plots()}
      </div>
    );
  }
}

export default ExperimentPlots;
