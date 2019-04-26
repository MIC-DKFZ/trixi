// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import DynamicTable from "../DynamicTable";

class ExperimentResults extends React.Component { // eslint-disable-line react/prefer-stateless-function

   get_data() {
    var data = [];
    var dummy_content = {
      "name": ["No data available"],
    };
    try {
      var data = this.props.results.results;
      data["name"] = this.props.results.exps;
      if (typeof data === 'undefined') {
        data = dummy_content
      }
    } catch (e) {
      if (e instanceof TypeError) {
        // console.log("not initialized yet...")
        data = dummy_content;
      } else {
        console.log(e)
      }
    }
    return data
  }

  render() {
    var data = this.get_data();
    return (
      <div>
        <h1>Results</h1>
        <DynamicTable key="experiment_logs_dynamic_table" data={data}/>
      </div>
    )
  }
}

export default ExperimentResults;
