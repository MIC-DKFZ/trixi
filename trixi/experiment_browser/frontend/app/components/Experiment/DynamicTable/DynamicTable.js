// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import {Table} from 'react-bootstrap';

class DynamicTable extends React.Component { // eslint-disable-line react/prefer-stateless-function

   get_data() {

    var data = [];
    var dummy_content = {
      "name": ["No data available"],
    };
    try {
      data = this.props.data;
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
    var keys = Object.keys(this.get_data());

    // fill values
    var values = new Array();

    for (var key_idx in keys) {
      var key = keys[key_idx];
      var value = "";
      if (typeof this.get_data()[key] !== 'undefined') {
        value = this.get_data()[key];
      }
      // if key occurs the first time
      if (!(key in values)) {
        values[key] = []
      }
      values[key].push(value)
    }

    return values;
  }

  generate_table_header() {
    var output = this.get_data().name.map((row_name) => {
      var line = <th key={row_name}>{row_name}</th>;
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
    )
  }
}

export default DynamicTable;
