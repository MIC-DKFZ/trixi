/*
 * HomePage
 *
 * This is the first thing users see of our App, at the '/' route
 */

import React from 'react';
import BootstrapTable from 'react-bootstrap-table-next';
import Button from 'react-bootstrap/lib/Button';
import Modal from 'react-bootstrap/lib/Modal';
import filterFactory, { textFilter } from 'react-bootstrap-table2-filter';
import PropTypes from 'prop-types';
import { Helmet } from 'react-helmet';
import './style.scss';


export default class OverviewPage extends React.PureComponent { // eslint-disable-line react/prefer-stateless-function
  /**
   * when initial state username is not null, submit the form to load repos
   */

  constructor(props, context) {
    super(props, context);

    this.handleShow = this.handleShow.bind(this);
    this.handleClose = this.handleClose.bind(this);

    this.state = {
      show: false,
    };
  }

  handleClose() {
    this.setState({ show: false });
  }

  handleShow() {
    this.setState({ show: true });
  }

  render() {     
    const baseInfo = { ccols1: ['another_entry', 'batch_size', 'num_epochs'], ccols2: ['description', 'patch_size'], rcols: ['auc', 'dice', 'train', 'val'], rows: [['experiment_browser/experiment1', false, 'experiments', '20180221-163831', 'Finished', '-', [['-', '-'], ['2', '2'], ['1000', '1000'], ['This is only to test the experiment browser', 'This is only to test the '], ['(128, 128, 128)', '(128, 128, 128)']], [['0.07968243956565857', '0.07968243956565857'], ['0.22086010873317719', '0.22086010873317719'], ['0.051016926765441895', '0.051016926765441895'], ['0.35155946016311646', '0.35155946016311646']]], ['experiment_browser/experiment2', false, 'experiments', '20180221-163831', 'Error', '-', [['something something', 'something something'], ['3', '3'], ['2000', '2000'], ['This is only to test the experiment browser', 'This is only to test the '], ['(128, 128, 128)', '(128, 128, 128)']], [['0.07968243956565857', '0.07968243956565857'], ['0.22086010873317719', '0.22086010873317719'], ['0.051016926765441895', '0.051016926765441895'], ['0.35155946016311646', '0.35155946016311646']]]], noexp: [] }

    const products = [{
      directory: '1',
      name: 'Laura',
      time: 'Jonas',
      state: 10,
      epochs: '10'
    }, {
      directory: '1',
      name: 'Laura',
      time: 'Jonas',
      state: 10,
      epochs: '12'
    }, {
      directory: '1',
      name: 'Laura',
      time: 'Jonas',
      state: 10,
      epochs: '11'
    }];
    const columns = [{
      dataField: 'directory',
      text: 'Directory',
      sort: true,
      filter: textFilter()
    }, {
      dataField: 'name',
      text: 'Name',
      sort: true,
      filter: textFilter()
    }, {
      dataField: 'time',
      text: 'Time',
      sort: true,
      filter: textFilter()
    }, {
      dataField: 'state',
      text: 'State',
      sort: true,
      filter: textFilter()
    }, {
      dataField: 'epochs',
      text: 'Epochs',
      sort: true,
      filter: textFilter()
    }];

    const buttonPaddingStyle = {
      paddingLeft: '4px',
    };

    const directory = '/home/klaus';


    return (
      <div>
        <Helmet>
          <title>Overview</title>
          <meta name="description" content="A React.js Boilerplate application homepage" />
        </Helmet>
        <div className="home-page">
          <section className="centered">
            <h2>Klaus</h2>

            <BootstrapTable keyField="idasdf" data={products} columns={columns} filter={filterFactory()} />

            <div className="container-fluid">
              <div className="row text-left">
                <div className="col-auto" style={{ paddingRight: '2px' }}>
                  <a role="button" className="btn btn-outline-secondary toggle-config">Toggle all</a>
                </div>
                <div className="col" style={buttonPaddingStyle}>
                  <div>
                    {baseInfo.ccols1.map((name) => {
                      return <Button key={name} variant="outline-success">{name}</Button>;
                    })}
                  </div>
                  <div>
                    {baseInfo.ccols2.map((name) => {
                      return <Button key={name} variant="outline-success">{name}</Button>;
                    })}
                  </div>
                </div>
              </div>
              <div className="row text-left">
                <div className="col-auto" style={{ paddingRight: '2px' }}>
                  <a role="button" className="btn btn-outline-secondary toggle-config">Toggle all</a>
                </div>
                <div className="col" style={buttonPaddingStyle}>
                  {baseInfo.rcols.map((name) => {
                    return <Button key={name} variant="outline-success">{name}</Button>;
                  })}
                </div>
              </div>
              <div className="row" style={{ marginTop: '5px' }}>
                <div className="col-auto">
                  <button type="button" className="btn btn-outline-secondary" onClick={this.handleShow}>
                    Switch to...
                  </button>
                </div>
              </div>
            </div>
          </section>
          <section>
            <Modal show={this.state.show} onHide={this.handleClose}>
              <Modal.Header closeButton>
                <Modal.Title>Switch to another directory</Modal.Title>
              </Modal.Header>
              <Modal.Body>Woohoo, you're reading this text in a modal!</Modal.Body>
              <Modal.Footer>
                <Button variant="secondary" onClick={this.handleClose}>
                  Close
                </Button>
                <Button variant="primary" onClick={this.handleClose}>
                  Save Changes
                </Button>
              </Modal.Footer>
            </Modal>
          </section>
        </div>
      </div>
    );
  }
}
