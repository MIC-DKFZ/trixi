/*
 * HomePage
 *
 * This is the first thing users see of our App, at the '/' route
 */

import React from 'react';
import PropTypes from 'prop-types';
import { Helmet } from 'react-helmet';

import ExperimentConfig from "../../components/Experiment/ExperimentConfig/ExperimentConfig";
import ExperimentImages from "../../components/Experiment/ExperimentImages/ExperimentImages";
import ExperimentResults from "../../components/Experiment/ExperimentResults/ExperimentResults";
import ExperimentPlots from "../../components/Experiment/ExperimentPlots/ExperimentPlots";
import axios from "axios/index";

export default class HomePage extends React.PureComponent { // eslint-disable-line react/prefer-stateless-function
  /**
   * when initial state username is not null, submit the form to load repos
   */

  constructor(props) {
    super(props);
    var dummy_obj = {
      config: {},
      images: {},
      logs: {},
      plots: {}
    }

    this.state = {experiment_result: dummy_obj};
  }

  componentDidMount() {
    if (this.props.username && this.props.username.trim().length > 0) {
      this.props.onSubmitForm();
    }

    var upper_obj = this;
    var url = "http://localhost:5000/get_experiment?dir=test_klaus&exp=test_klaus/experiment1";
    axios.get(url)
      .then((res) => {
        upper_obj.setState({experiment_result: res.data})
      });
  }

  render() {
    const { loading, error, repos } = this.props;
    const reposListProps = {
      loading,
      error,
      repos,
    };

    return (
      <article>
        <Helmet>
          <title>Home Page</title>
          <meta name="description" content="A React.js Boilerplate application homepage" />
        </Helmet>
        <div className="home-page">
          <section className="centered">
            <h2>EXPERIMENT RESULTS FOR {this.state.title}</h2>
          </section>
          <ExperimentConfig config={this.state.experiment_result.config}/>
          <ExperimentImages images={this.state.experiment_result.images}/>
          <ExperimentResults results={this.state.experiment_result.results}/>
          <ExperimentPlots plots={this.state.experiment_result.plots}/>
        </div>
      </article>
    );
  }
}

HomePage.propTypes = {
  loading: PropTypes.bool,
  error: PropTypes.oneOfType([
    PropTypes.object,
    PropTypes.bool,
  ]),
  repos: PropTypes.oneOfType([
    PropTypes.array,
    PropTypes.bool,
  ]),
  onSubmitForm: PropTypes.func,
  username: PropTypes.string,
  onChangeUsername: PropTypes.func,
};
