/*
 * HomePage
 *
 * This is the first thing users see of our App, at the '/' route
 */

import React from 'react';
import PropTypes from 'prop-types';
import { Helmet } from 'react-helmet';
import './style.scss';

export default class OverviewPage extends React.PureComponent { // eslint-disable-line react/prefer-stateless-function
  /**
   * when initial state username is not null, submit the form to load repos
   */
  render() {

    return (
      <div>
        <Helmet>
          <title>Overview</title>
          <meta name="description" content="A React.js Boilerplate application homepage" />
        </Helmet>
        <div className="home-page">
          <section className="centered">
            <h2>Klaus</h2>
          </section>
          <section>
            <h2>Laura</h2>
          </section>
        </div>
      </div>
    );
  }
}