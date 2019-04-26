/**
 *
 * App
 *
 * This component is the skeleton around the actual pages, and should only
 * contain code that should be seen on all pages. (e.g. navigation bar)
 */

import React from 'react';
import { Helmet } from 'react-helmet';
import { Switch, Route } from 'react-router-dom';

import HomePage from 'containers/HomePage/Loadable';
import OverviewPage from 'containers/OverviewPage/Loadable';
import FeaturePage from 'containers/FeaturePage/Loadable';
import ExperimentPage from 'containers/ExperimentPage/Loadable'
import NotFoundPage from 'containers/NotFoundPage/Loadable';
import Header from 'components/Header';
import Footer from 'components/Footer';
import Sidebar from 'components/Sidebar'
import './style.scss';

const App = () => (
  <div className="app-wrapper">
    <Helmet
      titleTemplate="%s - TRIXI"
      defaultTitle="TRIXI Experimentbrowser"
    >
      <meta name="description" content="An experiment browser for pytorch experiments." />
    </Helmet>
    <Header />
    <Sidebar/>
    <Switch>
      <Route exact path="/" component={HomePage} />
      <Route path="/features" component={FeaturePage} />
      <Route path="/experiment" component={ExperimentPage} />
      <Route path="/overview" component={OverviewPage} />
      <Route path="" component={NotFoundPage} />
    </Switch>
    <Footer />
  </div>
);

export default App;
