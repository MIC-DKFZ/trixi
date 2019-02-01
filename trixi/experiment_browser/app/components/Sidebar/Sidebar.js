// Example: https://trendmicro-frontend.github.io/react-sidenav/
import React from 'react';
import { Link } from 'react-router-dom';
import { Switch, Route, Router} from 'react-router-dom';
import createBrowserHistory from "history/createBrowserHistory"
import history from '../../history'


import SideNav, { Toggle, Nav, NavItem, NavIcon, NavText } from '@trendmicro/react-sidenav';

import '@trendmicro/react-sidenav/dist/react-sidenav.css';

class Sidebar extends React.Component { // eslint-disable-line react/prefer-stateless-function
  render() {
    return (
      <SideNav
        onSelect={(selected) => {
          const to = '/' + selected;
          if (location.pathname !== to) {

              history.push(to);
          }
        }}
      >
        <SideNav.Toggle />
        <SideNav.Nav defaultSelected="home">
          <NavItem eventKey="">
            <NavIcon>
              <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
            </NavIcon>
            <NavText>
              Home
            </NavText>
          </NavItem>
          <NavItem eventKey="experiment">
            <NavIcon>
              <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
            </NavIcon>
            <NavText>
              Experiment
            </NavText>
            <NavItem eventKey="experiment">
              <NavIcon>
                <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
              </NavIcon>
              <NavText>
                Config
              </NavText>
            </NavItem>
            <NavItem eventKey="experiment">
              <NavIcon>
                <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
              </NavIcon>
              <NavText>
                Images
              </NavText>
            </NavItem>
            <NavItem eventKey="experiment">
              <NavIcon>
                <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
              </NavIcon>
              <NavText>
                Plots
              </NavText>
            </NavItem>
            <NavItem eventKey="experiment">
              <NavIcon>
                <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
              </NavIcon>
              <NavText>
                Results
              </NavText>
            </NavItem>
            <NavItem eventKey="experiment">
              <NavIcon>
                <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
              </NavIcon>
              <NavText>
                Logs
              </NavText>
            </NavItem>
          </NavItem>
          <NavItem eventKey="overview">
            <NavIcon>
              <i className="fa fa-fw fa-home" style={{ fontSize: '1.75em' }} />
            </NavIcon>
            <NavText>
              Overview
            </NavText>
          </NavItem>
        </SideNav.Nav>
      </SideNav>
    )
  }
}

export default Sidebar;
