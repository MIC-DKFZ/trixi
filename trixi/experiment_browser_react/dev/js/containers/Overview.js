import React, {Component} from 'react';
import {bindActionCreators} from 'redux';
import {connect} from 'react-redux';
import style from '../../scss/style.scss'
import * as utils from "../utils/utils";
var classNames = require('classnames');


class Overview extends Component {

    render() {

        return (<div>Hello World</div>
        );
    }
}

/*
function mapStateToProps(state) {
    return 0
}

export default connect(mapStateToProps)(Page);
*/
export default Overview;