import {combineReducers} from 'redux'
import DummyReducer from './reducer-dummy'

const allReducers=combineReducers({
    dummy: DummyReducer,
})
export default allReducers