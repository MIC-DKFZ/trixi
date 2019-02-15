import '@trendmicro/react-sidenav/dist/react-sidenav.css';
import SideNav, {
    Toggle,
    Nav,
    NavItem,
    NavIcon,
    NavText
} from '@trendmicro/react-sidenav';
import styled from 'styled-components';

// Styled SideNav
const StyledSideNav = styled(SideNav)`
`;
StyledSideNav.defaultProps = SideNav.defaultProps;

// Styled Nav
const StyledNav = styled(Nav)`
`;
StyledNav.defaultProps = Nav.defaultProps;

export {
    Toggle,
    StyledNav as Nav, // Styled Nav
    NavItem,
    NavIcon,
    NavText
};
export default StyledSideNav; // Styled SideNav
