import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
    const [scrolled, setScrolled] = useState(false);
    const [menuOpen, setMenuOpen] = useState(false);
    const location = useLocation();

    useEffect(() => {
        const onScroll = () => setScrolled(window.scrollY > 20);
        window.addEventListener('scroll', onScroll);
        return () => window.removeEventListener('scroll', onScroll);
    }, []);

    useEffect(() => {
        setMenuOpen(false);
    }, [location]);

    return (
        <nav className={`navbar ${scrolled ? 'scrolled' : ''}`}>
            <div className="navbar-inner">
                <Link to="/" className="navbar-brand">
                    <img src="/LOGO.png" style={{ height: "30px", width: "auto", marginRight: "2px" }} alt="Safesight Logo" />
                    Safesight
                </Link>

                <button
                    className="hamburger"
                    onClick={() => setMenuOpen(!menuOpen)}
                    aria-label="Toggle menu"
                >
                    <span />
                    <span />
                    <span />
                </button>

                <ul className={`navbar-links ${menuOpen ? 'open' : ''}`}>
                    <li>
                        <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
                            Home
                        </Link>
                    </li>
                    <li>
                        <Link
                            to="/upload"
                            className={location.pathname === '/upload' ? 'active' : ''}
                        >
                            Upload
                        </Link>
                    </li>
                </ul>
            </div>
        </nav>
    );
}
