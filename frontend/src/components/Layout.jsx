import React from 'react';
import './Layout.css';

const Layout = ({ children }) => {
  return (
    <div className="layout">
      <nav className="sidebar">
        <div className="logo">
          <h1>VecBrain</h1>
        </div>
        <ul className="nav-links">
          <li>
            <a href="#chat" className="active">
              <i className="fas fa-comments"></i>
              Chat
            </a>
          </li>
          <li>
            <a href="#documents">
              <i className="fas fa-file-alt"></i>
              Documents
            </a>
          </li>
          <li>
            <a href="#prompts">
              <i className="fas fa-magic"></i>
              Prompts
            </a>
          </li>
          <li>
            <a href="#agent">
              <i className="fas fa-robot"></i>
              Agent
            </a>
          </li>
        </ul>
      </nav>
      <main className="main-content">
        {children}
      </main>
    </div>
  );
};

export default Layout; 