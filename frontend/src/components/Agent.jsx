import React, { useState } from 'react';
import './Agent.css';

const Agent = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    try {
      const response = await fetch('/agent/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });

      if (!response.ok) throw new Error('Query failed');
      const data = await response.json();

      const newInteraction = {
        id: Date.now(),
        query,
        response: data.response,
        tools: data.tools_used,
        timestamp: data.timestamp
      };

      setHistory(prev => [newInteraction, ...prev]);
      setResponse(data);
      setQuery('');
    } catch (error) {
      console.error('Error querying agent:', error);
      setResponse({
        response: 'Sorry, there was an error processing your query.',
        error: error.message
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="agent-container">
      <div className="agent-header">
        <h2>AI Agent</h2>
        <p>Ask the agent to perform complex tasks using various tools</p>
      </div>

      <div className="agent-content">
        <div className="query-section">
          <form onSubmit={handleSubmit}>
            <div className="input-group">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask the agent to do something..."
                disabled={isLoading}
              />
              <button type="submit" disabled={isLoading}>
                {isLoading ? 'Processing...' : 'Send'}
              </button>
            </div>
          </form>
        </div>

        <div className="history-section">
          {history.map(interaction => (
            <div key={interaction.id} className="interaction-card">
              <div className="interaction-header">
                <span className="timestamp">
                  {new Date(interaction.timestamp).toLocaleString()}
                </span>
                <div className="tools-used">
                  {interaction.tools.map(tool => (
                    <span key={tool} className="tool-badge">
                      {tool}
                    </span>
                  ))}
                </div>
              </div>
              
              <div className="query">
                <strong>Query:</strong> {interaction.query}
              </div>
              
              <div className="response">
                <strong>Response:</strong>
                <div className="response-content">
                  {interaction.response}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Agent; 