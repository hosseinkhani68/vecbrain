import React from 'react';
import '../styles/ChatMessage.css';

function ChatMessage({ message }) {
  const { role, text, timestamp, isError } = message;
  const isUser = role === 'user';
  
  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'} ${isError ? 'error-message' : ''}`}>
      <div className="message-content">
        <div className="message-text">{text}</div>
        <div className="message-time">{formatTime(timestamp)}</div>
      </div>
    </div>
  );
}

export default ChatMessage; 