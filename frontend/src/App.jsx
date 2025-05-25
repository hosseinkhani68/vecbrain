import React, { useState, useEffect, useRef } from 'react';
import './styles/App.css';
import Layout from './components/Layout';
import Documents from './components/Documents';
import Prompts from './components/Prompts';
import Agent from './components/Agent';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [contextId, setContextId] = useState(null);
  const messagesEndRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [input, setInput] = useState('');

  // Load chat history when component mounts
  useEffect(() => {
    const loadChatHistory = async () => {
      try {
        const response = await fetch('/chat-history');
        if (!response.ok) {
          throw new Error('Failed to load chat history');
        }
        const history = await response.json();
        
        // Sort messages by timestamp
        const sortedMessages = history.sort((a, b) => 
          new Date(a.timestamp) - new Date(b.timestamp)
        );
        
        setMessages(sortedMessages);
        
        // Set context ID from the most recent message if available
        if (sortedMessages.length > 0) {
          setContextId(sortedMessages[sortedMessages.length - 1].id);
        }
      } catch (error) {
        console.error('Error loading chat history:', error);
      }
    };

    loadChatHistory();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: input,
      role: 'user',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsStreaming(true);

    try {
      const response = await fetch('/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: input,
          context_id: contextId
        })
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = {
        id: (Date.now() + 1).toString(),
        text: '',
        role: 'assistant',
        timestamp: new Date().toISOString()
      };

      setMessages(prev => [...prev, assistantMessage]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            if (data.done) {
              setIsStreaming(false);
              break;
            }
            assistantMessage.text += data.chunk;
            setMessages(prev => [...prev]);
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setIsStreaming(false);
    }
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'chat':
        return (
          <div className="chat-container">
            <div className="chat-header">
              <h1>VecBrain Chat</h1>
              <p>AI-powered conversation with context awareness</p>
            </div>
            
            <div className="messages-container">
              {messages.map((message) => (
                <div key={message.id} className={`message ${message.role}`}>
                  <div className="message-content">
                    {message.text}
                    {message.role === 'assistant' && isStreaming && message.id === messages[messages.length - 1].id && (
                      <span className="cursor">▋</span>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>

            <form onSubmit={handleSubmit} className="input-form">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Type your message..."
                disabled={isStreaming}
              />
              <button type="submit" disabled={isStreaming}>
                {isStreaming ? 'Sending...' : 'Send'}
              </button>
            </form>
          </div>
        );
      case 'documents':
        return <Documents />;
      case 'prompts':
        return <Prompts />;
      case 'agent':
        return <Agent />;
      default:
        return null;
    }
  };

  return (
    <Layout>
      {renderContent()}
    </Layout>
  );
}

export default App; 