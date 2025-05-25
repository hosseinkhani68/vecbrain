import React, { useState, useEffect } from 'react';
import './Prompts.css';

const Prompts = () => {
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState(null);
  const [inputData, setInputData] = useState({});
  const [response, setResponse] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    loadTemplates();
  }, []);

  const loadTemplates = async () => {
    try {
      const response = await fetch('/prompt/templates');
      if (!response.ok) throw new Error('Failed to load templates');
      const data = await response.json();
      setTemplates(data);
    } catch (error) {
      console.error('Error loading templates:', error);
    }
  };

  const handleTemplateSelect = async (templateName) => {
    try {
      const response = await fetch(`/prompt/templates/${templateName}`);
      if (!response.ok) throw new Error('Failed to load template info');
      const data = await response.json();
      setSelectedTemplate(data);
      setInputData({});
    } catch (error) {
      console.error('Error loading template info:', error);
    }
  };

  const handleInputChange = (key, value) => {
    setInputData(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleGenerate = async () => {
    if (!selectedTemplate) return;

    setIsLoading(true);
    try {
      const response = await fetch('/prompt/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          template_name: selectedTemplate.name,
          input_data: inputData
        })
      });

      if (!response.ok) throw new Error('Generation failed');
      const data = await response.json();
      setResponse(data);
    } catch (error) {
      console.error('Error generating response:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="prompts-container">
      <div className="prompts-header">
        <h2>Prompt Templates</h2>
      </div>

      <div className="prompts-content">
        <div className="templates-list">
          <h3>Available Templates</h3>
          <ul>
            {templates.map(template => (
              <li
                key={template}
                className={selectedTemplate?.name === template ? 'active' : ''}
                onClick={() => handleTemplateSelect(template)}
              >
                {template}
              </li>
            ))}
          </ul>
        </div>

        <div className="template-editor">
          {selectedTemplate ? (
            <>
              <h3>{selectedTemplate.name}</h3>
              <p className="template-description">{selectedTemplate.description}</p>
              
              <div className="input-fields">
                {selectedTemplate.input_variables.map(variable => (
                  <div key={variable} className="input-field">
                    <label>{variable}</label>
                    <input
                      type="text"
                      value={inputData[variable] || ''}
                      onChange={(e) => handleInputChange(variable, e.target.value)}
                      placeholder={`Enter ${variable}`}
                    />
                  </div>
                ))}
              </div>

              <button
                className="generate-button"
                onClick={handleGenerate}
                disabled={isLoading}
              >
                {isLoading ? 'Generating...' : 'Generate Response'}
              </button>

              {response && (
                <div className="response-section">
                  <h4>Generated Response</h4>
                  <div className="response-content">
                    {response.response}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="no-template-selected">
              Select a template to get started
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Prompts; 