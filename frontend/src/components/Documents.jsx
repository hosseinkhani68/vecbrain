import React, { useState } from 'react';
import './Documents.css';

const Documents = () => {
  const [documents, setDocuments] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/documents/process', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error('Upload failed');

      const result = await response.json();
      setDocuments(prev => [...prev, result]);
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch('/documents/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery }),
      });

      if (!response.ok) throw new Error('Search failed');

      const results = await response.json();
      // Handle search results
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  return (
    <div className="documents-container">
      <div className="documents-header">
        <h2>Document Management</h2>
        <div className="documents-actions">
          <div className="upload-section">
            <label className="upload-button">
              <input
                type="file"
                onChange={handleFileUpload}
                disabled={isUploading}
                accept=".txt,.pdf,.doc,.docx"
              />
              {isUploading ? 'Uploading...' : 'Upload Document'}
            </label>
          </div>
          <div className="search-section">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search documents..."
            />
            <button onClick={handleSearch}>Search</button>
          </div>
        </div>
      </div>

      <div className="documents-list">
        {documents.map((doc) => (
          <div key={doc.doc_id} className="document-card">
            <h3>{doc.metadata?.title || 'Untitled Document'}</h3>
            <p>Chunks: {doc.chunks}</p>
            <p>Source: {doc.source}</p>
            <div className="document-actions">
              <button onClick={() => handleViewChunks(doc.doc_id)}>View Chunks</button>
              <button onClick={() => handleDeleteDocument(doc.doc_id)}>Delete</button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Documents; 