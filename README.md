# VecBrain

A powerful API for semantic search and question answering using OpenAI and Qdrant.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/hosseinkhani68/vecbrain.git
cd vecbrain
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following variables:
```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
```

4. Run the application:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Features

- Intelligent document processing with agent-based chunking
- Advanced search with context awareness
- AI-powered question answering
- Chat with memory and context
- Text simplification
- Diagram generation

## API Documentation

Once the server is running, you can access:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## Security

- API keys and sensitive credentials are stored in environment variables
- Never commit the `.env` file to version control
- The `.env` file is automatically ignored by git

## Project Structure

```
.
├── app/
│   ├── main.py           # FastAPI application
│   ├── config.py         # Configuration settings
│   ├── services/
│   │   ├── openai.py     # OpenAI service
│   │   └── qdrant.py     # Qdrant service
│   └── models/
│       └── schemas.py    # Pydantic models
├── requirements.txt
├── Procfile             # Deployment configuration
└── .env
```

## Deployment

The application includes a `Procfile` for deployment on platforms like Railway. The Procfile specifies how to run the application:

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Make sure to set the following environment variables in your deployment platform:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY` 