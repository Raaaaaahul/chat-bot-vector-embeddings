# Web Scraper with ChromaDB and Cohere Integration

This project is a web scraper that collects data from a specified website, generates embeddings using Cohere's API, and stores them in a local ChromaDB database. It can then answer questions based on the scraped data.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **Docker**: You need to have Docker installed on your machine to run ChromaDB. You can download Docker from [here](https://www.docker.com/products/docker-desktop).

- **API Keys**: You need an API key for Cohere or OpenAI to generate embeddings. If you're using Cohere (like me, because it's a more budget-friendly option), you'll need to sign up on their platform and get an API key. For OpenAI, you'll need an API key as well.

## Getting Started

Follow these steps to set up and run the project:

### 1. Set Up ChromaDB

First, pull the ChromaDB Docker image and start it using Docker Compose.

```bash
# Pull the ChromaDB Docker image
docker pull chromadb_image

# Start ChromaDB using Docker Compose
docker-compose up
```


# .env file
COHERE_API_KEY=your_cohere_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional if you're using OpenAI

