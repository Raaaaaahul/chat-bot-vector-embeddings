import axios from "axios";
import * as cheerio from "cheerio";
import dotenv from "dotenv";
import { ChromaClient } from "chromadb";
import { CohereClientV2 } from "cohere-ai";
import { URL } from "url";
import OpenAI from "openai";

dotenv.config(); // Load environment variables from a .env file

// ADD the webpage url and the subdomain
const BASE_URL = ""; // Base URL to start crawling from
const BASE_DOMAIN = ""; // Base domain to restrict crawling
const openai = new OpenAI(); // Initialize OpenAI client

const chromaClient = new ChromaClient({
  path: "http://localhost:8000", // Path to ChromaDB server
});

const cohere = new CohereClientV2({
  token: process.env.COHERE_API_KEY, // API key for Cohere
});

const EMBEDDING_COLLECTION = `WEB_SCRAPED_DATA_COLLECTION`; // Collection name in ChromaDB

// Class to represent a node in the crawl tree
class TreeNode {
  constructor(url) {
    this.url = url; // URL of the webpage
    this.children = new Set(); // Set of child nodes (links)
  }
}

// Function to generate vector embeddings for a given text
async function generateVectorEmbeddings(text) {
  const response = await cohere.embed({
    model: "embed-english-v3.0",
    texts: [text],
    embeddingTypes: ["float"],
    inputType: "search_query",
  });

  return response.embeddings[0]; // Return the first embedding
}

// Function to insert data into the ChromaDB
async function insertIntoDb({ embedding, url, body = "", head }) {
  const collection = await chromaClient.getOrCreateCollection({
    name: EMBEDDING_COLLECTION,
  });

  await collection.add({
    ids: [url], // ID of the document
    embeddings: [embedding], // Embedding vector
    metadatas: [{ url, body, head }], // Metadata including URL, body, and head
  });
}

// Function to chunk text into smaller pieces
function chunkText(text, chunkSize) {
  if (!text || chunkSize <= 0) return [];
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks; // Return array of text chunks
}

// Function to scrape a webpage for content and links
async function scrapeWebpage(url) {
  const { data } = await axios.get(url); // Fetch the webpage
  const $ = cheerio.load(data);

  const pageHead = $("head").html(); // Extract head content
  const pageBody = $("body").html(); // Extract body content
  const internalLinks = new Set(); // Set to store internal links

  // Find all anchor tags and extract valid internal links
  $("a").each((_, el) => {
    let link = $(el).attr("href");
    if (!link || link === "/" || link.startsWith("#")) return;

    try {
      const resolvedUrl = new URL(link, url).href;
      const parsedUrl = new URL(resolvedUrl);
      if (
        parsedUrl.hostname === BASE_DOMAIN &&
        resolvedUrl.startsWith(BASE_URL)
      ) {
        internalLinks.add(resolvedUrl);
      }
    } catch (error) {
      console.warn(`Skipping invalid URL: ${link}`);
    }
  });

  return {
    head: pageHead,
    body: pageBody,
    internalLinks: Array.from(internalLinks), // Return as array
  };
}

// Recursive function to crawl and ingest webpages
async function crawl(node, visited = new Set()) {
  if (visited.has(node.url)) return; // Avoid revisiting URLs
  visited.add(node.url);

  console.log(`🍴 Ingesting ${node.url}`);
  const { head, body, internalLinks } = await scrapeWebpage(node.url);

  const headEmbedding = await generateVectorEmbeddings(head);
  await insertIntoDb({ embedding: headEmbedding, url: node.url });

  const bodyChunks = chunkText(body, 1000); // Chunk body into smaller parts
  for (const chunk of bodyChunks) {
    const bodyEmbedding = await generateVectorEmbeddings(chunk);
    await insertIntoDb({
      embedding: bodyEmbedding,
      url: node.url,
      head,
      body: chunk,
    });
  }

  // Recursively crawl internal links
  for (const link of internalLinks) {
    if (!visited.has(link)) {
      const childNode = new TreeNode(link);
      node.children.add(childNode);
      await crawl(childNode, visited);
    }
  }
}

// Function to chat using the embedded data
async function chat(question = "") {
  const questionEmbedding = await generateVectorEmbeddings(question);

  if (!questionEmbedding) {
    console.error("❌ Error: Failed to generate embeddings for the question.");
    return;
  }

  const collection = await chromaClient.getOrCreateCollection({
    name: EMBEDDING_COLLECTION,
  });

  try {
    const collectionResult = await collection.query({
      nResults: 1,
      queryEmbeddings: questionEmbedding,
    });

    if (
      !collectionResult.metadatas ||
      collectionResult.metadatas.length === 0
    ) {
      console.log("🔍 No relevant data found in the database.");
      return;
    }

    const body = collectionResult.metadatas[0]
      .map((e) => e.body)
      .filter((e) => e && e.trim() !== "");

    const url = collectionResult.metadatas[0]
      .map((e) => e.url)
      .filter((e) => e && e.trim() !== "");

    // Create a chat prompt based on retrieved context
    const chatPrompt = `
      You are an AI support assistant specializing in providing information to users based on the given webpage context.
      Answer the user's question based on the retrieved content.
      
      Query: ${question}
      URL: ${url.join(", ")}
      Retrieved Context: ${body.join(", ")}
    `;

    const response = await cohere.chat({
      model: "command-r-plus",
      messages: [{ role: "user", content: chatPrompt }],
    });

    console.log(`🤖: ${response.message.content[0].text}`);
  } catch (error) {
    console.error("❌ Error querying ChromaDB:", error.message);
  }
}

const rootNode = new TreeNode(BASE_URL); // Create root node for crawling
crawl(rootNode); // Start crawling from the root node
chat("question"); // Add your question here to chat
