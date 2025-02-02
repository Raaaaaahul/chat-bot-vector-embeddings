import axios from "axios";
import * as cheerio from "cheerio";
import dotenv from "dotenv";
import { ChromaClient } from "chromadb";
import { CohereClientV2 } from "cohere-ai";
import { URL } from "url";
import OpenAI from "openai";

dotenv.config();

const BASE_URL = "https://piyushgarg.dev";
const BASE_DOMAIN = "piyushgarg.dev";
const openai = new OpenAI();

const chromaClient = new ChromaClient({
  path: "http://localhost:8000",
});

const cohere = new CohereClientV2({
  token: process.env.COHERE_API_KEY,
});

const EMBEDDING_COLLECTION = `WEB_SCRAPED_DATA_COLLECTION`;

class TreeNode {
  constructor(url) {
    this.url = url;
    this.children = new Set();
  }
}

async function generateVectorEmbeddings(text) {
  const response = await cohere.embed({
    model: "embed-english-v3.0",
    texts: [text],
    embeddingTypes: ["float"],
    inputType: "search_query",
  });

  return response.embeddings[0];
}

async function insertIntoDb({ embedding, url, body = "", head }) {
  const collection = await chromaClient.getOrCreateCollection({
    name: EMBEDDING_COLLECTION,
  });

  await collection.add({
    ids: [url],
    embeddings: [embedding],
    metadatas: [{ url, body, head }],
  });
}

function chunkText(text, chunkSize) {
  if (!text || chunkSize <= 0) return [];
  const words = text.split(/\s+/);
  const chunks = [];
  for (let i = 0; i < words.length; i += chunkSize) {
    chunks.push(words.slice(i, i + chunkSize).join(" "));
  }
  return chunks;
}

async function scrapeWebpage(url) {
  const { data } = await axios.get(url);
  const $ = cheerio.load(data);

  const pageHead = $("head").html();
  const pageBody = $("body").html();
  const internalLinks = new Set();

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
    internalLinks: Array.from(internalLinks),
  };
}

async function crawl(node, visited = new Set()) {
  if (visited.has(node.url)) return;
  visited.add(node.url);

  console.log(`🍴 Ingesting ${node.url}`);
  const { head, body, internalLinks } = await scrapeWebpage(node.url);

  const headEmbedding = await generateVectorEmbeddings(head);
  await insertIntoDb({ embedding: headEmbedding, url: node.url });

  const bodyChunks = chunkText(body, 1000);
  for (const chunk of bodyChunks) {
    const bodyEmbedding = await generateVectorEmbeddings(chunk);
    await insertIntoDb({
      embedding: bodyEmbedding,
      url: node.url,
      head,
      body: chunk,
    });
  }

  for (const link of internalLinks) {
    if (!visited.has(link)) {
      const childNode = new TreeNode(link);
      node.children.add(childNode);
      await crawl(childNode, visited);
    }
  }
}

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

const rootNode = new TreeNode(BASE_URL);
crawl(rootNode);
chat("what is cohort about?");
