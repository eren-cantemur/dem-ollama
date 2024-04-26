import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import "@tensorflow/tfjs-node";
import { TensorFlowEmbeddings } from "@langchain/community/embeddings/tensorflow";
import { RetrievalQAChain } from "langchain/chains";
import { Ollama } from "@langchain/community/llms/ollama";
import * as readline from "node:readline";

const ollama = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3",
});

const loader = new CheerioWebBaseLoader(
  "https://en.wikipedia.org/wiki/Mehmed_II"
);
const data = await loader.load();

// Split the text into 500 character chunks. And overlap each chunk by 20 characters
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 20,
});

const splitDocs = await textSplitter.splitDocuments(data);

// Then use the TensorFlow Embedding to store these chunks in the datastore
const vectorStore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  new TensorFlowEmbeddings()
);

const retriever = vectorStore.asRetriever();
const chain = RetrievalQAChain.fromLLM(ollama, retriever);
const inp = "";

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion() {
  rl.question(`> `, (input) => {
    if (input === "bye") {
      rl.close();
    } else {
      chain
        .call({ query: input })
        .then((result) => {
          console.log(result.text);
        })
        .catch((error) => {
          console.error("Error:", error);
        })
        .finally(() => askQuestion());
    }
  });
}

askQuestion(); // Initial call to start the loop
