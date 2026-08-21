const messages = document.getElementById("messages");
const question = document.getElementById("question");
const form = document.getElementById("chat-form");
const send = document.getElementById("send");
const clear = document.getElementById("clear");
const mode = document.getElementById("mode");
const searchCollection = document.getElementById("search-collection");
const uploadForm = document.getElementById("upload-form");
const files = document.getElementById("files");
const collection = document.getElementById("collection");
const upload = document.getElementById("upload");
const uploadStatus = document.getElementById("upload-status");

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function addUser(text) {
  const message = node("article", "message user");
  message.append(node("div", "label", "You"), node("div", "bubble", text));
  messages.append(message);
}

function addAnswer(body, seconds) {
  const message = node("article", "message");
  message.append(node("div", "label", "Scout answer"));
  if (!body.answer) {
    message.append(node("div", "empty", "No indexed passage matched that question."));
  } else {
    const bubble = node("div", "bubble markdown");
    bubble.innerHTML = marked.parse(body.answer);
    message.append(bubble);
    if (typeof seconds === "number") {
      message.append(node("div", "timing", `Answered in ${seconds.toFixed(1)} s`));
    }
    const sources = (body.sources || []).filter(source => source);
    if (sources.length) {
      const details = node("details", "sources");
      details.append(node("summary", "sources-summary", `${sources.length} source${sources.length === 1 ? "" : "s"}`));
      const list = node("div", "results");
      sources.forEach((source, index) => {
        const card = node("article", "result");
        const head = node("div", "result-head");
        head.append(node("div", "result-title", `[${index + 1}] ${source.title || source.relative_path || "Untitled document"}`));
        head.append(node("div", "path", source.collection || ""));
        card.append(head, node("div", "path", source.relative_path || ""), node("div", "excerpt", source.text || ""));
        list.append(card);
      });
      details.append(list);
      message.append(details);
    }
  }
  messages.append(message);
  message.scrollIntoView({ block:"end", behavior:"smooth" });
}

function addFailure(text) {
  const message = node("article", "message");
  message.append(node("div", "label", "Scout answer"), node("div", "failure", text));
  messages.append(message);
}

function setUploadStatus(text, failed = false) {
  uploadStatus.textContent = text;
  uploadStatus.classList.toggle("failure", failed);
}

async function loadSearchCollections() {
  const response = await fetch("/collections");
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || "Could not load collections");
  searchCollection.replaceChildren(new Option("All collections", ""));
  for (const entry of body.collections) {
    if (typeof entry.name === "string" && entry.name) {
      searchCollection.append(new Option(entry.name, entry.name));
    }
  }
}

async function setup() {
  try {
    const health = await fetch("/health").then(response => response.json());
    document.getElementById("dot").classList.toggle("ready", health.embedding_model_loaded);
    document.getElementById("status").textContent = health.embedding_model_loaded ? "Model ready" : "Model unavailable";
  } catch (error) {
    document.getElementById("status").textContent = "Server unavailable";
    addFailure("Could not reach the local Scout service.");
  }
  try {
    await loadSearchCollections();
  } catch (error) {
    console.error(error);
  }
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const query = question.value.trim();
  if (!query) return;
  addUser(query);
  question.value = "";
  send.disabled = true;
  send.textContent = "Asking";
  const started = performance.now();
  try {
    const response = await fetch("/ask", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        query,
        collections:searchCollection.value ? [searchCollection.value] : [],
        mode:mode.value,
        top_k:5,
        rerank:null,
      }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Ask failed");
    addAnswer(body, (performance.now() - started) / 1000);
  } catch (error) {
    addFailure(error.message || "Ask failed.");
  } finally {
    send.disabled = false;
    send.textContent = "Ask";
    question.focus();
  }
});

clear.addEventListener("click", () => {
  messages.replaceChildren();
  question.focus();
});

question.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

uploadForm.addEventListener("submit", async event => {
  event.preventDefault();
  const selectedFiles = [...files.files];
  const collectionName = collection.value.trim();
  if (!selectedFiles.length || !collectionName) return;
  const payload = new FormData();
  for (const file of selectedFiles) payload.append("files", file);
  payload.append("collection", collectionName);
  upload.disabled = true;
  upload.textContent = "Ingesting";
  setUploadStatus(`Ingesting ${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"}.`);
  try {
    const response = await fetch("/ingest", { method:"POST", body:payload });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Ingestion failed");
    files.value = "";
    const chunks = body.results.reduce((total, result) => total + (result.chunks || 0), 0);
    setUploadStatus(`Ingested ${body.files} file${body.files === 1 ? "" : "s"} into ${collectionName} (${chunks} chunks).`);
  } catch (error) {
    setUploadStatus(error.message || "Ingestion failed.", true);
  } finally {
    upload.disabled = false;
    upload.textContent = "Ingest files";
  }
});

setup();