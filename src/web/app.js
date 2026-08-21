const messages = document.getElementById("messages");
const question = document.getElementById("question");
const form = document.getElementById("chat-form");
const send = document.getElementById("send");
const mode = document.getElementById("mode");

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

function addResults(results) {
  const message = node("article", "message");
  message.append(node("div", "label", "Scout retrieval"));
  if (!results.length) {
    message.append(node("div", "empty", "No indexed passage matched that question."));
  } else {
    const list = node("div", "results");
    for (const result of results) {
      const card = node("article", "result");
      const head = node("div", "result-head");
      head.append(node("div", "result-title", result.title || result.relative_path || "Untitled document"));
      head.append(node("div", "path", result.collection || ""));
      card.append(head, node("div", "path", result.relative_path || ""), node("div", "excerpt", result.text || ""));
      list.append(card);
    }
    message.append(list);
  }
  messages.append(message);
  message.scrollIntoView({ block:"end", behavior:"smooth" });
}

function addFailure(text) {
  const message = node("article", "message");
  message.append(node("div", "label", "Scout retrieval"), node("div", "failure", text));
  messages.append(message);
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
}

form.addEventListener("submit", async event => {
  event.preventDefault();
  const query = question.value.trim();
  if (!query) return;
  addUser(query);
  question.value = "";
  send.disabled = true;
  send.textContent = "Searching";
  try {
    const response = await fetch("/query", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({ query, mode:mode.value, top_k:5, rerank:false }),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "Search failed");
    addResults(body);
  } catch (error) {
    addFailure(error.message || "Search failed.");
  } finally {
    send.disabled = false;
    send.textContent = "Search";
    question.focus();
  }
});

question.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

setup();