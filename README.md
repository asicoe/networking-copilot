# networking-copilot
The Networking Copilot is an AI agent that helps you build and interrogate a live knowledge graph of the people in your on-line or off-line community

## Acknowledgements
This repo is based on this example repo: https://github.com/surrealdb/langchain-surrealdb/tree/main/examples/graphrag-travel-group-chat from the SurrealDB team which is meant to illustrate the integration between LangChain and SurrealDB to provide GraphRAG functionality to chatbots and AI agents.

Featuring:

- SurrealDBVectorStore: for similarity/relevance search
- SurrealDBGraph: a langchain graph store
- SurrealDBGraphQAChain: a Question/Answering langchain chain class capable of querying the graph using LLM models

## Prerequisites

### [SurrealDB](https://surrealdb.com)

You can run SurrealDB locally or start with a [free SurrealDB cloud account](https://surrealdb.com/docs/cloud/getting-started).

For local, two options:
1. [Install SurrealDB](https://surrealdb.com/docs/surrealdb/installation) and [run SurrealDB](https://surrealdb.com/docs/surrealdb/installation/running). Run in-memory with:

    ```shell
    surreal start -u root -p root
    ```

2. [Run with Docker](https://surrealdb.com/docs/surrealdb/installation/running/docker).

    ```shell
    docker run --rm --pull always --name surrealdb -p 8000:8000 surrealdb/surrealdb:v2.6 start --log trace --user root --pass root memory
    ```

### [Ollama](https://ollama.com/download)

Install Ollama:

```shell
curl -fsSL https://ollama.com/install.sh | sh
```

Open Ollama app and download the following models:
- `nomic-embed-text`
- `llama3.2`

### [LangSmith](https://smith.langchain.com/) observability (Optional)

```shell
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=[YOUR_LANGSMITH_API_KEY]
```

## [Surrealist app](https://surrealdb.com/surrealist) (Optional)

Download and run app

Configure new connection with:
- Remote address: `WS` + `localhost:8000`
- Username: `root`
- Password: `root`

Select connection and set up:
- Namespace: networking-copilot
- Database: profiles

After running the `ingest` command (see below) add query `SELECT *, <->?<->? FROM graph_keyword;` in editor and click the `Run query` button and select `Graph` as the result mode, to visualize the entire graph.

## Run locally

Using [just](https://just.systems/man/en/packages.html) from this directory:

```shell
just ingest whatsapp data/_profiles.txt
just chat
```

OR without just:

```shell
uv run cli ingest whatsapp data/_profiles.txt
uv run cli chat
```
