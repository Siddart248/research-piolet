# Knowledge Base Directory

Place local documents (e.g., `.txt`, `.md`, or `.pdf`) here for indexing and retrieval by ResearchPilot.

The retrieval pipeline in `app/retrieval.py` will read files from this directory, split them into chunks, compute embeddings via `app/embeddings.py`, and perform vector similarity search.
