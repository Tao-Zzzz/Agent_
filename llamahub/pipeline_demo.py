def main() -> None:
    try:
        from llama_index.core import Document, Settings, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.llms import MockLLM
        from llama_index.core.node_parser import SentenceSplitter
    except ImportError as exc:
        print("Missing dependency:", exc)
        print()
        print("Run this command in your activated virtual environment:")
        print("    pip install llama-index-core")
        return
    print("=== 1. Create source documents ===")
    documents = [
        Document(
            text=(
                "Alfred is planning a dinner party at Wayne Manor. "
                "Bruce prefers light meals with grilled fish and salad. "
                "Diana prefers Mediterranean food and avoids very sweet desserts."
            ),
            metadata={"source": "guest_preferences"},
        ),
        Document(
            text=(
                "The successful menu from the last party included tomato soup, "
                "roasted vegetables, grilled salmon, and lemon tea. "
                "Guests disliked heavy cream sauces."
            ),
            metadata={"source": "past_menu"},
        ),
    ]
    print(f"Loaded documents: {len(documents)}")
    for index, document in enumerate(documents, start=1):
        print(f"Document {index} source:", document.metadata["source"])
        print(document.text)
        print()
    print("=== 2. Build ingestion pipeline ===")
    embed_model = MockEmbedding(embed_dim=32)
    llm = MockLLM(max_tokens=128)
    Settings.embed_model = embed_model
    Settings.llm = llm
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(chunk_size=40, chunk_overlap=5),
            embed_model,
        ]
    )
    print("Pipeline transformations:")
    print("- SentenceSplitter: split long documents into smaller nodes")
    print("- MockEmbedding: turn each node into a vector")
    print()
    print("=== 3. Run pipeline: Document -> Node + Embedding ===")
    nodes = pipeline.run(documents=documents)
    print(f"Created nodes: {len(nodes)}")
    for index, node in enumerate(nodes, start=1):
        print(f"Node {index}:")
        print(node.get_content())
        print("Metadata:", node.metadata)
        print()
    print("=== 4. Create VectorStoreIndex from nodes ===")
    index = VectorStoreIndex(nodes, embed_model=embed_model)
    print("Index is ready.")
    print()
    question = "What should Alfred prepare for the dinner party?"
    print("=== 5. Retrieve relevant nodes ===")
    print("Question:", question)
    retriever = index.as_retriever(similarity_top_k=2)
    retrieved_nodes = retriever.retrieve(question)
    for rank, node_with_score in enumerate(retrieved_nodes, start=1):
        print(f"Retrieved node {rank}: score={node_with_score.score}")
        print(node_with_score.node.get_content())
        print()
    print("=== 6. QueryEngine answer ===")
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=2,
        response_mode="compact",
    )
    response = query_engine.query(question)
    print(response)
    print()
    print("=== 7. Plain-English summary ===")
    print("Documents were split into nodes.")
    print("Nodes were embedded as vectors.")
    print("The retriever found relevant nodes.")
    print("QueryEngine used those nodes as context to produce an answer.")


if __name__ == "__main__":
    main()
