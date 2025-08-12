from embedding_system import create_embeddings_from_csv
from vector_store import create_sss_collection, store_sss_embeddings

# Create embeddings
embeddings_data = create_embeddings_from_csv("faq_data.csv")

# Store in vector database
create_sss_collection(vector_size=1024)
store_sss_embeddings(embeddings_data)