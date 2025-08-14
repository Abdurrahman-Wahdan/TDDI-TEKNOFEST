# Copyright 2025 kermits
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from embedding_system import create_embeddings_from_csv
from vector_store import create_sss_collection, store_sss_embeddings

# Create embeddings
embeddings_data = create_embeddings_from_csv("faq_data.csv")

# Store in vector database
create_sss_collection(vector_size=1024)
store_sss_embeddings(embeddings_data)