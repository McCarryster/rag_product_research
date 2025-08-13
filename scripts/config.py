# paths
vectorstore_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/vector_store'
data_dir = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/data/pdfs'
model_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/model/mistral-7b-instruct-v0.1.Q4_K_M.gguf'

# vector_store params
chunk_size = 1000
chunk_overlap = 100
top_k = 15

# generation params
max_tokens=512,
n_gpu_layers=-1
n_ctx=6000
temperature=0.7,
top_p=0.9,
device='cuda'