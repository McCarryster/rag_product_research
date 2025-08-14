# paths
vectorstore_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/vector_store'
data_dir = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/data/pdfs'
model_path = '/home/mccarryster/very_big_work_ubuntu/ML_projects/rag_product_research/model/mistral-7b-instruct-v0.1.Q4_K_M.gguf'

# vector_store params
embedding_model_name="BAAI/bge-m3"
chunk_size = 400 # 300
chunk_overlap = 50
top_k = 5

# generation params
max_tokens=1024
n_gpu_layers=-1
n_ctx=10000
temperature=0.7
device='cuda'
top_p = 0.9
top_k_gen = 50
typical_p = 0.95
repeat_penalty = 1.2

# tg params
telegram_token='bot_token'