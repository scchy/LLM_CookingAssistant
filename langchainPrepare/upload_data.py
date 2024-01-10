# python3
# Create Date: 2024-01-10
# Author: Scc_hy
# Func: 数据上传openxlab
# ==============================================================================


from openxlab.dataset import upload_file

upload_file(
    dataset_repo='Scchy/LLM-Data', 
    source_path='cookingBook.json', 
    # target_path='Scchy/LLM-Data'
)