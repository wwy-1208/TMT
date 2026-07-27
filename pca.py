import torch

from sklearn.decomposition import PCA

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

model = GPT2Model.from_pretrained(
    "D:/TMT/gpt2_tokenizer",  # 简化相对路径
    local_files_only=True  # 强制本地加载，避免访问Hub
)

wte = model.wte.state_dict()['weight'].cpu().numpy()

pca = PCA(n_components=700)

wte_pca = pca.fit_transform(wte.T)

torch.save(wte_pca, "wte_pca_500.pt")