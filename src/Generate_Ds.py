# %% [markdown]
# ## Script per la generazione del dataset con features estratte
# 
# 

# %%
from CU_Dataset_Factory import CU_Dataset_Factory
from pathlib import PosixPath

# %%
builder = CU_Dataset_Factory(target_feature='label', batch_size=6, features_enable=['category', 'subcategory', 'type', 'reference', 'languages','G_mean_pr','G_nodes', 'G_num_cliques', 'G_rank', 'label'])

# %%
#train = builder.produce(PosixPath('./dataset/'), True, True)
validation = builder.produce(PosixPath('./dataset/'), True, False)


