# %% [markdown]
# ## Script per la generazione del dataset con features estratte
# 
# 

# %%
from CU_Dataset_Factory import CU_Dataset_Factory
from pathlib import PosixPath

# %%
builder = CU_Dataset_Factory(target_feature='label', batch_size=16, features_enable=['G','name', 'category', 'subcategory', 'type', 'reference', 'languages','label'])

# %%
#train = builder.produce(PosixPath('./dataset/'), True, True)
validation = builder.produce(PosixPath('./dataset/'), True, False)


