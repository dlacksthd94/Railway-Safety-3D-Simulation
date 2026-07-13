import os
from modules import (
    build_config, fetch_image_cand, fetch_image_seq, fetch_SfM, examine_SfM,
    preprocess_image, augment_image,
)

print('###########################################################################')
print('###########################################################################')

############### config
cfg = build_config()
print('------------Configuration DONE!!------------')


# ############### fetch crossing images from mapillary (ONLY ONE-TIME TASK)
# df_image = fetch_image_cand(cfg)
# df_image_seq = fetch_image_seq(cfg, download=True)

# print('------------Fetching Images DONE!!------------')


############### fetch 3D reconstruction from mapillary (ONLY ONE-TIME TASK)
# fetch_SfM(cfg)
examine_SfM(cfg)
print('------------Fetching SfM DONE!!------------')


############### 

print('###########################################################################')
print('###########################################################################')