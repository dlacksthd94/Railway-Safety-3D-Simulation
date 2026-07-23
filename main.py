import os
from modules import (
    build_config, fetch_image_cand, fetch_image_seq, fetch_SfM, examine_SfM,
    preprocess_image, augment_image, visualize_SfM
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


############### fetch SfM from mapillary (ONLY ONE-TIME TASK)
# fetch_SfM(cfg)
# examine_SfM(cfg)
visualize_SfM(cfg, crossing_id='753180E', seq_id='T1DxArocIfbQLG4MHEmSsd', sfm_id='3269502816551910')
print('------------Fetching SfM DONE!!------------')


############### 

print('###########################################################################')
print('###########################################################################')