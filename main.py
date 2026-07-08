import os
from modules import (
    build_config, fetch_image_cand, fetch_image_seq, preprocess_image, augment_image,
    scrape_3D
)

print('###########################################################################')
print('###########################################################################')

############### config
cfg = build_config()
print('------------Configuration DONE!!------------')


############### scrape crossing images from mapillary (ONLY ONE-TIME TASK)
df_image = fetch_image_cand(cfg)
df_image_seq = fetch_image_seq(cfg, download=True)

print('------------Scraping Images DONE!!------------')


############### preprocess images (ONLY ONE-TIME TASK)
# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.5, text_input=['traffic sign', 'traffic light'])
# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.1, text_input=["X-shaped white traffic sign with black text", "Two white rectangular boards crossed in an X-shape", "X-shaped railroad crossing sign on a metal pole", "White wooden or metal planks forming a cross with 'RAILROAD CROSSING' text", "X-shaped sign with small red reflectors on the edges"]) # not bad, but not good enough
# preprocess_image(cfg, model_name='yoloe-26x-seg', confidence_threshold=0.1, visual_input='crossbuck_4.jpg')

# preprocess_image(cfg, model_name='IDEA-Research/grounding-dino-base', confidence_threshold=0.3, text_input=["traffic sign", "traffic light"]) # grounding dino preprocesses text inputs as '. '.join(list_of_labels), not treating each label as an individual token.
# preprocess_image(cfg, model_name='IDEA-Research/grounding-dino-base', confidence_threshold=0.3, text_input=["gate arm", "barrier arm", "lifted gate arm", "lifted barrier arm"])

# preprocess_image(cfg, model_name='facebook/sam-vit-huge', confidence_threshold=0.3, visual_input='crossbuck_4.jpg')

# preprocess_image(cfg, model_name='facebook/sam3', confidence_threshold=0.5, visual_input='crossbuck_4.jpg')

# print('------------Preprocessing Images DONE!!------------')

# ############### image augmentation (ONLY ONE-TIME TASK)
# augment_image(cfg)

# print('------------Augmenting Images DONE!!------------')


# ############### scrape 3D reconstruction from mapillary (ONLY ONE-TIME TASK)
# df_3D = scrape_3D(cfg)


############### 

print('###########################################################################')
print('###########################################################################')