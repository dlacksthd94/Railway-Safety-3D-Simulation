from dataclasses import dataclass, asdict, astuple, field
from typing import Final
import os
import argparse
import re
import json
from .utils import make_dir, remove_dir, sanitize_model_path

# directories
DN_DATA_ROOT: Final[str] = "/data2/clim090/railway/data"
DN_MAPILLARY: Final[str] = 'mapillary'
DN_IMAGE_SEQ: Final[str] = 'image_seq'
DN_SFM_SEQ: Final[str] = 'SfM_seq'

# files
FN_DICT_API_KEY: Final[str] = 'dict_api_key.json'

FN_DF_CROSSING: Final[str] = '251009 NTAD_Railroad_Grade_Crossings_1739202960140128164.csv'
FN_DF_IMAGE_CAND: Final[str] = 'df_image_cand.csv'
FN_DF_IMAGE_IDS_PER_SEQ: Final[str] = 'df_image_ids_per_seq.csv'
FN_DF_IMAGE_SEQ: Final[str] = 'df_image_seq.csv'

# configurations
TARGET_STATES: Final[tuple[str, ...]] = ('California',)

IMG_SEARCH_FIELDS: Final[tuple[str, ...]] = ("id",)
IMG_DETAIL_FIELDS: Final[tuple[str, ...]] = (
    "id",
    "altitude", "computed_altitude", "geometry", "computed_geometry",  # geographic info
    "captured_at",  # timestamp
    "height", "width", "thumb_original_url",  # image
    "compass_angle", "computed_compass_angle", "computed_rotation", "exif_orientation", # orientation info
    "camera_type", "is_pano", "make", "model",  # camera info
    "sequence",  # sequence info
    "camera_parameters", "atomic_scale", "merge_cc", "mesh", "sfm_cluster", # SfM info
    "creator", "organization",  # uploader info
    "quality_score", # image quality info
    # "detections", # object detection - can be fetched using other API: https://www.mapillary.com/developer/api-documentation#detection
)
BBOX_OFFSET: Final[float] = 0.0001 # 0.00001 ≒ 1.11 meters
DIST_THRES_FILTER_IMG_SEQ: Final[float] = 0.0005 # 0.00001 ≒ 1.11 meters


def parse_args() -> argparse.Namespace:
    """Create an argument parser for building configs from the CLI and parse CLI args."""
    parser = argparse.ArgumentParser(description="Project configuration parser for image scraping, preprocessing, and 3D reconstruction tasks.")

    # g_scrape = parser.add_argument_group("scraping")
    # g_scrape.add_argument(
    #     "--c_api",
    #     type=str,
    #     choices=list(CONVERSION_API_MODEL_CHOICES.keys()),
    #     required=True,
    #     help="API to use for form transcription"
    # )
    
    # g_ret = parser.add_argument_group("retrieval")
    # g_ret.add_argument(
    #     "--r_n_generate",
    #     type=int,
    #     choices=N_GENERATE_RANGE,
    #     required=False,
    #     default=1,
    #     help="Number of QAs before aggregation"
    # )
    
    args = parser.parse_args()

    # sanity check
    # if args.c_model not in CONVERSION_API_MODEL_CHOICES[args.c_api]:
    #     parser.error(f"Model '{args.c_model}' is invalid for API '{args.c_api}'. "
    #                  f"Allowed models: {CONVERSION_API_MODEL_CHOICES[args.c_api]}")
    
    return args


@dataclass(frozen=True)
class BaseConfig:
    api: str
    model: str
    n_generate: int
    
    def to_dict(self):
        return asdict(self)

    def to_tuple(self):
        return astuple(self)


@dataclass(frozen=True)
class ScrapingConfig:
    target_states: tuple[str, ...]

    img_search_fields: tuple[str, ...]
    img_detail_fields: tuple[str, ...]
    bbox_offset: float
    dist_thres_filter_img_seq: float

    def __post_init__(self):
        ### sanity check
        for state in self.target_states:
            assert state in ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'District Of Columbia', 
                                     'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 
                                     'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 
                                     'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 
                                     'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                                     'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']


@dataclass(frozen=True)
class PathConfig:
    # generated directories
    dir_image_seq: str
    dir_SfM_seq: str
    
    # files
    dict_api_key: str

    df_crossing: str

    df_image_cand: str
    df_image_ids_per_seq: str
    df_image_seq: str


@dataclass()
class APIkeyConfig:
    openai: str
    google: str
    textract: str
    mapillary: str


@dataclass(frozen=True)
class Config:
    scrp: ScrapingConfig
    path: PathConfig
    apikey: APIkeyConfig


def _compute_paths() -> PathConfig:
    dp_mapillary = os.path.join(DN_DATA_ROOT, DN_MAPILLARY)

    dp_image_seq = os.path.join(dp_mapillary, DN_IMAGE_SEQ)
    make_dir(dp_image_seq)
    dp_SfM_seq = os.path.join(dp_mapillary, DN_SFM_SEQ)
    make_dir(dp_SfM_seq)
    
    return PathConfig(
        df_crossing=os.path.join(DN_DATA_ROOT, FN_DF_CROSSING),
        
        dir_image_seq=dp_image_seq,
        dir_SfM_seq=dp_SfM_seq,
        
        dict_api_key=os.path.join(DN_DATA_ROOT, FN_DICT_API_KEY),
        
        df_image_cand=os.path.join(dp_mapillary, FN_DF_IMAGE_CAND),
        df_image_ids_per_seq=os.path.join(dp_mapillary, FN_DF_IMAGE_IDS_PER_SEQ),
        df_image_seq=os.path.join(dp_mapillary, FN_DF_IMAGE_SEQ),
    )

def _load_api_key(path_cfg: PathConfig) -> APIkeyConfig:
    with open(path_cfg.dict_api_key, 'r') as f:
        dict_api_key = json.load(f)
    dict_api_key = {api: info['key'] for api, info in dict_api_key.items()}
    
    return APIkeyConfig(**dict_api_key)

def build_config(args_dict=None) -> Config:
    if args_dict is None:
        args = parse_args()
        args_dict = vars(args)
    # conv_args = {k.replace('c_', ''): v for k, v in args_dict.items() if k.startswith('c_')}
    # retr_args = {k.replace('r_', ''): v for k, v in args_dict.items() if k.startswith('r_')}
    # conv_args['model'] = sanitize_model_path(conv_args['model'])
    # retr_args['model'] = sanitize_model_path(retr_args['model'])
    scrp_cfg = ScrapingConfig(
        TARGET_STATES, IMG_SEARCH_FIELDS, IMG_DETAIL_FIELDS,
        BBOX_OFFSET, DIST_THRES_FILTER_IMG_SEQ
    )
    path_cfg = _compute_paths()
    apikey_cfg = _load_api_key(path_cfg)
    return Config(scrp=scrp_cfg, path=path_cfg, apikey=apikey_cfg)

if __name__ == '__main__':
    cfg = build_config()