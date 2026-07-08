import os
import time
from transformers import pipeline
import json
import json5
import ast
import shutil
import pandas as pd
from pprint import pprint


def make_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def remove_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def sanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("/", "--")


def desanitize_model_path(model_path: str) -> str:
    """Replace path separators so model_path names are safe in folder names."""
    return model_path.replace("--", "/")


def generate_openai(client, model_path, content, generation_config=None):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    response = client.responses.create(model=model_path, input=messages)
    output = response.output_text
    return output


def generate_hf(pipe, model_path, content, generation_config={}):
    messages = [
        {
            "role": "user",
            "content": content,
        },
    ]
    # response = pipe(messages, return_full_text=False, generate_kwargs=generation_config)
    response = pipe(messages, return_full_text=False, **generation_config)
    output = response[0]['generated_text']
    return output


def generate_google(client, model_path, content, generation_config=None):
    response = client.models.generate_content(
        model=model_path,
        contents=content,
    )
    output = response.text
    return output


def select_generate_func(api):
    if api == 'OpenAI':
        generate_func = generate_openai
    elif api == 'Huggingface':
        generate_func = generate_hf
    elif api == 'Google':
        generate_func = generate_google
    else:
        raise ValueError(f"Unsupported API: {api}")
    return generate_func

class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self._start = time.perf_counter()
        return self           # (optional) so you can read .elapsed later
    
    def __exit__(self, exc_type, exc, tb):
        self.elapsed = time.perf_counter() - self._start
        elapsed = self.format_hms(self.elapsed)
        print(f"[{self.label}]\t elapsed: {elapsed}")
            
    def format_hms(self, total_seconds):
        h = int(total_seconds // 3600)
        m = int((total_seconds % 3600) // 60)
        s = total_seconds % 60               # still a float now
        return f"{h:02d}:{m:02d}:{s:06.3f}"   # e.g. 00:01:02.357


def as_float(val):
    try:
        return float(val)
    except:
        return val


def as_int(val):
    try:
        return int(val)
    except:
        return val


def lower_str(val):
    if isinstance(val, str):
        return val.lower()
    else:
        return val


def parse_json_from_output(output):
    """Parse JSON from the output text of OpenAI response.
    If the output is a plain JSON string, it parses that directly.
    Otherwise, find the code block containing JSON and parse it.
    """
    try:
        if '```' in output:
            json_start_index = output.index('```')
            json_end_index = output.rindex('```')
            str_form57 = output[json_start_index:json_end_index].strip('`')
            if str_form57.startswith('json'):
                str_form57 = str_form57.replace('json', '', 1)
        else:
            str_form57 = output
        try:
            dict_form57 = json.loads(str_form57)
        except:
            dict_form57 = ast.literal_eval(str_form57)
    except:
        dict_form57 = {}
    return dict_form57


def text_binary_classification(pipe, prompt, dict_answer_choice, num_sim):
    list_output = pipe(prompt, max_new_tokens=1, num_return_sequences=num_sim, return_full_text=False)
    list_answer = list(map(lambda output: output['generated_text'].upper(), list_output))
    list_answer_filter = list(filter(lambda answer: answer in dict_answer_choice, list_answer))
    list_answer_map = list(map(lambda answer: dict_answer_choice[answer], list_answer_filter))
    return list_answer_map


def text_generation(pipe, prompt, max_new_tokens=4096):
    output = pipe(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    answer = output[0]['generated_text']
    return answer


def prepare_df_record(cfg):
    df_record = pd.read_csv(cfg.path.df_record, parse_dates=['Date'])
    df_record = df_record[df_record['State Name'].str.title().isin(cfg.scrp.target_states)]
    # df_record = df_record[df_record['County Name'].str.title().isin(cfg.scrp.target_counties)]
    df_record = df_record[df_record['Date'] >= cfg.scrp.start_date]
    assert df_record['Report Key'].is_unique
    return df_record


def prepare_df_crossing(cfg):
    df_crossing = pd.read_csv(cfg.path.df_crossing, low_memory=False)
    assert df_crossing['CROSSING'].is_unique
    df_crossing = df_crossing[df_crossing['STATENAME'].str.title().isin(cfg.scrp.target_states)]
    # df_crossing = df_crossing[df_crossing['COUNTYNAME'].str.title().isin(cfg.scrp.target_counties)]
    # df_crossing = df_crossing[df_crossing['CITYNAME'].str.lower().isin(list(cfg.crss.us_cities) + ['san francisco'])]
    df_crossing = df_crossing[df_crossing['CROSSINGCL'] == 2]
    df_crossing = df_crossing[df_crossing['POSXING'] == 1]
    # df_crossing = df_crossing[df_crossing['XPURPOSE'] == 1]
    df_crossing['EFFDATE'] = df_crossing['EFFDATE'].astype(str).str.zfill(6)
    df_crossing['EFFDATE'] = pd.to_datetime(df_crossing['EFFDATE'], format='%y%m%d')
    df_crossing[['REVISIONDA', 'LASTUPDATE']] = df_crossing[['REVISIONDA', 'LASTUPDATE']].apply(lambda col: pd.to_datetime(col, format='%m/%d/%Y %H:%M:%S AM').dt.date)
    
    ### EDA
    # pprint(df_crossing[df_crossing['TYPEXING'] == '3'][df_crossing['CITYNAME'] == 'RIVERSIDE']
    #        [['LATITUDE', 'LONGITUD', 'STREET', 'TYPEXING', 'PRVCAT']].iloc[20:30].to_records().tolist())
    # df_crossing['LATITUDE'].iloc[3].item()
    # df_crossing['y'].iloc[3].item()
    # df_crossing['LONGITUD'].iloc[0].item()
    # df_crossing['x'].iloc[0].item()
    # df_crossing['XPURPOSE'].value_counts()

    # list_useful = ['CROSSING', 'HIGHWAY', 'STREET', 'RAILROAD', 'RRDIV', 'RRSUBDIV', 'REASON', 'XPURPOSE', 'PRVCAT', 'TYPEXING', 'POSXING', 'PRVIND', 'PRVSIGN', 'LATITUDE', 'LONGITUD', 'LLSOURCE', 'WHISTBAN', 'INV_LINK']
    list_locinfo = ['CROSSING', 'HIGHWAY', 'STREET', 'RAILROAD', 'RRDIV', 'RRSUBDIV', 'REASON', 'XPURPOSE', 'PRVCAT', 'LATITUDE', 'LONGITUD', 'LLSOURCE']
    df_crossing = df_crossing[list_locinfo]
    df_crossing = df_crossing.reset_index(drop=True)

    return df_crossing


def prepare_df_image_cand(cfg):
    df_image_cand = pd.read_csv(cfg.path.df_image_cand)
    df_image_cand['img_id'] = df_image_cand['img_id'].apply(lambda x: str(int(x)) if pd.notna(x) else x)
    df_image_cand = df_image_cand.sort_values('crossing_id', ignore_index=True)
    df_image_cand['computed_rotation'] = df_image_cand['computed_rotation'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image_cand['mesh'] = df_image_cand['mesh'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image_cand['sfm_cluster'] = df_image_cand['sfm_cluster'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    # df_image_cand['detections'] = df_image_cand['detections'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image_cand['creator'] = df_image_cand['creator'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df_image_cand['camera_parameters'] = df_image_cand['camera_parameters'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    
    return df_image_cand


def prepare_df_image_ids_per_seq(cfg):
    df_image_ids_per_seq = pd.read_csv(cfg.path.df_image_ids_per_seq)
    df_image_ids_per_seq['img_ids'] = df_image_ids_per_seq['img_ids'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    return df_image_ids_per_seq


def prepare_df_image_seq(cfg):
    df_image_seq = pd.read_csv(cfg.path.df_image_seq)
    return df_image_seq


def prepare_df_retrieval(cfg):
    df_retrieval = pd.read_csv(cfg.path.df_retrieval, parse_dates=['pub_date'])
    assert df_retrieval['news_id'].is_unique, '==========Warning: News is not unique!!!==========='
    return df_retrieval


def prepare_df_retrieval_realtime(cfg):
    df_retrieval = pd.read_csv(cfg.path.df_retrieval_realtime, parse_dates=['pub_date', 'accident_date'])
    return df_retrieval


def prepare_df_match(cfg):
    df_match = pd.read_csv(cfg.path.df_match)
    df_match = df_match[df_match['match'] == 1]
    idx_content_match = df_match.columns.get_loc('content')
    df_match = df_match.iloc[:, :idx_content_match + 1] # type: ignore
    assert df_match['news_id'].is_unique, '==========Warning: News is not unique!!!==========='
    return df_match


def prepare_df_3D(cfg):
    df_3D = pd.read_csv(cfg.path.df_3D, parse_dates=['captured_at'])
    return df_3D


def prepare_dict_col_indexing(cfg):
    with open(cfg.path.dict_col_indexing, 'r') as f:
        dict_col_indexing = json5.load(f)
    return dict_col_indexing


def prepare_dict_idx_mapping(cfg):
    with open(cfg.path.dict_idx_mapping, 'r') as f:
        dict_idx_mapping = json5.load(f)
        dict_idx_mapping_inverse = {v: k for k, v in dict_idx_mapping.items()}
        if '' in dict_idx_mapping_inverse:
            dict_idx_mapping_inverse.pop('')
    return dict_idx_mapping, dict_idx_mapping_inverse


def prepare_dict_answer_places(cfg):
    with open(cfg.path.dict_answer_places, 'r') as f:
        dict_answer_places = json5.load(f)
    return dict_answer_places


def prepare_dict_form57(cfg):
    with open(cfg.path.form57_json, 'r') as f:
        dict_form57 = json.load(f)
    return dict_form57


def prepare_dict_form57_group(cfg):
    with open(cfg.path.form57_json_group, 'r') as f:
        dict_form57_group = json5.load(f)
    return dict_form57_group


def prepare_dict_bounding_box(cfg):
    with open(cfg.path.dict_bounding_box, "r") as f:
        dict_bounding_box = json.load(f)
    return dict_bounding_box
