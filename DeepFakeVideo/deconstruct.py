"""
First part of DeepFake Video.

This part is responsible for extracting valid data from concept A (source) for poisoning.

In this step, the Video-CLIP network (a pre-trained network that associates videos with texts and enables this 'translation' from one to the other)
validates the videos we want to poison with the algorithm, checking them against the concept (class) passed as a hyperparameter to the model (e.g., "this is a video of a _cat_").

For each selected video, the script saves its extracted frames in a dedicated output folder and stores the corresponding embeddings (vector representations) in a *pickle* file,
associating each embedding with its video for later use by DeepFake Video and test networks.

"""

import sys
import os
import glob
import shutil
import random

import numpy as np
import pickle
import pandas as pd
import cv2

from PIL import Image
import torch
from torchvision import transforms

from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel  # HuggingFace importing models

# Baixando o VideoCLIP-XL-v2 com Huggingface
VIDEOCLIPXL_PATH = snapshot_download(repo_id="alibaba-pai/VideoCLIP-XL")
VIDEOCLIPXLV2_PATH = snapshot_download(
    repo_id="alibaba-pai/VideoCLIP-XL-v2"
)  # new weights for VideoCLIP-XL
sys.path.insert(0, VIDEOCLIPXL_PATH)  # Setting the PATH for the imports


# CONFIGURAÇÕES DE DIRETÓRIO
SOURCE_CONCEPT = "PlayingPiano"
CONCEPT_TEXTS = [
    # "Um vídeo de uma pessoa tocando piano",
    # "Um vídeo de um piano sendo tocado",
    # "Um piano sendo tocado",
    # "Piano",
    "Um vídeo contendo um piano",
]  # Podem ser um ou mais textos
DATA_DIR = f"./data/{SOURCE_CONCEPT}"
OUTPUT_DIR = f"/output_embeddings/{SOURCE_CONCEPT}"
MAX_VIDEOS = None  # se None, todos os vídeos serão analisados

# ao selecionar vídeos candidatos, este é o número máximo a ser considerado
MAX_CANDIDATES = 5

from modeling import (
    VideoCLIP_XL,
)  # Ref: https://huggingface.co/alibaba-pai/VideoCLIP-XL
from utils.text_encoder import text_encoder

# CONFIGURAÇÕES DO MODELO
MODEL_NAME = "alibaba-pai/VideoCLIP-XL"
WEIGHTS_PATH = VIDEOCLIPXLV2_PATH  # or VIDEOCLIPXL_PATH
# Considerar que PASSOS_AMOSTRAGEM = MAX(1, TOTAL / FNUM)
FNUM = 8
TYPE_SAMPLING = "uniform"
SIM_THRESHOLD = 0.20  # limiar mínimo de similaridade

# Carregando o modelo
print(f"Carregando modelo {MODEL_NAME}...")
model = VideoCLIP_XL() # type: ignore
state_dict = torch.load(
    # os.path.join(WEIGHTS_PATH, "VideoCLIP-XL.bin"),
    os.path.join(WEIGHTS_PATH, "VideoCLIP-XL-v2.bin"),
    map_location="cpu",
)
model.load_state_dict(state_dict)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()


# Normalização de quadros para o VideoCLIP-XL
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def sample_frames_with_normalization(video_path, fnum=1000, type_sample="uniform"):
    """
    Extrai frames de um vídeo, uniformemente distribuídos ao longo do tempo.
    Também, os frames são normalizados para o VideoCLIP-XL, como no redimensionamento
    para 224x224 e transposição para tensor da rede.

    Args:
    - video_path (str): O caminho para o vídeo de entrada.
    - fnum (int): O intervalo de passos dos frames a serem extraídos. Se fnum=1, todos os frames serão amostrados.
    - type_sample (str): O tipo de amostragem a ser utilizado. Pode ser 'uniform' ou 'gaussian'.

    Returns:
    - Torch.tensor: Um tensor contendo os frames extraídos e normalizados, com formato (B, T, C, H, W)
    """
    frames = sample_frames(video_path, fnum=fnum, type_sample=type_sample)

    if frames is None:
        return None

    # Normalizando os frames amostrados
    normalized_frames = []
    for fr in frames:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)  # ou fr[:,:,::-1]  # BGR -> RGB
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), (0, 1))  # normalizando
        normalized_frames.append(fr)

    # Convertendo para tensor
    normalized_frames = np.concatenate(normalized_frames, axis=1)
    normalized_frames = np.transpose(
        normalized_frames,
        (0, 1, 4, 2, 3),  # [B, T, C, H, W]
    )
    normalized_frames = torch.from_numpy(normalized_frames)

    return normalized_frames


def sample_frames(video_path, fnum=1000, type_sample="uniform"):
    """
    Extrai frames de um vídeo, uniformemente distribuídos ao longo do tempo.
    Os frames NÃO são normalizados ou sofrem qualquer transformação
    ou codificação de imagens.

    Args:
    - video_path (str): O caminho para o vídeo de entrada.
    - fnum (int): O intervalo de passos dos frames a serem extraídos. Se fnum=1, todos os frames serão amostrados.
    - type_sample (str): O tipo de amostragem a ser utilizado. Pode ser 'uniform' ou 'gaussian'.

    Returns:
    - list(frames): Uma lista de frames extraídos do vídeo.
    """
    # Carregando o vídeo
    video = cv2.VideoCapture(video_path)
    # fps = video.get(cv2.CAP_PROP_FPS)
    # total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()

    # Caso tenha dado erro, returna nulo
    if len(frames) == 0:
        return None

    # Amostrando os frames com base em passos e no tipo de amostragem
    step = max(1, len(frames) // fnum)
    if type_sample == "uniform":
        frames = frames[::step][:fnum]
    elif type_sample == "gaussian":
        total_frames = int(len(frames))
        mean = total_frames / 2
        std_dev = mean * 0.4  # 20% para cada lado
        if total_frames <= fnum:
            frame_indices = list(range(total_frames))
        # Seleciona apenas o intervalo central (exclui 20% das margens)
        start = int(total_frames * 0.2)
        end = int(total_frames * 0.8)
        if (end - start) < fnum:
            frame_indices = list(range(start, end))
        else:
            # Amostragem uniforme dentro do intervalo central
            frame_indices = np.linspace(start, end - 1, fnum, dtype=int)
        frame_indices = list(frame_indices)
        frames = [frames[i] for i in frame_indices]

    return frames


def get_video_embedding(video_path, fnum=1000, type_sample="uniform"):
    """
    Extrai embedding do vídeo usando VideoCLIP-XL.

    Args:
    - video_path (str): O caminho para o vídeo de entrada.
    - fnum (int): O número de frames a serem extraídos do vídeo.
    - type_sample (str): O tipo de amostragem a ser utilizado. Pode ser 'uniform' ou 'gaussian'.

    Output:
    - embedding (np.ndarray): O embedding extraído do vídeo.
    """
    normalized_frames = sample_frames_with_normalization(
        video_path, fnum=fnum, type_sample=type_sample,
    )
    if normalized_frames is None:
        return None

    # Entrando no modo de avaliação da rede
    with torch.no_grad():
        video_inputs = normalized_frames.float().to(device)
        video_features = model.vision_model.get_vid_features(video_inputs).float()
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

    return video_features.cpu().numpy()


def get_text_embedding(texts):
    """
    Extrai embedding de uma lista de textos usando VideoCLIP-XL.

    Args:
    - text (list[str]): A lista de textos de entrada.

    Output:
    - embedding (np.ndarray): Os embeddings extraídos dos textos.
    """
    with torch.no_grad():
        text_inputs = text_encoder.tokenize(texts, truncate=True).to(device)
        text_features = model.text_model.encode_text(text_inputs).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def process_videos(
    video_folder,
    concept_texts,
    max_videos=None,
    max_candidates=100,
    fnum=1000,
    type_sample="uniform",
):
    """
    Processa vídeos em um diretório, extraindo embeddings e salvando frames.

    Args:
    - video_folder (str): Caminho para a pasta contendo os vídeos.
    - concept_texts (list[str]): Lista de textos a ser utilizado para a extração de embeddings relativos ao conceito.
    - max_videos (int, opcional): Número máximo de vídeos a serem processados. Se None, processa todos os vídeos.
    - max_candidates (int, opcional): Número máximo de vídeos candidatos a serem selecionados. Padrão é 100. Caso não hajam 100 amostras, seleciona todas disponíveis.
    - fnum (int, opcional): Número de frames a serem extraídos de cada vídeo. Padrão é 1.
    - type_sample (str, opcional): Tipo de amostragem a ser utilizado. Pode ser "uniform" ou "gaussian". Padrão é "uniform".
    """

    # Obtendo o caminho absoluto dos vídeos
    video_files = glob.glob(
        os.path.join(video_folder, "*.avi")
    )  # Na base escolhida, os vídeos tem formato .avi
    if max_videos:
        video_files = random.sample(video_files, min(len(video_files), max_videos))

    # Obtendo o embedding do prompt do Conceito A para cálculo de similaridade
    embeddings_dict = {}
    text_emb = get_text_embedding(concept_texts)

    # Obtendo os candidatos a partir da similaridade de cosseno dos vídeos com o prompt do Conceito A
    candidates_paths = []
    best_sims = []
    best_texts = []

    for video_path in video_files:
        print(f"Processando: {video_path}")
        try:
            video_emb = get_video_embedding(
                video_path=video_path,
                fnum=fnum,
                type_sample=type_sample,
            )

            # sim = cosine_similarity([video_emb], text_emb)[0][0]  # type: ignore
            sim = cosine_similarity(text_emb, video_emb)  # type: ignore
            # sim = float(np.dot(video_emb, text_emb.T))  # type: ignore

            # Obtendo a melhor similaridade e prompt referente a ela
            best_sim = np.max(sim)
            best_text = concept_texts[np.argmax(sim)]
            print(f" → Melhor Similaridade: {best_sim:.3f}")

            if best_sim >= SIM_THRESHOLD:
                # Se a similaridade foi maior que o limiar, salva o caminho do vídeo
                # para posterior análise
                candidates_paths.append(video_path)
                best_sims.append(best_sim)
                best_texts.append(best_text)
                print(f" → Adicionado como candidato.")

        except Exception as e:
            print(f"Erro no vídeo {video_path}: {e}")

    # Com a lista de vídeos candidatos obtida, selecionamos as melhores X amostras
    if candidates_paths:
        print("\n" + "-" * 15 + " INÍCIO DOS RESULTADOS " + "-" * 15 + "\n")
        print(f"Total de candidatos encontrados: {len(candidates_paths)}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        # selected_videos = random.sample(
        #     candidates_paths, min(len(candidates_paths), max_candidates)
        # )
        num_select = min(len(candidates_paths), max_candidates)
        top_indices = np.argsort(best_sims)[-num_select:][::-1]  # índices dos maiores
        selected_videos = [candidates_paths[i] for i in top_indices]
        selected_sims = [best_sims[i] for i in top_indices]
        selected_texts = [best_texts[i] for i in top_indices]
        print(f"Vídeos selecionados: {selected_videos}")
        print(f"Quantidade de vídeos selecionados: {len(selected_videos)}")

        print(f"Iniciando processamento dos vídeos selecionados...")
        # Para cada vídeo selecionado, extraímos os frames e salvamos os embeddings
        for idx, video_path in enumerate(selected_videos):
            frames = sample_frames(video_path, type_sample=type_sample)
            if frames is not None:
                print(f" → Salvando frames do vídeo {video_path}...")
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                frame_dir = os.path.join(OUTPUT_DIR, base_name)
                os.makedirs(frame_dir, exist_ok=True)

                for i, frame in enumerate(frames):
                    frame = Image.fromarray(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    )  # Coloração RGB
                    # frame.save(os.path.join(frame_dir, f"frame_{i:03d}.jpg"))
                    frame.save(os.path.join(frame_dir, f"{i:03d}.jpg"))

                # Salvar embedding
                embeddings_dict[base_name] = {
                    "embedding": get_video_embedding(video_path, fnum=fnum, type_sample=type_sample),  # type: ignore
                    "text": selected_texts[idx],
                    "similarity": selected_sims[idx],
                }
                print(f" → Amostra candidata salva.")
            else:
                raise Exception(
                    f"Erro ao processar a amostra candidata {video_path}. Tente novamente."
                )
        # Salvar todos embeddings em pickle
        with open(os.path.join(OUTPUT_DIR, "video_embeddings.pkl"), "wb") as f:
            pickle.dump(embeddings_dict, f)
    else:
        raise Exception(
            "Não foram encontradas amostras candidatas para o processo. Tente novamente."
        )
    print(
        f"Processamento concluído! {len(embeddings_dict)} vídeos salvos em {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    process_videos(
        video_folder=DATA_DIR,
        concept_texts=CONCEPT_TEXTS,
        max_videos=MAX_VIDEOS,
        max_candidates=MAX_CANDIDATES,
        fnum=FNUM,
        type_sample=TYPE_SAMPLING,
    )
