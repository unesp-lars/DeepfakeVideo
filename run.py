# nightshade_video_step1.py

import os
import glob
import shutil
import random
import numpy as np
import pickle
import pandas as pd
import cv2
import sys
import warnings
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download


print("Baixando modelos VideoCLIP...")
VIDEOCLIPXL_PATH = snapshot_download(repo_id="alibaba-pai/VideoCLIP-XL")
VIDEOCLIPXLV2_PATH = snapshot_download(repo_id="alibaba-pai/VideoCLIP-XL-v2")

print(f"Caminho VideoCLIP-XL: {VIDEOCLIPXL_PATH}")
print(f"Caminho VideoCLIP-XL-v2: {VIDEOCLIPXLV2_PATH}")

sys.path.insert(0, VIDEOCLIPXL_PATH)

#Importações Específicas do Modelo
try:
    from modeling import VideoCLIP_XL
    from utils.text_encoder import text_encoder
except ImportError:
    print(f"Erro: Não foi possível importar 'modeling' ou 'utils.text_encoder' do caminho {VIDEOCLIPXL_PATH}")
    print("Verifique se o download do snapshot_download foi bem-sucedido.")
    sys.exit(1)

# Normalização de quadros para o VideoCLIP-XL
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

def normalize(data):
    """Normaliza os dados da imagem."""
    return (data / 255.0 - v_mean) / v_std

def sample_frames_with_normalization(video_path, fnum=1000, type_sample="uniform"):
    """
    Extrai frames de um vídeo, uniformemente distribuídos ao longo do tempo.
    Também, os frames são normalizados para o VideoCLIP-XL.
    """
    frames = sample_frames(video_path, fnum=fnum, type_sample=type_sample)

    if frames is None:
        return None

    # Normalizando os frames amostrados
    normalized_frames = []
    for fr in frames:
        fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        fr = cv2.resize(fr, (224, 224))
        fr = np.expand_dims(normalize(fr), (0, 1))
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
    Os frames NÃO são normalizados ou sofrem qualquer transformação.
    """
    # Carregando o vídeo
    video = cv2.VideoCapture(video_path)
    frames = []
    while True:
        success, frame = video.read()
        if not success:
            break
        frames.append(frame)
    video.release()

    if len(frames) == 0:
        return None

    step = max(1, len(frames) // fnum)
    if type_sample == "uniform":
        frames = frames[::step][:fnum]
    elif type_sample == "gaussian":
        total_frames = int(len(frames))
        if total_frames <= fnum:
            frame_indices = list(range(total_frames))
        else:
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

def get_video_embedding(model, device, video_path, fnum=1000, type_sample="uniform"):
    """
    Extrai embedding do vídeo usando VideoCLIP-XL.
    """
    normalized_frames = sample_frames_with_normalization(
        video_path,
        fnum=fnum,
        type_sample=type_sample,
    )
    if normalized_frames is None:
        return None

    # Entrando no modo de avaliação da rede
    with torch.no_grad():
        video_inputs = normalized_frames.float().to(device)
        video_features = model.vision_model.get_vid_features(video_inputs).float()
        video_features = video_features / video_features.norm(dim=-1, keepdim=True)

    return video_features.cpu().numpy()


def get_text_embedding(model, device, texts):
    """
    Extrai embedding de uma lista de textos usando VideoCLIP-XL.
    """
    with torch.no_grad():
        text_inputs = text_encoder.tokenize(texts, truncate=True).to(device)
        text_features = model.text_model.encode_text(text_inputs).float()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def main():
    """Função principal para executar o Passo 1 de extração."""
    
    warnings.filterwarnings("ignore")  # ignora avisos do PyTorch
    print(f"Versão do PyTorch: {torch.__version__}")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # --- LISTA DE VIDEOS
    classes = [
        "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
        "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
        "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
        "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
        "CleanAndJerk", "CliffDiving", "CricketBowling", "CricketShot", "CuttingInKitchen",
        "Diving", "Drumming", "Fencing", "FieldHockeyPenalty", "FloorGymnastics",
        "FrisbeeCatch", "FrontCrawl", "GolfSwing", "Haircut", "Hammering",
        "HammerThrow", "HandstandPushups", "HandstandWalking", "HeadMassage", "HighJump",
        "HorseRace", "HorseRiding", "HulaHoop", "IceDancing", "JavelinThrow",
        "JugglingBalls", "JumpingJack", "JumpRope", "Kayaking", "Knitting",
        "LongJump", "Lunges", "MilitaryParade", "Mixing", "MoppingFloor",
        "Nunchucks", "ParallelBars", "PizzaTossing", "PlayingCello", "PlayingDaf",
        "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingSitar",
        "PlayingTabla", "PlayingViolin", "PoleVault", "PommelHorse", "PullUps",
        "Punch", "PushUps", "Rafting", "RockClimbingIndoor", "RopeClimbing",
        "Rowing", "SalsaSpin", "ShavingBeard", "Shotput", "SkateBoarding",
        "Skiing", "Skijet", "SkyDiving", "SoccerJuggling", "SoccerPenalty",
        "StillRings", "SumoWrestling", "Surfing", "Swing", "TableTennisShot",
        "TaiChi", "TennisSwing", "ThrowDiscus", "TrampolineJumping", "Typing",
        "UnevenBars", "VolleyballSpiking", "WalkingWithDog", "WallPushups", "WritingOnBoard",
        "YoYo"
    ]
    
    # CONFIGURAÇÕES GERAIS
    MAX_VIDEOS = None  # se None, todos os vídeos serão analisados
    MAX_CANDIDATES = 5
    MODEL_NAME = "alibaba-pai/VideoCLIP-XL"
    # WEIGHTS_PATH é o VIDEOCLIPXLV2_PATH baixado
    FNUM = 8
    TYPE_SAMPLING = "uniform"
    SIM_THRESHOLD = 0.20  # limiar mínimo de similaridade


    print(f"Carregando modelo {MODEL_NAME}...")
    model = VideoCLIP_XL()
    state_dict = torch.load(
        os.path.join(VIDEOCLIPXLV2_PATH, "VideoCLIP-XL-v2.bin"),
        map_location="cpu",
    )
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print("Modelo carregado com sucesso.")

    #LOOP PRINCIPAL SOBRE AS CLASSES 
    for concept_name in classes:
        print("\n" + "="*20 + f" PROCESSANDO VIDEO: {concept_name} " + "="*20 + "\n")
        
        SOURCE_CONCEPT = concept_name
        CONCEPT_TEXTS = [
            f"Um vídeo de {SOURCE_CONCEPT}",
            f"A video of {SOURCE_CONCEPT}"
        ]
        DATA_DIR = f"./videos/val/{SOURCE_CONCEPT}"
        OUTPUT_DIR = f"./output_embeddings/{SOURCE_CONCEPT}"

        print(f"Procurando vídeos em: {DATA_DIR}")
        video_files = glob.glob(os.path.join(DATA_DIR, "*.avi"))
        
        if not video_files:
            print(f"Aviso: Nenhum vídeo .avi encontrado em {DATA_DIR}")
            print(f"Pulando para o próximo video...")
            continue # Pula para a próxima classe

        if MAX_VIDEOS:
            video_files = random.sample(video_files, min(len(video_files), MAX_VIDEOS))

        # Obtendo o embedding
        embeddings_dict = {} # Reinicia para cada video
        text_emb = get_text_embedding(model, device, CONCEPT_TEXTS)

        # Obtendo os candidatos
        candidates_paths = []
        best_sims = []
        best_texts = []
        for video_path in video_files:
            print(f"Processando: {video_path}")
            try:
                video_emb = get_video_embedding(
                    model=model,
                    device=device,
                    video_path=video_path,
                    fnum=FNUM,
                    type_sample=TYPE_SAMPLING,
                )
                if video_emb is None:
                    print(f" → Falha ao ler o vídeo.")
                    continue

                sim = cosine_similarity(text_emb, video_emb)
                best_sim = np.max(sim)
                best_text = CONCEPT_TEXTS[np.argmax(sim)]
                print(f" → Melhor Similaridade: {best_sim:.3f}")

                if best_sim >= SIM_THRESHOLD:
                    candidates_paths.append(video_path)
                    best_sims.append(best_sim)
                    best_texts.append(best_text)
                    print(f" → Adicionado como candidato.")

            except Exception as e:
                print(f"Erro no vídeo {video_path}: {e}")

        # Processando os candidatos selecionados
        if candidates_paths:
            print("\n" + "-" * 15 + f" RESULTADOS PARA {SOURCE_CONCEPT} " + "-" * 15 + "\n")
            print(f"Total de candidatos encontrados: {len(candidates_paths)}")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            num_select = min(len(candidates_paths), MAX_CANDIDATES)
            top_indices = np.argsort(best_sims)[-num_select:][::-1]  # índices dos maiores

            selected_videos = [candidates_paths[i] for i in top_indices]
            selected_sims = [best_sims[i] for i in top_indices]
            selected_texts = [best_texts[i] for i in top_indices]
            
            print(f"Vídeos selecionados: {selected_videos}")
            print(f"Quantidade de vídeos selecionados: {len(selected_videos)}")
            print(f"Iniciando processamento dos vídeos selecionados...")
            
            # Para cada vídeo selecionado, extrai os frames e salvamos os embeddings
            for idx, video_path in enumerate(selected_videos):
                frames = sample_frames(video_path, type_sample=TYPE_SAMPLING)
                if frames is not None:
                    print(f" → Salvando frames do vídeo {video_path}...")
                    base_name = os.path.splitext(os.path.basename(video_path))[0]
                    frame_dir = os.path.join(OUTPUT_DIR, base_name)
                    os.makedirs(frame_dir, exist_ok=True)

                    for i, frame in enumerate(frames):
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        frame_pil.save(os.path.join(frame_dir, f"{i:03d}.jpg"))

                    # Salvar embedding
                    video_embedding = get_video_embedding(
                        model=model,
                        device=device,
                        video_path=video_path, 
                        fnum=FNUM, 
                        type_sample=TYPE_SAMPLING
                    )
                    
                    embeddings_dict[base_name] = {
                        "embedding": video_embedding,
                        "text": selected_texts[idx],
                        "similarity": selected_sims[idx],
                    }
                    print(f" → Amostra candidata salva.")
                else:
                    print(f"Erro ao processar a amostra candidata {video_path}.")

            pickle_path = os.path.join(OUTPUT_DIR, "video_embeddings.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(embeddings_dict, f)
            print(f"Dicionário de embeddings salvo em: {pickle_path}")
        else:
            print(f"Não foram encontradas amostras candidatas para o conceito {SOURCE_CONCEPT}.")
        
        print(f"Processamento do conceito {SOURCE_CONCEPT} concluído! {len(embeddings_dict)} vídeos salvos em {OUTPUT_DIR}")

    print("\n" + "="*20 + " PROCESSAMENTO TOTAL CONCLUÍDO " + "="*20 + "\n")


if __name__ == "__main__":
    main()