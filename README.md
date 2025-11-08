# Deepfake Vídeo

Este repositório contém o código para a abordagem de utilização do **Nightshade em vídeos**

Para usar o repositório:
1. Instale os *requirements* 
2. Use o comando `python run.py` ou `python3 run.py`

O código é separado entre `deconstruct.py`, `poison.py` e `reconstruct.py`, respectivamente  dividindo o vídeo em frames, aplicando o envenenamento e posteriormente reconstruindo o vídeo.

Ao rodar será criada uma pasta, caso ela ainda não exista, para armazenar os frames do vídeo, também, ao final da execução haverá um arquivo de vídeo decorrente da reconstrução dos frames.
