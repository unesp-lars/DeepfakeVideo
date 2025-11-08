# ## THIS IS THE FRAMEWORK FILE
# ## RUN THE FILE TO:
# # 1. Calculate the number of frames in the video
# # 2. Split the video into frames (deconstruct.py)
# # 3. Calculate the frame loss in the process
# # 4. Process the frames in parallel inputting the frames into Nightshade and obtain the poisoned frames
# # 5. Reconstruct the video from the frames (reconstruct.py)

# from NightshadeVideo.reconstruct import frames_to_video
# from NightshadeVideo.deconstruct import extract_frames
# from pathlib import Path

# # DEFINE INPUT FILE NAME
# input_video = "test.mp4"


# frames_output_folder = "frames_" + input_video + "/"
# video_output_name = "poisoned_" + input_video
# Path(frames_output_folder).mkdir(exist_ok=True)

# # PRECISO FAZER OVERWRITE DA PASTA SE JÁ HOUVER IMAGENS? - APARENTEMENTE NÃO
# print("---------------------------------------------") # MELHORAR ISSO PARA COBRIR DINAMICAMENTE O ESPAÇO DA JANELA
# print("Splitting input video " + input_video + " into frames")
# print("The frames will be stored in " + frames_output_folder + ". The folder will be created if it can't be accessed")
# print("---------------------------------------------")


# # Deconstruct video into frames
# extract_frames(input_video, frames_output_folder)

# # Input in Nightshade


# # Reconstruct video from frames
# frames_to_video(frames_output_folder, video_output_name, framerate=30)