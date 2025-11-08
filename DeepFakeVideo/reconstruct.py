"""
Last part of DeepFake Video.

This part is responsible for reconstructing poisoned video data from the extracted frames and their embeddings.

The process involves taking the individual frames saved during the deconstruction phase, along with the poisoned frames, and using them to create a new poisoned video.

"""

# import av
# import glob

# def frames_to_video(input_folder, output_path, framerate):
#     # Create output container and stream
#     container = av.open(output_path, mode='w')
#     stream = container.add_stream('libx264', rate=framerate)
#     stream.width = 1280  # Set frame width (adjust to your frames)
#     stream.height = 720  # Set frame height
#     stream.pix_fmt = 'yuv420p'

#     # Iterate over sorted frames
#     frame_files = sorted(glob.glob(f"{input_folder}/frame_*.png"))

#     for frame_path in frame_files:
#         # Open the image file as an AV container
#         img_container = av.open(frame_path)
#         img_stream = img_container.streams.video[0]

#         # Decode the image frame and convert to VideoFrame
#         for frame in img_container.decode(img_stream):
#             frame = frame.reformat(format='yuv420p')  # Match the stream's pixel format
#             packet = stream.encode(frame)
#             container.mux(packet)

#     # Flush the encoder
#     for packet in stream.encode():
#         container.mux(packet)

#     container.close()
