import os
import sys
from PIL import Image

def make_gif(folder_path, output_file="output.gif", duration=100):
    # Get all PNG files from the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    png_files.sort()  # Sort files by name to maintain order

    if not png_files:
        print("No PNG files found in the folder.")
        return

    # Open images and store them in a list
    images = [Image.open(os.path.join(folder_path, file)) for file in png_files]

    # Save images as a GIF
    images[0].save(
        output_file,
        save_all=True,
        append_images=images[1:],  # Append remaining images
        duration=duration,  # Duration between frames in milliseconds
        loop=0  # Infinite loop
    )

    print(f"GIF created successfully and saved as {output_file}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python make_gif.py <folder_path> [output_file] [duration]")
        sys.exit(1)

    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.gif"
    duration = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    make_gif(folder_path, output_file, duration)
