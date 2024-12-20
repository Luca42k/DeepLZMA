import os
from PIL import Image

def batch_convert_to_ppm():
    """
    Converts all PNG images in the 'Xrays' folder to PPM format
    and saves them to the 'ppmXrays' folder.
    """
    # Define input and output folder paths
    input_dir = os.path.join(os.getcwd(), "Xrays")  # Input folder
    output_dir = os.path.join(os.getcwd(), "ppmXrays")  # Output folder

    # Create output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all PNG files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png'):  # Check if the file is a PNG
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name.replace('.png', '.ppm'))
            try:
                # Open the PNG image and convert to RGB mode
                img = Image.open(input_path).convert('RGB')
                # Save as PPM format in the output folder
                img.save(output_path, format='PPM')
                print(f"Converted {file_name} to {output_path}")
            except Exception as e:
                print(f"Failed to convert {file_name}: {e}")

    # Print completion message
    print("Conversion completed successfully. All PPM files are saved in the 'ppmXrays' folder.")

# Run the function
batch_convert_to_ppm()
