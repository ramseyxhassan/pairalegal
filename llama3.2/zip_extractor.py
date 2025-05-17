import os
import zipfile
import time

def is_file_fully_downloaded(filepath):
    initial_size = os.path.getsize(filepath)
    time.sleep(5)  # Wait 5 seconds and check size again
    final_size = os.path.getsize(filepath)
    return initial_size == final_size

def extract_zip_files(base_dir):
    for root, dirs, files in os.walk(base_dir):
        # Check if we're in an SFMA folder
        if os.path.basename(root).startswith("SFMA-"):
            for file in files:
                if file.endswith(".zip"):
                    zip_path = os.path.join(root, file)
                    print(f"Extracting: {zip_path}")

                    # Check if the file is fully downloaded before extracting
                    if not is_file_fully_downloaded(zip_path):
                        print(f"File {zip_path} is still downloading. Skipping for now.")
                        continue

                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(root)
                        print(f"Successfully extracted: {zip_path}")

                        # Remove the zip file after extraction
                        os.remove(zip_path)
                        print(f"Removed zip file: {zip_path}")
                    except zipfile.BadZipFile as e:
                        print(f"Bad zip file {zip_path}: {e}")
                    except Exception as e:
                        print(f"Error processing {zip_path}: {e}")

def main():
    # Change this to the actual path where your ZIP files are stored
    base_dir = r"C:\Developer\Workspace\llama3.2\data"
    insurance_types = ["state_farm_4.0_homeowners", "state_farm_19.0_personal_auto"]

    print("Starting ZIP file extractor. Press Ctrl+C to exit.")
    try:
        while True:
            for insurance_type in insurance_types:
                insurance_dir = os.path.join(base_dir, insurance_type)
                if os.path.exists(insurance_dir):
                    print(f"Processing: {insurance_dir}")
                    extract_zip_files(insurance_dir)

            print("Finished processing. Waiting before next scan...")
            time.sleep(30)  # Wait for 30 seconds before the next scan
    except KeyboardInterrupt:
        print("\nExiting ZIP file extractor.")

if __name__ == "__main__":
    main()
