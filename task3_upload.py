import os
from datasets import Dataset, Image
from huggingface_hub import login

def create_and_upload_dataset():
    # 1. Ask for credentials
    print("=== Hugging Face Dataset Uploader ===")
    token = input("1. Paste your Hugging Face WRITE token here: ")
    login(token=token)
    
    hf_username = input("2. Enter your Hugging Face username (e.g., Shauryaaa05): ")
    repo_name = input("3. Enter the dataset name you want to create (e.g., uav-drone-detections): ")
    repo_id = f"{hf_username}/{repo_name}"

    # 2. Gather the images from our detections folder
    print("\nGathering images from 'detections' folder...")
    image_paths = []
    det_dir = "detections"
    
    for img_name in os.listdir(det_dir):
        if img_name.endswith(".jpg"):
            image_paths.append(os.path.join(det_dir, img_name))
            
    if not image_paths:
        print("Error: No images found in the 'detections' folder!")
        return

    # 3. Create a Hugging Face Dataset object
    # We use the Image() feature so Hugging Face knows these are pictures, not just raw bytes
    print(f"Found {len(image_paths)} images. Creating Parquet dataset...")
    dataset = Dataset.from_dict({"image": image_paths}).cast_column("image", Image())

    # 4. Upload directly to Hugging Face
    print(f"Uploading dataset to https://huggingface.co/datasets/{repo_id} ...")
    dataset.push_to_hub(repo_id)
    
    print("\nSuccess! Your dataset is now live on Hugging Face in Parquet format.")

if __name__ == "__main__":
    create_and_upload_dataset()