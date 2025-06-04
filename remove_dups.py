# removes duplicates of iPhone photos
# This script deletes duplicate files that have a "-1" suffix in their name, if they are identical to the original file.
# It uses SHA256 hashing to compare files.
import os
import hashlib

def hash_file(filepath):
    """Generate SHA256 hash of the file"""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def find_original(file_path):
    """Given a file ending in -1, try to find the original file"""
    dir_name, file_name = os.path.split(file_path)
    if '-1' not in file_name:
        return None
    base_name = file_name.replace('-1', '', 1)
    potential_path = os.path.join(dir_name, base_name)
    if os.path.exists(potential_path):
        return potential_path
    return None

def delete_identical_duplicates(folder):
    deleted_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if '-1' in file:
                duplicate_path = os.path.join(root, file)
                original_path = find_original(duplicate_path)
                if original_path and os.path.isfile(original_path):
                    if hash_file(duplicate_path) == hash_file(original_path):
                        print(f"Deleting duplicate: {duplicate_path}")
                        os.remove(duplicate_path)
                        deleted_files.append(duplicate_path)
    print(f"\nDone. Deleted {len(deleted_files)} duplicate files.")
    return deleted_files

# 🔁 Replace this path with the folder where your iPhone photos are
target_folder = r"D:\Users\Carlos\Pictures"

delete_identical_duplicates(target_folder)
