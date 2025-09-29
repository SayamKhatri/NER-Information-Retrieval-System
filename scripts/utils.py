import boto3
import os

def get_client():
    s3 = boto3.client('s3')
    return s3

def download_data(bucket_name, key_name):
    s3 = get_client()
    os.makedirs('data', exist_ok=True)
    local_path = os.path.join('data', os.path.basename(key_name))
    s3.download_file(bucket_name, key_name, local_path)
    
def save_data(bucket_name, local_path, s3_prefix="processed-data/"):
    s3 = get_client()
    s3_key = os.path.join(s3_prefix, os.path.basename(local_path)).replace("\\", "/")
    s3.upload_file(local_path, bucket_name, s3_key)








