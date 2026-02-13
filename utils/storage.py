import boto3
from botocore.client import Config
from config import settings
import os

s3_client = boto3.client(
    's3',
    endpoint_url=settings.s3_endpoint,
    aws_access_key_id=settings.s3_access_key,
    aws_secret_access_key=settings.s3_secret_key,
    config=Config(signature_version='s3v4')
)

def upload_to_s3(file_data: bytes, object_name: str) -> str:
    try:
        s3_client.put_object(
            Bucket=settings.s3_bucket,
            Key=object_name,
            Body=file_data
        )
        return f"{settings.s3_endpoint}/{settings.s3_bucket}/{object_name}"
    except Exception as e:
        raise Exception(f"S3 upload failed: {str(e)}")

def download_from_s3(object_name: str) -> str:
    local_path = f"/tmp/{os.path.basename(object_name)}"
    try:
        s3_client.download_file(settings.s3_bucket, object_name, local_path)
        return local_path
    except Exception as e:
        raise Exception(f"S3 download failed: {str(e)}")

def delete_from_s3(object_name: str):
    try:
        s3_client.delete_object(Bucket=settings.s3_bucket, Key=object_name)
    except Exception as e:
        raise Exception(f"S3 delete failed: {str(e)}")
