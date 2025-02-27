import os
import boto3

GGACCESSKEYID = os.getenv("GGACCESSKEYID")
HMACKEY = os.getenv("HMACKEY")

bucket_name = 'gc-gha-demo-migrate-s3'
model_name = 'vit-human-pose/'
local_path = 'ml-models/' + model_name


def download_dir(local_path, model_name, google_access_key_id, google_access_key_secret):
    s3 = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url="https://storage.googleapis.com",
        aws_access_key_id=google_access_key_id,
        aws_secret_access_key=google_access_key_secret,
    )
    

    s3_prefix = 'models/' + model_name
    
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                if key['Key'] != 'models/vit-human-pose/':
                    s3_key = key['Key']

                    local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                    # os.makedirs(os.path.dirname(local_file), exist_ok=True)
                    s3.download_file(bucket_name, s3_key, local_file)
