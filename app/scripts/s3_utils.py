import os
import boto3

bucket_name = 'ml-ops-130125-1558'

# local_path = 'vit-human-pose'
# s3_prefix = 'models/vit-human-pose'

s3 = boto3.client('s3')


def download_dir(local_path, model_name):
    s3_prefix = 'models/' + model_name
    
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']

                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                # os.makedirs(os.path.dirname(local_file), exist_ok=True)

                s3.download_file(bucket_name, s3_key, local_file)
                

def upload_img_to_s3(file_path, s3_prefix='test/images', s3_object_name=None):
    if s3_object_name is None:
        s3_object_name = os.path.basename(file_path)  # if no s3 object name is passed, use the same local's filename 
    
    s3_object_name = f"{s3_prefix}/{s3_object_name}"
    
    s3.upload_file(file_path, bucket_name, s3_object_name)
    
    # generate s3 url of the uploaded image
    response = s3.generate_presigned_url('get_object',
                                         Params={
                                             "Bucket": bucket_name,
                                             "Key": s3_object_name,
                                         },
                                         ExpiresIn=600)  # url expires in 600 seconds
    return response  
    