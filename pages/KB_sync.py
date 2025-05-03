import streamlit as st
import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
import logging
import toml
import time

# --- Page Config ---
st.set_page_config(page_title="Bedrock KB Sync", layout="wide")

SECRETS_FILE_PATH = ".streamlit/secrets.toml"

# --- Load AWS Credentials ---
try:
    secrets = toml.load(SECRETS_FILE_PATH)
    AWS_ACCESS_KEY = secrets.get("access_key_id")
    AWS_SECRET_KEY = secrets.get("secret_access_key")
    AWS_REGION = secrets.get("region")
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    AWS_ACCESS_KEY = None
    AWS_SECRET_KEY = None
    AWS_REGION = None

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- AWS Helper Functions ---
def get_aws_client(service_name, region_name, access_key, secret_key):
    if not access_key or not secret_key:
        st.error("AWS credentials missing. Cannot create client.")
        return None
    if not region_name:
        st.error("AWS Region is required but not provided. Cannot create client.")
        return None
    try:
        client = boto3.client(
            service_name,
            region_name=region_name,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        # Simple call to verify client (list_buckets for S3, list_knowledge_bases for bedrock-agent)
        if service_name == 's3':
            client.list_buckets()
        elif service_name == 'bedrock-agent':
            client.list_knowledge_bases()
        logging.info(f"Successfully created Boto3 client for {service_name} in {region_name}")
        return client
    except (NoCredentialsError, PartialCredentialsError) as e:
        st.error(f"AWS credentials issue: {e}")
        return None
    except ClientError as e:
        st.error(f"AWS Client Error for {service_name} in {region_name}: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error creating AWS client for {service_name}: {e}")
        return None

def upload_to_s3(s3_client, bucket_name, file_obj, object_name=None):
    if s3_client is None:
        st.error("S3 client is not available. Cannot upload.")
        return False, None
    if object_name is None:
        object_name = file_obj.name
    try:
        s3_client.upload_fileobj(file_obj, bucket_name, object_name)
        s3_uri = f"s3://{bucket_name}/{object_name}"
        logging.info(f"File '{object_name}' uploaded successfully to bucket '{bucket_name}'. URI: {s3_uri}")
        return True, s3_uri
    except ClientError as e:
        st.error(f"Error uploading file '{object_name}' to S3: {e}")
        return False, None
    except Exception as e:
        st.error(f"Unexpected error during S3 upload: {e}")
        return False, None

def start_bedrock_ingestion_job(bedrock_agent_client, knowledge_base_id, data_source_id):
    if bedrock_agent_client is None:
        st.error("Bedrock Agent client is not available. Cannot start ingestion job.")
        return False, None
    try:
        response = bedrock_agent_client.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id
        )
        ingestion_job = response.get('ingestionJob', {})
        job_id = ingestion_job.get('ingestionJobId', 'N/A')
        status = ingestion_job.get('status', 'N/A')
        logging.info(f"Started ingestion job for KB '{knowledge_base_id}', DS '{data_source_id}'. Job ID: {job_id}, Status: {status}")
        return True, ingestion_job
    except ClientError as e:
        st.error(f"Error starting Bedrock ingestion job: {e}")
        return False, None
    except Exception as e:
        st.error(f"Unexpected error starting Bedrock ingestion job: {e}")
        return False, None

# --- Streamlit App UI ---
st.title("⬆️ Sync AWS Bedrock Knowledge Base from S3 Upload")
st.info(
    "**Instructions:**\n"
    "1. **Configure AWS Credentials:** Ensure your AWS credentials (`access_key_id`, `secret_access_key`) and optionally a default `region` are set in `.streamlit/secrets.toml`.\n"
    "2. **Enter AWS Region:** Provide the AWS Region where your S3 bucket and Bedrock Knowledge Base reside.\n"
    "3. **Enter S3 Bucket Name:** Specify the S3 bucket linked to your Bedrock data source.\n"
    "4. **Enter Bedrock IDs:** Provide your Bedrock Knowledge Base ID and the specific Data Source ID.\n"
    "5. **Upload File(s):** Choose one or more files to upload/update in the Knowledge Base.\n"
    "6. **Click 'Upload and Start Sync'.**"
)

# --- User Inputs ---
st.sidebar.header("AWS Configuration")
default_region = AWS_REGION if AWS_REGION else os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
aws_region_input = st.sidebar.text_input("AWS Region", default_region)
s3_bucket_name = st.sidebar.text_input("S3 Bucket Name")
kb_id = st.sidebar.text_input("Knowledge Base ID")
ds_id = st.sidebar.text_input("Data Source ID")
st.sidebar.header("File Upload")
uploaded_files = st.sidebar.file_uploader("Choose file(s) to upload", accept_multiple_files=True)

# --- Main Logic ---
if st.sidebar.button("🚀 Upload and Start Sync"):
    valid_inputs = True
    if not aws_region_input:
        st.error("AWS Region is required.")
        valid_inputs = False
    if not s3_bucket_name:
        st.error("S3 Bucket Name is required.")
        valid_inputs = False
    if not kb_id:
        st.error("Bedrock Knowledge Base ID is required.")
        valid_inputs = False
    if not ds_id:
        st.error("Bedrock Data Source ID is required.")
        valid_inputs = False
    if not uploaded_files:
        st.error("Please upload at least one file.")
        valid_inputs = False
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        st.error("AWS credentials not loaded. Cannot proceed.")
        valid_inputs = False

    if valid_inputs:
        with st.status("Initializing AWS clients...", expanded=True) as status_bar:
            st.write(f"Target Region: {aws_region_input}")
            s3_client = get_aws_client('s3', aws_region_input, AWS_ACCESS_KEY, AWS_SECRET_KEY)
            bedrock_agent_client = get_aws_client('bedrock-agent', aws_region_input, AWS_ACCESS_KEY, AWS_SECRET_KEY)

            if s3_client and bedrock_agent_client:
                status_bar.update(label="AWS Clients Initialized.", state="running", expanded=True)
                st.write(f"Attempting to upload {len(uploaded_files)} file(s) to bucket '{s3_bucket_name}'...")
                all_uploads_successful = True
                successful_uploads_details = []
                upload_progress_bar = st.progress(0, text="Uploading files...")
                files_uploaded_count = 0

                for file_obj in uploaded_files:
                    st.write(f" Uploading '{file_obj.name}'...")
                    upload_success, s3_uri = upload_to_s3(s3_client, s3_bucket_name, file_obj, file_obj.name)
                    if upload_success:
                        st.write(f" ✅ Successfully uploaded '{file_obj.name}' to: `{s3_uri}`")
                        successful_uploads_details.append({"name": file_obj.name, "uri": s3_uri})
                        files_uploaded_count += 1
                    else:
                        st.write(f"  Failed to upload '{file_obj.name}'.")
                        all_uploads_successful = False
                    progress_percent = int((files_uploaded_count / len(uploaded_files)) * 100)
                    upload_progress_bar.progress(progress_percent, text=f"Uploading files... ({files_uploaded_count}/{len(uploaded_files)})")

                if all_uploads_successful:
                    upload_progress_bar.progress(100, text="All files uploaded successfully!")
                else:
                    upload_progress_bar.progress(progress_percent, text="File upload failed for one or more files.")

                if all_uploads_successful:
                    status_bar.update(label=f"All {len(uploaded_files)} files uploaded to S3. Starting Bedrock sync...", state="running", expanded=True)
                    st.success(f"Successfully uploaded {len(successful_uploads_details)} files.")
                    st.write(f"Triggering ingestion job for Knowledge Base '{kb_id}' / Data Source '{ds_id}'...")
                    sync_started, job_info = start_bedrock_ingestion_job(bedrock_agent_client, kb_id, ds_id)
                    if sync_started:
                        job_id = job_info.get('ingestionJobId', 'N/A')
                        initial_status = job_info.get('status', 'N/A')
                        st.write(f" Bedrock sync (Ingestion Job) started.")
                        st.json(job_info)
                        status_bar.update(label=f"✅ Sync started! Job ID: {job_id}, Status: {initial_status}", state="complete", expanded=False)
                        st.success(f"Process completed: {len(successful_uploads_details)} file(s) uploaded and Bedrock sync initiated (Job ID: {job_id}).")
                        st.info("Note: The sync process runs asynchronously in AWS. Check the AWS Bedrock console for the final status of the ingestion job.")
                    else:
                        status_bar.update(label=" Sync initiation failed.", state="error", expanded=True)
                        st.error("Could not start the Bedrock ingestion job after successful uploads.")
                else:
                    status_bar.update(label=f" S3 Upload Failed for one or more files.", state="error", expanded=True)
                    st.error("File upload process failed. Check errors above. Bedrock sync was not started.")
            else:
                status_bar.update(label=" AWS Client Initialization Failed.", state="error", expanded=True)
                st.error("Could not initialize necessary AWS clients. Check configuration and permissions.")

# --- Footer/Info ---
st.sidebar.markdown("---")
st.sidebar.markdown("Created with [Streamlit](https://streamlit.io) and [Boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)")
current_local_time = time.strftime('%Y-%m-%d %H:%M:%S %Z')
st.sidebar.markdown(f"App Time: {current_local_time}")
