from flask import Flask, request, jsonify
import sys
import os
from utils.logging import configure_logging, get_logger

# Configure logging
configure_logging()
logger = get_logger(__name__)

# Ensure the base directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrate_workflow import orchestrate_workflow
from pathlib import Path
import boto3
import pandas as pd
from config import USE_CLOUD
from utils.usage_tracker import check_quota, increment_request
from utils.user_profile import get_user_tier

if not USE_CLOUD:
    from storage.local_backend import (
        load_datalake_dfs as local_load_datalake_dfs,
        LocalStorage,
    )
    local_storage = LocalStorage()

app = Flask(__name__)


@app.after_request
def add_csp_header(response):
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

def load_datalake_dfs():
    if USE_CLOUD:
        s3_client_east_2 = boto3.client('s3', region_name='us-east-2')
        bucket_name = "mydatatrendsbucket"
        prefix = "datasets/"
        datalake_dfs = {}

        response = s3_client_east_2.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' not in response:
            logger.warning("No files found in the specified S3 folder: %s", prefix)
            return datalake_dfs

        for obj in response.get('Contents', []):
            key = obj['Key']
            logger.info("Found file in datalake: %s", key)
            if key.endswith('.csv'):
                obj = s3_client_east_2.get_object(Bucket=bucket_name, Key=key)
                datalake_dfs[key] = pd.read_csv(obj['Body'])
            elif key.endswith('.json'):
                obj = s3_client_east_2.get_object(Bucket=bucket_name, Key=key)
                datalake_dfs[key] = pd.read_json(obj['Body'])
            elif key.endswith('.xlsx'):
                obj = s3_client_east_2.get_object(Bucket=bucket_name, Key=key)
                datalake_dfs[key] = pd.read_excel(obj['Body'])
            # Add more file types as needed

        return datalake_dfs
    else:
        return local_load_datalake_dfs()

@app.route('/process', methods=['POST'])
def process_files():
    try:
        data = request.json
        logger.info("Received request data: %s", data)
        user_id = data.get('user_id')
        tier = get_user_tier(user_id)
        allowed, msg, status_code = check_quota(user_id)
        if not allowed:
            return jsonify(msg), status_code
        warning = msg.get("warning") if msg else None
        increment_request(user_id, 0)
        data.get('file_pattern', '*.*')  # Default to processing all files
        target_column = data.get('target_column', None)  # Get the target column from the request, if provided
        category = data.get('category')
        datalake_dfs = load_datalake_dfs()

        # List all files in the user's folder
        if USE_CLOUD:
            s3_client_east_1 = boto3.client('s3', region_name='us-east-1')
            bucket_name = "elasticbeanstalk-us-east-1-971422695246"
            prefix = f"User_Data/{user_id}/"
            logger.info("Listing files in bucket: %s, prefix: %s", bucket_name, prefix)
            response = s3_client_east_1.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            file_keys = [obj['Key'].replace(prefix, '') for obj in response.get('Contents', []) if obj['Key'].replace(prefix, '')]
        else:
            user_dir = Path(os.getenv("LOCAL_DATA_DIR", "local_data")) / "User_Data" / user_id
            file_keys = [p.name for p in user_dir.glob('*') if p.is_file()]

        if tier == "free" and len(file_keys) > 1:
            return jsonify({"error": "Feature not available for free-tier users"}), 403

        if not file_keys:
            logger.warning("No files found for user folder: %s", user_id)
            return jsonify({"status": "success", "results": []})

        results = []
        for file_name in file_keys:
            try:
                logger.info("Calling orchestrate_workflow for file: %s", file_name)
                result = orchestrate_workflow(
                    user_id,
                    file_name,
                    datalake_dfs,
                    target_column,
                    category,
                )
                results.append({"file_name": file_name, "result": result})
            except Exception as e:
                logger.error("Error processing file: %s - %s", file_name, str(e))
                results.append({"file_name": file_name, "error": str(e)})
        response = {"status": "success", "results": results}
        if warning:
            response["warning"] = warning
        return jsonify(response)
    except Exception as e:
        logger.error("Error in process_files: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route('/rerun/<run_id>', methods=['GET'])
def rerun_workflow(run_id):
    """Rerun a previously saved workflow."""
    try:
        meta = load_run_metadata(run_id)
        if not meta:
            return jsonify({"error": "Invalid run_id"}), 404
        user_id = meta.get("user_id")
        file_name = meta.get("file_name")
        allowed, msg, status_code = check_quota(user_id)
        if not allowed:
            return jsonify(msg), status_code
        warning = msg.get("warning") if msg else None
        datalake_dfs = load_datalake_dfs()
        result = orchestrate_workflow(
            user_id=user_id,
            file_name=file_name,
            datalake_dfs=datalake_dfs,
            run_id=run_id,
        )
        if warning:
            result["warning"] = warning
        return jsonify(result)
    except Exception as e:
        logger.error("Error in rerun_workflow: %s", str(e))
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000)
