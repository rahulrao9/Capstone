import os
import os.path
import logging

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

# Set up logging to write to drive.log file
logging.basicConfig(filename='drive.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SCOPES = ["https://www.googleapis.com/auth/drive"]

creds = None

down_imgs_path = './downloaded_images'

if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            "drive_credentials.json", SCOPES
        )
        creds = flow.run_local_server(port=0)

    with open("token.json", "w") as token:
        token.write(creds.to_json())

try:
    service = build("drive", "v3", credentials=creds)

    # Find the 'farmbot' folder on Google Drive
    response = service.files().list(
        q="name='farmbot' and mimeType='application/vnd.google-apps.folder'",
        spaces='drive'
    ).execute()

    if not response['files']:
        raise FileNotFoundError("The 'farmbot' folder does not exist in Google Drive.")

    farmbot_folder_id = response['files'][0]['id']

    imgs = os.listdir(down_imgs_path)
    if imgs:
        first_file = imgs[0]
    folder_name = first_file.split('_')[0]

    # Check if the folder already exists within the 'farmbot' folder
    response = service.files().list(
        q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{farmbot_folder_id}' in parents",
        spaces='drive'
    ).execute()

    if not response['files']:
        # Create a new folder within the 'farmbot' folder
        file_metadata = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [farmbot_folder_id]
        }

        file = service.files().create(body=file_metadata, fields="id").execute()
        folder_id = file.get('id')
    else:
        folder_id = response["files"][0]["id"]

    # Upload images to the folder
    for file in os.listdir(down_imgs_path):
        # Check if the file already exists in the folder
        response = service.files().list(
            q=f"name='{file}' and '{folder_id}' in parents",
            spaces='drive'
        ).execute()

        if response['files']:
            logging.info(f"File '{file}' already exists in the '{folder_name}' folder. Skipping upload.")
            continue

        # Upload the file if it doesn't already exist
        file_metadata = {
            "name": file,
            "parents": [folder_id]
        }

        media = MediaFileUpload(f"downloaded_images/{file}")
        upload_file = service.files().create(body=file_metadata,
                                             media_body=media,
                                             fields="id").execute()
        
        logging.info(f"Backed up file: {file}")

    # Check if 'farmbot_run.log' already exists in the 'farmbot' folder
    response = service.files().list(
        q=f"name='farmbot_run.log' and '{farmbot_folder_id}' in parents",
        spaces='drive'
    ).execute()

    if response['files']:
        # If the file exists, get its ID and update it
        app_log_file_id = response['files'][0]['id']
        media = MediaFileUpload('farmbot_run.log')
        service.files().update(fileId=app_log_file_id, media_body=media).execute()
        logging.info("farmbot_run.log file updated successfully.")
    else:
        # If the file doesn't exist, create and upload it
        file_metadata = {
            "name": "farmbot_run.log",
            "parents": [farmbot_folder_id]
        }
        media = MediaFileUpload('farmbot_run.log')
        service.files().create(body=file_metadata, media_body=media).execute()
        logging.info("farmbot_run.log file uploaded successfully.")

    # Check if 'drive.log' already exists in the 'farmbot' folder
    response = service.files().list(
        q=f"name='drive.log' and '{farmbot_folder_id}' in parents",
        spaces='drive'
    ).execute()

    if response['files']:
        # If the file exists, get its ID and update it
        app_log_file_id = response['files'][0]['id']
        media = MediaFileUpload('drive.log')
        service.files().update(fileId=app_log_file_id, media_body=media).execute()
        logging.info("drive.log file updated successfully.")
    else:
        # If the file doesn't exist, create and upload it
        file_metadata = {
            "name": "drive.log",
            "parents": [farmbot_folder_id]
        }
        media = MediaFileUpload('drive.log')
        service.files().create(body=file_metadata, media_body=media).execute()
        logging.info("drive.log file uploaded successfully.")

except FileNotFoundError as e:
    logging.error(str(e))

except HttpError as e:
    logging.error("Error: " + str(e))