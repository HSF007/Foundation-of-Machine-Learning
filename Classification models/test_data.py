import os
import pandas as pd
from data_preprocessing import preprocess_text
from data_preprocessing import vectorizer


def load_emails(folder_path):
    email_data = []
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return
    
    # Read each .txt file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                email_content = file.read()
                email_content = email_content.split('\n')
                email_content = ' '.join(email_content)
                email_data.append({"email_content": email_content})
    return email_data


directory_name = os.path.dirname(__file__)
folder_path = os.path.join(directory_name, 'test')

# Create DataFrame
emails = load_emails(folder_path)
email_df = pd.DataFrame(emails, columns=["email_content"])

# Preprocessing Data
email_df['clean_data'] = email_df['email_content'].apply(lambda x: preprocess_text(x) if isinstance(x, str) else "")

final_input_test = email_df['clean_data']

final_input_test_vect = vectorizer.transform(final_input_test).toarray()
