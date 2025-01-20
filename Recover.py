import os
from tqdm import tqdm
path = '/data/dehghani/recovered'
for file in tqdm(os.listdir(path)):
    file_path = os.path.join(path, file)
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        if 'ultralytics.utils.ops' in file_content and 'ultralytics.engine.results' in file_content and 'ultralytics' in file_content:
            print(file_path)
            print("File Content:\n", file_content)
            
            print("**********************")
            answer = input('Do you know this file?')
            if answer.lower() == 'yes' or answer.lower() == 'y':
                os.rename(file_path, os.path.join('/data/dehghani/new', input('suggest a name for this file: ')))
        
        

    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")