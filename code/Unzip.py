import zipfile
import os

def unzip_files(zip_folder, output_folder):

    # 指定フォルダ内のすべてのZIPファイルを取得
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith('.zip')]

    for zip_file in zip_files:
        zip_path = os.path.join(zip_folder, zip_file)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_folder)
            print(f'解凍完了: {zip_file} -> {output_folder}')

zip_folder = '/home/slab/ishiiayuka/M2/zipdata'  # ZIPファイルがあるフォルダのパスを指定
output_folder = '/home/slab/ishiiayuka/M2/data'  # 解凍先のフォルダを指定
unzip_files(zip_folder, output_folder)
