# Aerial Drone Imageを用いたセグメンテーション
このレポジトリではオープンデータ「[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)」を用いたセマンティック・セグメンテーションのモデルを公開している。
かつ、そのモデルをベースにファインチューニングを行うソースコードや、モデルの推論を高速化するテクニックなども紹介している。
## オープンデータのクラス
オープンデータ「Semantic Drone Dataset」は24Classから構成されるデータセットでアノテーションとクラスの対応は以下のとおりである。
|id|クラス名|RGB値|
|-|-|-|
|0|ラベルなし|0, 0, 0|
|1|道路|128, 64, 128|
|2|土壌|130, 76, 0|
|3|草原|0, 102, 0|
|4|砂利|112, 103, 87|
|5|水|28, 42, 168|
|6|岩|48, 41, 30|
|7|プール|0, 50, 89|
|8|草木|107, 142, 35|
|9|屋根|70, 70, 70|
|10|壁|102, 102, 156|
|11|窓|254, 228, 12|
|12|ドア|254, 148, 12|
|13|フェンス|190, 153, 153|
|14|フェンス支柱|153, 153, 153|
|15|人|255, 22, 96|
|16|犬|102, 51, 0|
|17|車|9, 143, 150|
|18|自転車|119, 11, 32|
|19|木|51, 51, 0|
|20|枯れ木|190, 250, 190|
|21|arマーカー|112, 150, 146|
|22|障害|2, 135, 115|
|23|衝突|255, 0, 0|

## Getting Started / スタートガイド
### Prerequisites / 必要条件
#### オンプレでやる場合
- Intel CPU（Core or Xeon）を搭載したマシン
    - Core: 第10世代以上
    - Xeon: 第2世代Xeonスケーラブル・プロセッサー以上
- OS: Windows 10(WSL2) / Ubuntu 18.04以降
- Docker（※以下にインストール手順記載）
#### Google Colaboratoryでやる場合
- インターネット環境にアクセスできるWin10/Mac/Linux PC
- Googleアカウント
### Installing / インストール
#### ホストOSのポート開放（リモートアクセスする場合のみ）
このハンズオンではJupyter Labを使用します。特にサーバーにリモートアクセスしながら実施する場合は各環境ごとの手順に則り、ホストOSのポート「8080」番を開放ください。
#### Dockerインストール
```Bash
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
sudo apt update
apt-cache policy docker-ce
sudo apt install -y docker-ce
sudo usermod -aG docker ${USER}
su - ${USER}
id -nG
```

#### Dockerイメージのダウンロード
```Bash
docker pull continuumio/anaconda3
```
#### Dockerコンテナの起動
コンテナはRootで起動します。また、8888番ポートをホストOSとコンテナとでバインドしておきます。
```Bash
sudo docker run -it -u 0 --privileged -p 8888:8888 continuumio/anaconda3 /bin/bash
```
以降はコンテナ上での作業になります。
#### 追加モジュールのインストール
```Bash
apt-get update
apt-get install -y wget unzip git sudo vim numactl
apt-get install -y libgl1-mesa-dev
conda create -n ov python=3.8 -y
conda activate ov
pip install --upgrade pip
pip install torch torchvision torchaudio pandas scikit-learn statistics pillow opencv-python albumentations tqdm matplotlib typing-extensions==4.4.0 jupyterlab segmentation-models-pytorch torchsummary
pip install ipywidgets widgetsnbextension
```
#### 本レポジトリをClone
```Bash
cd ~
git clone https://github.com/hiouchiy/Drone_Segmentation.git
```
#### Jupyter Labの起動
```Bash
jupyter lab --allow-root --ip=0.0.0.0 --no-browser --port=8888
```
#### WebブラウザからJupyter Labにアクセス
前のコマンド実行すると以下のようなログが出力されまして、最後にローカルホスト（127.0.0.1）のトークン付きURLが表示されるはずです。こちらをWebブラウザにペーストしてアクセスください。リモートアクセスされている場合はIPアドレスをサーバーのホストOSのIPアドレスに変更してください。
```
root@f79f54d47c1b:~# jupyter lab --allow-root --ip=0.0.0.0 --no-browser
[I 09:13:08.932 LabApp] JupyterLab extension loaded from /usr/local/lib/python3.6/dist-packages/jupyterlab
[I 09:13:08.933 LabApp] JupyterLab application directory is /usr/local/share/jupyter/lab
[I 09:13:08.935 LabApp] Serving notebooks from local directory: /root
[I 09:13:08.935 LabApp] Jupyter Notebook 6.1.4 is running at:
[I 09:13:08.935 LabApp] http://f79f54d47c1b:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
[I 09:13:08.935 LabApp]  or http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
[I 09:13:08.935 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 09:13:08.941 LabApp] 
    
    To access the notebook, open this file in a browser:
        file:///root/.local/share/jupyter/runtime/nbserver-33-open.html
    Or copy and paste one of these URLs:
        http://f79f54d47c1b:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
     or http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3
```
↑こちらの例の場合は、最後の "http://127.0.0.1:8888/?token=2d6863a5b833a3dcb1a57e3252e641311ea7bc8e65ad9ca3" です。
#### Notebookの起動
Jupyter Lab上で下記の中から好みのノートブックを開き、後はノートブックの内容に従って進めてください。

- TrainModel.ipynb
[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)」を用いた学習
- Test_PreTrainModel.ipynb
[事前学習済みのモデル](https://drive.google.com/file/d/14PtYuFZc-5sB2n9lLUDku8bgyEKSLZG5/view?usp=share_link)を用いて対象データに対してのセグメンテーションを実施する
- Fine-Tuning.ipynb
[事前学習済みのモデル](https://drive.google.com/file/d/14PtYuFZc-5sB2n9lLUDku8bgyEKSLZG5/view?usp=share_link)をベースとして、カスタム画像データを用いてFine-Tuning(転移学習)を行う
- Test_CustumModel.ipynb
[Fine-Tuningを行ったモデル](https://drive.google.com/file/d/1JXPHg4brau1T93z79VNr4VqLeCEx2CcW/view?usp=share_link)を用いて対象画像データに対してのセグメンテーションを実施する
- Model_Optimization_and_Quantization.ipynb
モデルの推論をCPU上で高速化するためのテクニック集
##### 実行方法
ノートブックのパス設定を任意のパスに書き換えてノートブック上部から実行
- IMAGE_PATH, MASK_PATH セグメンテーション対象データの入力画像とアノテーション画像のファイルパス
- MODEL_PATH　事前学習済みのモデルを保存しているファイルパス
- SAVE_PATH　セグメンテーションの結果 or Fine-Tuningを行ったモデルを保存するフォルダパス

## License / ライセンス
このプロジェクトは MITライセンスです。

## Acknowledgments / 謝辞
本ソースコードは[こちら](https://github.com/G21TKA01/Drone_Segmentation)をベースにアレンジを加えたものです。作者である[G21TKA01](https://github.com/G21TKA01)には事前に了承を取ったうえで使用しております。改めまして、作者には素晴らしいアプリケーションを提供いただいたことに感謝いたします。