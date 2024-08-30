# cloths_segmentation-onnx-sample
衣服セグメンテーションモデルの[ternaus/cloths_segmentation](https://github.com/ternaus/cloths_segmentation)のONNX変換/推論のサンプルです。<br>
変換自体を試したい方はColaboratoryなどで[Convert2ONNX.ipynb](Convert2ONNX.ipynb)を使用ください。<br>

![image](https://github.com/user-attachments/assets/284b8f86-b345-489b-bd59-22ecce9dcf19)

# Requirement
* OpenCV 4.5.3.56 or later
* onnxruntime-gpu 1.9.0 or later <br>※onnxruntimeでも動作しますが、推論時間がかかるのでGPUをお勧めします

# Convert ONNX
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kazuhito00/cloths_segmentation-onnx-sample/blob/main/Convert2ONNX.ipynb)<br>
モデル変換を試す場合は、Colaboratoryでノートブックを開き、上から順に実行してください。<br>

# Demo
デモの実行方法は以下です。
```bash
python sample_onnx.py --image=sample.jpg
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --image<br>
画像ファイルの指定 ※指定時はカメラデバイスや動画より優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
* --model<br>
ロードするモデルの格納パス<br>
デフォルト：model/cloths_segmentation.onnx
* --input_resize_rate<br>
モデル入力時の画像リサイズ割合（処理時間がかかる場合、小さいサイズを指定し、精度を下げて推論速度を向上させる）<br>
デフォルト：1.0

# Reference
* [ternaus/cloths_segmentation](https://github.com/ternaus/cloths_segmentation)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
cloths_segmentation-onnx-sample is under [MIT License](LICENSE).

# License(Movie, Image)
サンプルの画像は[ぱくたそ](https://www.pakutaso.com/)様の[慌てて本社に戻る部長](https://www.pakutaso.com/20240303088post-50910.html)を使用しています。
