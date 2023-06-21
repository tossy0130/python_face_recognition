# !pip install face_recognition==1.2.3
# !pip install dlib==19.18.0

#======================================================================================
#=============================== google colaboratory で実行 ============================
#======================================================================================

import face_recognition
from PIL import Image, ImageDraw
from google.colab import files

image = face_recognition.load_image_file("001.jpg")
face_locations = face_recognition.face_locations(image)


# 出力　結果：[(617, 1256, 1576, 298)] , xy , xy で顔の位置
# face_locations

# 出力 
# image


# ========= 顔の部分を、四角に描画する
def face_detection(image, face_locations):
  # 座標の入れ替え
  face_locations = (face_locations[0][1], face_locations[0][0], face_locations[0][3], face_locations[0][2])
  # オブジェクトを格納
  im = Image.fromarray(image)
  draw = ImageDraw.Draw(im)
  # 描画をする
  draw.rectangle(face_locations, fill=None, outline=(255,0,0), width=5)
  return im

# ===　face_detection 実行
# face_detection(image, face_locations)

# ====================== 類似度検証 start ======================

# ============ ２つの画像の違いを検証 start ============ 
# 画像読み込み 
image1 = face_recognition.load_image_file("001.jpg")
image2 = face_recognition.load_image_file("003.jpg")

# エンコーディング
encoding_01 = face_recognition.face_encodings(image1)[0]
encoding_02 = face_recognition.face_encodings(image2)[0]

# 配列の要素を画像情報に変換
Image.fromarray(image1)
Image.fromarray(image2)

# ========= 画像を比較して、違いを表示    実行結果：[False]
face_recognition.compare_faces([encoding_01], encoding_02, tolerance=0.5)

# ============ ２つの画像の違いを検証 END ============ 

# ============ 画像が同一の画像か検証 start ============ 




# ============ 画像が同一の画像か検証 END ============ 