import os
import shutil
import json
import cv2

imgs_file="./0914-ok"
json_file="./5-17-D (3)缺陷_1_0_.json"

imgs=os.listdir(imgs_file)
for img in imgs:

    shutil.copy(json_file, "./newName.json")

    json_file2= "./newName.json"

    img=os.path.join(imgs_file,img)
    print(img)

    img_name=img.split("\\")[1]
    print(img_name)
    file_name=img_name.split(".")[0]
    print(file_name)

    img = cv2.imread(img)
    img_width,img_height=img.shape[1],img.shape[0]
    with open(json_file2,"r",encoding="utf8") as f:
        json_data = json.load(f)
        json_data["imagePath"]=str(file_name+".bmp")
        json_data["imageHeight"]=img_height
        json_data["imageWidth"]=img_width
    f.close()

    save_path = os.path.join(imgs_file, str(file_name))
    save_path = "{}{}".format(save_path, '.json')

    with open(save_path, 'w') as f:
        json.dump(json_data, f)
    f.close()

    # os.rename(json_file2, os.path.join(imgs_file,file_name+".json"))
