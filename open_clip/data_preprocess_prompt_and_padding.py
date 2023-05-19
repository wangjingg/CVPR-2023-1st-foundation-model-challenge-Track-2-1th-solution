import os
import cv2
import pandas as pd

color = ['grey', 'orange', 'red', 'Golden', 'brown', 'yellow', 'pink', 'blue', 'green', 'purple', 'white', 'black']
brand = ['XIALI', 'Chevrolet', 'Balong', 'Honda', 'Suzuki', 'Porsche', 'SKODA',
         'BAOJUN', 'Chery', 'Haima', 'Hongyan', 'Landrover', 'Mazda', 'Luxgen',
         'KINGLONG', 'Yutong', 'BYD', 'MORRIS-GARAGE', 'Mitsubishi', 'Cadillac',
         'Jeep', 'China-Moto', 'FOTON', 'Shacman', 'Bentley',
          'Volkswagen', 'Karma', 'Soueast', 'SGMW', 'ROEWE', 'Shuanghuan',
          'Toyota', 'Nissan', 'Dongfeng', 'LEOPAARD', 'FAW', 'Lexus', 'Ford',
          'Jinbei', 'Style', 'GreatWall', 'ZXAUTO', 'FORLAND', 'JAC', 'Chana',
          'Hyundai', 'Audi', 'Dragon', 'PEUGEOT', 'Buick', 'Geely', 'Infiniti',
          'BESTUNE', 'OPEL', 'Iveco', 'Benz', 'Subaru', 'Citroen', 'Kia', 'Isuzu',
          'HAFEI', 'JMC', 'BMW', 'Volvo']

car_type = ['SUV', 'Bus', 'Microbus', 'Sedan', 'Minivan', 'Truck' ]


def prompt_argumentation(train_file, trai_argument):
    slice = 42704  ## 数据集相关，默认适配A榜数据，数据集变化需要修改！！

    for index, line in enumerate(train_list.readlines()):
        trai_argument.writelines(line)
        origin_line = line
        line = line.strip("\n").split("$")
        image_name = line[0]
        attribute = "empty" #line[1]
        text = line[2]

        # print(image_name, attribute, text)
        color_prompt = ""
        type_prompt = ""
        brand_prompt = ""

        if index < slice: # vehicle
            # print(image_name, attribute, text)
            for attri in attribute[:-1].split(" "):
                # print(attri)
                if attri in color:
                    color_prompt = "The color of the vehicle is " + attri + "."
                    print("color prompt:", color_prompt)
                elif attri in brand:
                    brand_prompt = "The brand of the vehicle is " + attri + "."
                    print("brand prompt:", brand_prompt)
                 #     type_prompt = "The type of the vehicle is " + attri + "."
                #     print("type prompt:", type_prompt)
            v_prompt = color_prompt + brand_prompt
            if len(v_prompt) > 3:
                trai_argument.writelines(image_name + "$" + attribute + "$" + v_prompt + "\n")
                trai_argument.writelines(image_name + "$"   + attribute + "$" + text+ v_prompt + "\n")  

        else: # pedestian
            for prompt in text[:-1].split("."):
                if prompt[0] == " ": prompt = prompt[1:]
                trai_argument.writelines(image_name + "$" + attribute + "$" + prompt + "." + "\n")
# resize and padding
def pad_image(image):
    height, width = image.shape[:2]
    ratio = 224 / max(height, width)
    new_height = int(height * ratio)
    new_width = int(width * ratio)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    top_pad = (224 - new_height) // 2
    bottom_pad = 224 - new_height - top_pad
    left_pad = (224 - new_width) // 2
    right_pad = 224 - new_width - left_pad
    
    padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
    
    return padded_image


def process_padding(path: str, sub_dir: str):
    if not os.path.exists(os.path.join(path, sub_dir + "_padding")):
        os.mkdir(os.path.join(path, sub_dir + "_padding"))

    for img in os.listdir(os.path.join(path, sub_dir)):
        image = cv2.imread(os.path.join(path+"/"+sub_dir, img))
        try:
            image_pad = pad_image(image)
            cv2.imwrite(os.path.join(path+"/"+sub_dir + "_padding", img), image_pad)
        except Exception as e:
            print("warning:", e, img)
            import shutil
            shutil.copy(os.path.join(path+"/"+sub_dir, img), os.path.join(path, sub_dir + "_padding"))

if __name__ == "__main__":
    ## dataset path
    data_set_path = "/home/aistudio/data/data218656/"
    
    ## step1 padding
    process_padding(os.path.join(data_set_path, "train"), "train_images")
    process_padding(os.path.join(data_set_path, "val"), "val_images")
    process_padding(os.path.join(data_set_path, "test"), "test_images")

    ## step2 training prompt argumentation

    train_list = open(data_set_path + "/train/train_label.txt", "r")
    trai_argument = open(data_set_path + "/train/train_label_prompt_argumentation.txt", "w")

    prompt_argumentation(train_list, trai_argument)

    ## step3 convert to cvs format file for open clip training
    # 读取train txt文件
    with open(data_set_path + "/train/train_label_prompt_argumentation.txt", 'r') as f:
        data = f.readlines()

        csv_data = []
        for index, x in enumerate(data):
            xsplit = x.strip().split("$")
            # print("xsplit", index, xsplit, x, data[-1])
            csv_data.append([os.path.join(data_set_path, "train", "train_images_padding", xsplit[0]), xsplit[1], xsplit[2]])

    df = pd.DataFrame(csv_data, columns=['filepath', 'class', 'title'])
    df.to_csv(data_set_path + "/train/train_label_prompt_argumentation.csv", index=False)

    # 读取val txt文件
    with open(data_set_path + "/val/val_label.txt", 'r') as f:
        data = f.readlines()

        csv_data = []
        for x in data:
            xsplit = x.strip().split("$")
            csv_data.append([os.path.join(data_set_path, "val", "val_images_padding",xsplit[0]), xsplit[1], xsplit[2]])

    df = pd.DataFrame(csv_data, columns=['filepath', 'class', 'title'])
    df.to_csv(data_set_path + "/val/val_label.csv", index=False)

