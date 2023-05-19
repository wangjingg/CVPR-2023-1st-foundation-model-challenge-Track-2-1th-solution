import os
import json

prompt_list = []
    
txt_file = open('./datasets/train/train_label.txt', 'r')
save_file = open('./datasets/train/train_label_prompt.txt', 'w')
lines = txt_file.readlines()
car_info = {
        "color": {'grey', 'orange', 'red', 'brown', 'yellow', 'pink', 'blue', 'green', 'purple', 'white', 'black'},
        "brand": {'XIALI', 'Chevrolet', 'Balong', 'Honda', 'Suzuki', 'Porsche', 'SKODA', 'BAOJUN', 'Chery', 'Haima', 'Hongyan', 
                  'Landrover', 'Mazda', 'Luxgen', 'KINGLONG', 'Yutong', 'BYD', 'MORRIS-GARAGE', 'Mitsubishi',
                    'Cadillac', 'Jeep', 'China-Moto', 'FOTON', 'Shacman', 'Bentley', 'Volkswagen', 'Karma',
                    'Soueast', 'SGMW', 'ROEWE', 'Shuanghuan', 'Toyota', 'Nissan', 'Dongfeng', 'LEOPAARD', 
                    'FAW', 'Lexus', 'Ford', 'Jinbei', 'Style', 'GreatWall', 'ZXAUTO', 'FORLAND', 'JAC', 
                    'Chana', 'Hyundai', 'Audi', 'Dragon', 'PEUGEOT', 'Buick', 'Geely', 'Infiniti', 'BESTUNE', 
                    'OPEL', 'Iveco', 'Benz', 'Subaru', 'Citroen', 'Kia', 'Isuzu', 'HAFEI', 'JMC', 'BMW', 'Volvo'},
        "type": {'SUV', 'Bus', 'Microbus', 'Sedan', 'Minivan', 'Truck'}
        }

car_dict = {}
for line in lines:
    img, label, text = line.strip().split('$')
    info = label.strip('.').split()
    # color prompt:
    color_prompt = ''
    brand_prompt = ''
    type_prompt = ''

    for i in info:
        if i in car_info['color']:
            color_prompt = 'The color of the vehicle is ' + i + '.'
        if i in car_info['brand']:
            brand_prompt = 'The brand of the vehicle is ' + i + '.'
        if i in car_info['type']:
            type_prompt = "The vehicle's model is " + i + '.'

    prompt = "This is a vehicle." + brand_prompt + type_prompt + color_prompt
    
    save_line = line.strip() + prompt
    save_file.write(save_line + '\n')










