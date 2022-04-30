import os


#   {
#     painter: 'Albrecht Durer',
#     img_src: require('@/assets/train/Albrecht_Durer/Albrecht_Durer_1.jpg'),
#     srcList: [
#       require('@/assets/train/Albrecht_Durer/Albrecht_Durer_1.jpg'),
#       require('@/assets/train/Albrecht_Durer/Albrecht_Durer_1.jpg')
#     ]
#   }
def get_format(name_str, file_path):
    res = "   {\n" + \
          "    painter: '{}',\n".format(name_str) + \
          "    img_src: require('@/assets/train/{}'),\n".format(file_path) + \
          "     srcList: [\n" + \
          "       require('@/assets/train/{}'),\n".format(file_path) + \
          "       require('@/assets/cam_train_grad_cam/{}')\n".format(file_path) + \
          "     ]\n   },\n"
    return res


os.chdir('/Users/apple/Desktop/FRproject/冯如资料/CNNapp/train')
with open("result.txt", "w") as f:
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name.find(".") == -1 and name == "Diego_Velazquez":
                for root2, dirs2, files2 in os.walk("./" + name, topdown=False):
                    for name2 in files2:
                        print(get_format(name, name + "/" + name2))
                        f.write(get_format(name, name + "/" + name2))
        # for name in files:
            # print(os.path.join(root, name))
            # if name.find(" 2.jpg") != -1:
            #     print(os.path.join(root, name))
            #     os.system("rm '" + os.path.join(root, name) + "'")
