import hashlib
import os
import shutil
import argparse


def config():
    parser = argparse.ArgumentParser(description="")

    # 如果指定image_path，则统计image_path中图片的对应的瑕疵类别
    parser.add_argument('--in_path', type=str, default="E:/DL_demos/demo8_similar/datasets_AEresult/part7/image",
                        help="Path that need to clean up the duplicate image")
    parser.add_argument('--out_path', type=str, default="E:/DL_demos/demo8_similar/datasets_AEresult/part7/dup",
                        help="Path that save duplicate image")

    args = parser.parse_args()

    return args


def get_md5(file):
    file = open(file, 'rb')
    md5 = hashlib.md5(file.read())
    file.close()
    md5_values = md5.hexdigest()
    return md5_values


if __name__ == '__main__':
    cfg = config()

    in_path = cfg.in_path
    out_path = cfg.out_path
    # os.chdir(file_path)
    file_list = os.listdir(in_path)
    md5_list = []
    for file in file_list:
        filename_in = os.path.join(in_path, file)
        # print(filename_in)
        md5 = get_md5(filename_in)
        if md5 not in md5_list:
            md5_list.append(md5)
        else:
            # os.remove(file)
            filename_out = os.path.join(out_path, file)
            print(filename_out)
            shutil.move(filename_in, filename_out)
