import os
import shutil


def keep_images_with_pattern(root_dir, pattern="_0_"):
    """
    遍历 root_dir 下的所有子文件夹，只保留文件名中包含 pattern 的图片，
    删除其他所有图片文件。
    """
    # 支持的图片扩展名
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # 检查是否是图片文件
            ext = os.path.splitext(filename)[1].lower()
            if ext not in image_extensions:
                continue  # 不是图片文件，跳过

            file_path = os.path.join(dirpath, filename)

            # 如果文件名不包含 pattern，则删除
            if pattern not in filename:
                try:
                    os.remove(file_path)
                    print(f"已删除: {file_path}")
                except Exception as e:
                    print(f"删除失败 {file_path}: {e}")
            else:
                print(f"保留: {file_path}")


if __name__ == "__main__":
    # 当前脚本所在目录下的 PARSE 文件夹
    parse_folder = os.path.join(os.getcwd(), "angiographs")

    if not os.path.exists(parse_folder):
        print(f"错误: 文件夹 '{parse_folder}' 不存在")
    else:
        keep_images_with_pattern(parse_folder, "_0_")
        print("处理完成！")