from Crypto.Cipher import AES
import sys, os

def encrypt_file(input_path, output_path, key):
    """AES-256加密RKNN模型"""
    with open(input_path, 'rb') as f:
        data = f.read()

    # 添加PKCS7填充
    pad_len = AES.block_size - (len(data) % AES.block_size)
    data += bytes([pad_len]) * pad_len

    # 创建加密器
    cipher = AES.new(key.encode(), AES.MODE_ECB)
    ciphertext = cipher.encrypt(data)

    with open(output_path, 'wb') as f:
        f.write(ciphertext)

if __name__ == "__main__":
    # 使用命令行参数
    if len(sys.argv) != 4:
        print(f"用法: {sys.argv[0]} <输入文件> <输出文件> <32字符密钥>")
        sys.exit(1)

    # key = sys.argv[3]
    key = "5XjPuInpUYfTg4LCMrLl/BCFk/seThC/"
    print(f'7月15日, 密钥固定为: {key}')
    if len(key) != 32:
        print("错误: 密钥长度必须为32字符")
        sys.exit(2)

    encrypt_file(sys.argv[1], sys.argv[2], key)
    print(f"加密成功: {sys.argv[1]} -> {sys.argv[2]}")
