import sys
print("Python版本:", sys.version)
print("Python路径:", sys.path)

try:
    import cv2
    print("OpenCV版本:", cv2.__version__)
except ImportError as e:
    print("无法导入cv2模块:", e)
    print("\n尝试检查包是否安装:")
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
        print("\npip已安装的包:")
        print(result.stdout)
    except Exception as pip_error:
        print("无法检查pip包:", pip_error)