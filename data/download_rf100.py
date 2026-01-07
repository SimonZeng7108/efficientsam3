#download instructions for RF100-VL and RF20-VL datasets: https://github.com/roboflow/rf100-vl/?tab=readme-ov-file
from rf100vl import download_rf100vl, download_rf100vl_fsod, download_rf20vl_fsod, download_rf20vl_full

# download_rf100vl(path="./rf100-vl/")
# download_rf100vl_fsod(path="./rf100-vl-fsod/")
# download_rf20vl_fsod(path="./rf20-vl-fsod/")
download_rf20vl_full(path="./rf20-vl/")