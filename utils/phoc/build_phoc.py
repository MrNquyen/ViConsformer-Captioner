import numpy as np
import os
# print(os.getcwd())
# from utils.phoc.src.cphoc import build_phoc as _build_phoc_raw
from utils.phoc.src.cphoc_vn import build_phoc as _build_phoc_raw
from icecream import ic

_alphabet = {
    'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'f', 'g', 'h', 'i', 'j', 'k',
    'l', 'm', 'n', 'o', 'ô', 'ơ', 'p', 'q', 'r', 's', 't', 'u', 'ư', 'v', 'w', 'x', 'y', 'z'
}
acute = {'á',  'ắ',  'ấ',  'é',  'ế',  'í',  'ó',  'ố',  'ớ',  'ú',  'ứ',  'ý'}  # ´
grave = {'à',  'ằ',  'ầ',  'è',  'ề',  'ì',  'ò',  'ồ',  'ờ',  'ù',  'ừ',  'ỳ'}  # `
hook = {'ả',  'ẳ',  'ẩ',  'ẻ',  'ể',  'ỉ',  'ỏ',  'ổ',  'ở',  'ủ',  'ử',  'ỷ'} # ~
tilde = {'ã',  'ẵ',  'ẫ',  'ẽ',  'ễ',  'ĩ',  'õ',  'ỗ',  'ỡ',  'ũ',  'ữ',  'ỹ'}  # ~
dot = {'ạ',  'ặ',  'ậ',  'ẹ',  'ệ',  'ị',  'ọ',  'ộ',  'ợ',  'ụ',  'ự',  'ỵ'}   # ˀ
number = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
_alphabet = _alphabet | acute | grave | hook | tilde | dot

def build_phoc(token):
    token = token.lower().strip()
    token = ''.join([c for c in token if c in _alphabet])
    phoc = None
    phoc = _build_phoc_raw(token)
    # except:
    #     ic(token)
    phoc = np.array(phoc, dtype=np.float32)
    return phoc