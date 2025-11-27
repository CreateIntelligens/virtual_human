import re

from .num import verbalize_digit

# 台灣手機號碼正則表達式
# 支援多種格式：
# 0966, 0967 (遠傳)
# 0910, 0912 (中華)
# 0926, 0927 (台哥大)
# 0935, 0938 (亞太)
# 0905, 0906 (台灣之星)
RE_MOBILE_PHONE = re.compile(
    r"(?<!\d)(((\+?886 ?)?0?9[0-9]{8})|((\+?886 ?)?9[0-9]{8})|"
    r"((\+?886 ?)?09[0-9]{8}))(?!\d)")

# 台灣市話正則表達式
# 支援多種格式：
# 02-XXXX-XXXX
# 02XXXXXXXX
# (02)XXXXXXXX
# 區碼：02(台北)、03(桃園)、037(苗栗)、04(台中)、05(嘉義)、06(台南)、07(高雄)、08(屏東)、039(宜蘭)
RE_TELEPHONE = re.compile(
    r"(?<!\d)((0[2-8]\d{1,2}[-]?\d{4}[-]?\d{4})|"
    r"(\(0[2-8]\d{1,2}\)\d{8}))(?!\d)")

# 免付費或服務專線
RE_SERVICE_NUMBER = re.compile(r"(0800)(-)?\d{3}(-)?\d{4}")


def phone2str(phone_string: str, mobile=True) -> str:
    """
    將電話號碼轉換為語音可讀的文字表示
    
    Args:
        phone_string (str): 電話號碼字串
        mobile (bool): 是否為手機號碼，預設為True
    
    Returns:
        str: 語音文字表示
    """
    # 移除所有非數字字符
    cleaned_phone = re.sub(r'[^\d]', '', phone_string)
    
    # 如果是手機號碼，去除開頭的0或+886
    if mobile:
        cleaned_phone = re.sub(r'^(0|\+886)', '', cleaned_phone)
    
    # 將號碼分成適當的部分
    if mobile:
        parts = [cleaned_phone[:4], cleaned_phone[4:]]
    else:
        # 市話：區碼 + 電話號碼
        parts = [cleaned_phone[:2], cleaned_phone[2:6], cleaned_phone[6:]]
    
    result = '，'.join(
        [verbalize_digit(part, alt_one=True) for part in parts])
    return result


def replace_phone(match) -> str:
    """
    替換市話號碼
    
    Args:
        match (re.Match): 匹配的市話號碼
    Returns:
        str: 轉換後的語音文字
    """
    return phone2str(match.group(0), mobile=False)


def replace_mobile(match) -> str:
    """
    替換手機號碼
    
    Args:
        match (re.Match): 匹配的手機號碼
    Returns:
        str: 轉換後的語音文字
    """
    return phone2str(match.group(0))