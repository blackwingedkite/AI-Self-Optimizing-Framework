import re

def get_last_full_number(text):
    """
    從字串中抓取最後一個完整的數字。如果沒有找到數字，則返回 0。

    Args:
        text (str): 輸入的字串。

    Returns:
        int: 字串中最後一個完整的數字，如果沒有找到則返回 0。
    """
    # 使用正則表達式找到所有連續的數字序列
    numbers = re.findall(r'\d+', text)

    if numbers:
        # 如果找到數字，返回最後一個數字序列的整數形式
        return int(numbers[-1])
    else:
        # 如果沒有找到數字，返回 0
        return 0
# 測試範例
print(f"g3b33r3333germogesdg4324rnlgnerog -> {get_last_full_number('g3b33r3333germogesdg4324rnlgnerog')}")
print(f"0123rrr -> {get_last_full_number('0123rrr')}")
print(f"55555555 -> {get_last_full_number('55555555')}")
print(f"nhgtrefdbgnhfjutytgrefvb njthygrefdvbfc gnjfhtydresg -> {get_last_full_number('nhgtrefdbgnhfjutytgrefvb njthygrefdvbfc gnjfhtydresg')}")
print(f"vincent922 -> {get_last_full_number('vincent922')}")
print(f"no_number_here -> {get_last_full_number('no_number_here')}")

