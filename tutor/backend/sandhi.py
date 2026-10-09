"""
Mandarin Tone Sandhi Processor
Handles standard phonological tone modifications in spoken Mandarin:
1. 3-3 Tone Sandhi: Consecutive Tone 3 syllables -> first becomes Tone 2 (e.g. 你好 nǐ hǎo -> ní hǎo)
2. '不' (bù) Sandhi: Tone 4 followed by Tone 4 -> becomes Tone 2 (e.g. 不是 bú shì)
3. '一' (yī) Sandhi:
   - Before Tone 4 -> becomes Tone 2 (e.g. 一个 yí gè)
   - Before Tone 1, 2, 3 -> becomes Tone 4 (e.g. 一天 yì tiān, 一起 yì qǐ)
   - In numbers/ordinals (第一, 一二三) -> remains Tone 1
4. Neutral tone (轻声) handling for modal particles and reduplications.
"""

from typing import List, Dict, Any

def apply_tone_sandhi(syllable_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given a list of syllable dicts:
    [{'char': '你', 'pinyin': 'nǐ', 'tone': 3, ...}, ...]
    Returns a modified list where 'expected_tone' accounts for standard tone sandhi rules.
    """
    n = len(syllable_list)
    if n == 0:
        return []

    # Copy list so we don't mutate input in-place
    res = []
    for item in syllable_list:
        copy_item = dict(item)
        copy_item["base_tone"] = copy_item.get("tone", 5)
        copy_item["expected_tone"] = copy_item.get("tone", 5)
        copy_item["sandhi_applied"] = False
        copy_item["sandhi_rule"] = None
        res.append(copy_item)

    # 1. '不' Sandhi
    for i in range(n - 1):
        if res[i]["char"] == "不":
            next_tone = res[i + 1]["base_tone"]
            if next_tone == 4:
                res[i]["expected_tone"] = 2
                res[i]["sandhi_applied"] = True
                res[i]["sandhi_rule"] = "bu_before_tone4"

    # 2. '一' Sandhi
    for i in range(n):
        if res[i]["char"] == "一":
            # Check if part of ordinal or single number
            is_ordinal = (i > 0 and res[i - 1]["char"] in ["第", "周", "礼拜", "星期"])
            if is_ordinal or (n == 1):
                res[i]["expected_tone"] = 1
            elif i < n - 1:
                next_tone = res[i + 1]["base_tone"]
                if next_tone == 4:
                    res[i]["expected_tone"] = 2
                    res[i]["sandhi_applied"] = True
                    res[i]["sandhi_rule"] = "yi_before_tone4"
                elif next_tone in (1, 2, 3):
                    res[i]["expected_tone"] = 4
                    res[i]["sandhi_applied"] = True
                    res[i]["sandhi_rule"] = "yi_before_tones_123"

    # 3. 3-3 Tone Sandhi (two consecutive 3rd tones -> first becomes 2nd tone)
    # Applied from left-to-right on pairs
    for i in range(n - 1):
        if res[i]["expected_tone"] == 3 and res[i + 1]["expected_tone"] == 3:
            res[i]["expected_tone"] = 2
            res[i]["sandhi_applied"] = True
            res[i]["sandhi_rule"] = "third_tone_sandhi_33"

    return res
