# -*- coding: utf-8 -*-

"""
渲染阶段简体转繁体工具。

注意：
1. 训练和推理阶段仍然使用简体；
2. result.json 仍然保存简体；
3. 只有写入图片模板前才转繁体。
"""

try:
    from opencc import OpenCC
    _OPENCC = OpenCC("s2t")
except Exception:
    _OPENCC = None


S2T_FALLBACK_MAP = str.maketrans({
    "专": "專",
    "为": "為",
    "买": "買",
    "亲": "親",
    "启": "啟",
    "头": "頭",
    "银": "銀",
    "闻": "聞",
    "顺": "順",
    "药": "藥",
    "医": "醫",
    "东": "東",
    "发": "發",
    "长": "長",
    "国": "國",
    "来": "來",
    "体": "體",
    "岁": "歲",
    "写": "寫",
    "万": "萬",
    "这": "這",
    "里": "裡",
    "后": "後",
    "现": "現",
    "乡": "鄉",
    "广": "廣",
    "阳": "陽",
    "开": "開",
    "关": "關",
    "务": "務",
    "钱": "錢",
    "气": "氣",
    "丰": "豐",
    "无": "無",
    "还": "還",
    "读": "讀",
    "书": "書",
    "学": "學",
    "劝": "勸",
})


def to_traditional_for_render(text: str) -> str:
    text = str(text or "")
    if _OPENCC is not None:
        return _OPENCC.convert(text)
    return text.translate(S2T_FALLBACK_MAP)


def clean_render_text(text: str) -> str:
    text = to_traditional_for_render(text)
    return text.replace(" ", "").replace("\n", "").replace("\t", "")