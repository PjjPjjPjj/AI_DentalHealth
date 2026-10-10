"""Lesion counting + intervention advice (SRS module 5).

Per product decision the numeric score / risk grading was removed:
the app reports lesion counts and intervention advice only.
"""
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 XMU
#
# This file is part of OralAI-DentalHealth.
# Licensed under the GNU Affero General Public License v3.0 or later.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.
from __future__ import annotations

ADVICE = {
    "plaque": [
        "改善刷牙方式：巴氏刷牙法，每日早晚各 2 分钟",
        "配合牙线/牙间隙刷清理邻面菌斑",
        "必要时使用漱口水辅助，定期口腔检查（每 6-12 个月）",
    ],
    "calculus": [
        "菌斑钙化后无法刷牙去除，建议进行专业洁牙（洗牙）",
        "洁牙后抛光可延缓再沉积",
        "每 6-12 个月复查，吸烟/饮茶人群缩短周期",
    ],
    "caries": [
        "发现疑似龋齿区域，请预约口腔科进一步检查（探诊/影像）",
        "浅龋阶段补牙简单成本低，拖延可能发展为牙髓炎",
        "减少高糖饮食频率，餐后清水漱口",
    ],
    "clean": [
        "每日早晚刷牙 + 牙线",
        "每 6-12 个月常规口腔检查",
    ],
}


def assess(result: dict) -> dict:
    teeth = result.get("teeth", [])
    n_p = sum(1 for t in teeth if t["plaque"]["present"])
    n_c = len(result.get("calculus_lesions", []))
    n_r = len(result.get("caries_lesions", []))

    items = []
    if n_p:
        items.append({"lesion": "牙菌斑", "count": n_p, "advice": ADVICE["plaque"]})
    if n_c:
        items.append({"lesion": "牙结石", "count": n_c, "advice": ADVICE["calculus"]})
    if n_r:
        items.append({"lesion": "龋齿", "count": n_r, "advice": ADVICE["caries"]})
    if not items:
        items.append({"lesion": "综合", "count": 0,
                      "advice": ["三类病灶均未检出，继续保持良好口腔卫生习惯"]
                      + ADVICE["clean"]})

    return {
        "plaque": {"count": n_p},
        "calculus": {"count": n_c},
        "caries": {"count": n_r},
        "items": items,
    }
