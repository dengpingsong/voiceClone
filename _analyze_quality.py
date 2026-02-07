"""分析 manifest 数据质量：找出重复文本、注解标记、过短片段等。"""
import json, re
from collections import Counter

rows = []
for line in open("emb_dataset/manifest.jsonl"):
    rows.append(json.loads(line.strip()))

print(f"Total utterances: {len(rows)}\n")

# ── 1. 文本频率统计（找高频重复 = 幻觉） ───────────────
text_counts = Counter(r["text"].strip() for r in rows)

print("=== Top 30 most repeated texts ===")
for txt, cnt in text_counts.most_common(30):
    print(f"  {cnt:5d}x  \"{txt}\"")

# ── 2. 注解/标记类文本 ────────────────────────────────────
annotation_re = re.compile(
    r"^[\(\（].*[\)\）][。\.\s]*$"   # (笑), (スタッフ)はい。, （拍手）etc
)
# 纯括号内容
pure_annot_re = re.compile(
    r"^[\(\（][^\)\）]+[\)\）][。\.\s]*$"  # (笑)。 (小声)
)

annot_count = sum(1 for r in rows if annotation_re.match(r["text"].strip()))
pure_annot = sum(1 for r in rows if pure_annot_re.match(r["text"].strip()))

print(f"\n=== Annotation-like texts ===")
print(f"  Starts with (/（: {annot_count}")
print(f"  Pure annotation (xxx): {pure_annot}")

# ── 3. 高频重复（>=10次相同文本 = 大概率幻觉） ───────────
halluc_threshold = 10
halluc_texts = {txt for txt, cnt in text_counts.items() if cnt >= halluc_threshold}
halluc_count = sum(1 for r in rows if r["text"].strip() in halluc_texts)
print(f"\n=== High-frequency repeats (same text >= {halluc_threshold} times) ===")
print(f"  Unique texts: {len(halluc_texts)}")
print(f"  Total utterances: {halluc_count}")
for txt in sorted(halluc_texts, key=lambda t: -text_counts[t]):
    print(f"    {text_counts[txt]:5d}x  \"{txt}\"")

# ── 4. 过短文本（<= 2字符） ────────────────────────────────
short_text = [(r["text"].strip(), r["duration"]) for r in rows if len(r["text"].strip()) <= 2]
print(f"\n=== Very short text (<= 2 chars): {len(short_text)} ===")
st_counts = Counter(t for t, _ in short_text)
for txt, cnt in st_counts.most_common(20):
    print(f"    {cnt:5d}x  \"{txt}\"")

# ── 5. 综合统计：如果过滤掉这些，剩多少？ ──────────────
# 过滤条件：
#   a) 高频重复 (>=10)
#   b) 纯注解 (笑) (小声) (スタッフ)はい。etc
#   c) 文本 <= 1 字符
#   d) 时长 < 1s 或 > 30s
filtered = []
reasons = Counter()
for r in rows:
    txt = r["text"].strip()
    dur = float(r["duration"])
    
    if txt in halluc_texts:
        reasons["hallucination"] += 1
        continue
    if pure_annot_re.match(txt):
        reasons["pure_annotation"] += 1
        continue
    if len(txt) <= 1:
        reasons["too_short_text"] += 1
        continue
    if dur < 1.0:
        reasons["too_short_dur"] += 1
        continue
    if dur > 30.0:
        reasons["too_long_dur"] += 1
        continue
    filtered.append(r)

print(f"\n=== Filter summary ===")
for reason, cnt in reasons.most_common():
    print(f"  {reason}: -{cnt}")
print(f"  ─────────────────")
print(f"  Before: {len(rows)}")
print(f"  After:  {len(filtered)}  ({len(filtered)/len(rows)*100:.1f}%)")
