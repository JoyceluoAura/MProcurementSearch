import os
import re
import json
from typing import List, Optional, Dict, Any

import pandas as pd
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
import tldextract

from tavily import TavilyClient
from firecrawl import FirecrawlApp
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ====== 配置区 ======
SHEET_URL = "https://docs.google.com/spreadsheets/d/1JzGWhAh0rDNrqG3KiwIIEMdRzTF4pEM80z5wRkQdS30/edit?usp=sharing"

FALLBACK_EQUIPMENT_LIST = [
    {"id": 1, "name_cn": "离子抛光仪", "model_hint": "Gantan PIPS 2 或 Cross Section Polisher™", "budget_wan_rmb": 80},
    {"id": 2, "name_cn": "超声波数字清洗机", "model_hint": "Hielscher UP400St / UIP500hdT", "budget_wan_rmb": 1},
    {"id": 3, "name_cn": "镀膜仪", "model_hint": "Leica EM ACE600 或同类型", "budget_wan_rmb": 50},
]

MAX_URLS_PER_ITEM = 12
MIN_CONFIDENCE = 0.55

ALLOWED_DOMAINS = {
    "ccgp.gov.cn",
    "china-bidding.com",
    "bidcenter.com.cn",
    "zhaobiao.cn",
    "okcis.com",
    "thomassci.com",
    "fishersci.com",
    "vwr.com",
    "coleparmer.com",
}

MANUFACTURER_KEYWORDS = [
    "leica", "hielscher", "gatan", "thermofisher", "bruker", "zeiss",
    "jeol", "rigaku", "netzsch", "malvern", "instron", "agilent",
    "labconco", "retsch", "keyence", "perkinelmer", "micromeritics",
    "buehler", "struers",
]

BLOCKLIST_DOMAINS = {
    "zhihu.com", "weibo.com", "xiaohongshu.com", "douban.com",
    "reddit.com", "58.com", "2.taobao.com", "xianyu.com",
}

BLOCKLIST_PATH_KEYWORDS = [
    "/forum", "/bbs", "/post", "/question", "/answers", "/tieba"
]

STATIC_FX = {
    "USD": 7.2,
    "EUR": 7.8,
    "GBP": 9.0,
    "JPY": 0.05,
    "CNY": 1.0,
    "RMB": 1.0,
}

SOURCE_PRIORITY = {
    "award": 1,
    "tender": 2,
    "list": 3,
    "dealer_quote": 4,
    "ecommerce": 5,
    "unknown": 6,
}

assert os.environ.get("TAVILY_API_KEY"), "TAVILY_API_KEY 未设置"
assert os.environ.get("FIRECRAWL_API_KEY"), "FIRECRAWL_API_KEY 未设置"
assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY 未设置"


def sheet_url_to_csv(url: str) -> str:
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    if not match:
        raise ValueError("无法解析 Google Sheets 链接")
    sheet_id = match.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"


def load_equipment_list(sheet_url: str, fallback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    try:
        csv_url = sheet_url_to_csv(sheet_url)
        df = pd.read_csv(csv_url)
        columns = {
            "id": "id",
            "编号": "id",
            "name_cn": "name_cn",
            "设备名称": "name_cn",
            "model_hint": "model_hint",
            "品牌及型号提示": "model_hint",
            "budget_wan_rmb": "budget_wan_rmb",
            "拟购预算(万RMB)": "budget_wan_rmb",
        }
        df = df.rename(columns=columns)
        required = {"id", "name_cn", "budget_wan_rmb"}
        if not required.issubset(df.columns):
            raise ValueError("表格缺少必要列")
        records = df.to_dict(orient="records")
        for record in records:
            record.setdefault("model_hint", "")
        return records
    except Exception as exc:
        print(f"读取表格失败，使用默认清单：{exc}")
        return fallback


EQUIPMENT_LIST = load_equipment_list(SHEET_URL, FALLBACK_EQUIPMENT_LIST)


def build_queries(item: Dict[str, Any]) -> List[str]:
    name_cn = item["name_cn"]
    model_hint = item.get("model_hint", "")
    base_en = model_hint if model_hint else name_cn

    cn_queries = [
        f"{name_cn} {model_hint} 价格",
        f"{name_cn} {model_hint} 招标 中标 采购 金额",
        f"{name_cn} {model_hint} 报价 单价",
    ]
    en_queries = [
        f"{base_en} price quotation",
        f"{base_en} tender award price",
        f"{base_en} procurement contract price",
    ]

    site_targets = [
        "ccgp.gov.cn",
        ".edu.cn",
        "china-bidding.com",
        "zhaobiao.cn",
    ]
    site_queries = [f"site:{site} {q}" for site in site_targets for q in cn_queries[:1]]

    return cn_queries + en_queries + site_queries


tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def tavily_search(queries: List[str], max_results: int = 8) -> List[Dict[str, Any]]:
    results = []
    for q in queries:
        resp = tavily_client.search(query=q, max_results=max_results)
        results.extend(resp.get("results", []))
    return results


def normalize_hostname(url: str) -> str:
    ext = tldextract.extract(url)
    if not ext.domain:
        return ""
    return f"{ext.domain}.{ext.suffix}".lower()


def is_allowed(url: str) -> bool:
    url_lower = url.lower()
    hostname = normalize_hostname(url_lower)

    if any(bad in url_lower for bad in BLOCKLIST_PATH_KEYWORDS):
        return False
    if hostname in BLOCKLIST_DOMAINS:
        return False

    if hostname in ALLOWED_DOMAINS:
        return True
    if hostname.endswith(".edu.cn"):
        return True
    if any(keyword in hostname for keyword in MANUFACTURER_KEYWORDS):
        return True

    path_hint = any(tag in url_lower for tag in ["/tender", "/procurement", "/zhaobiao", "/cg/"])
    if path_hint and hostname.endswith(".edu.cn"):
        return True

    return False


def filter_urls(results: List[Dict[str, Any]]) -> List[str]:
    urls = []
    for r in results:
        url = r.get("url")
        if not url:
            continue
        if is_allowed(url) and url not in urls:
            urls.append(url)
        if len(urls) >= MAX_URLS_PER_ITEM:
            break
    return urls


firecrawl_app = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])


@retry(wait=wait_exponential(multiplier=1, min=1, max=8), stop=stop_after_attempt(3))
def scrape_url(url: str) -> Dict[str, Any]:
    return firecrawl_app.scrape_url(
        url,
        params={
            "formats": ["markdown", "html"],
            "includeTags": ["title", "article", "main"],
        },
    )


class PriceEvidence(BaseModel):
    brand: Optional[str] = None
    model: Optional[str] = None
    price_value: Optional[float] = None
    currency: Optional[str] = None
    price_type: str = "unknown"
    matches_target: int = 0
    confidence: float = 0.0
    evidence_snippet: str = Field("", max_length=200)
    url: str
    published_date: Optional[str] = None


openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EXTRACTION_SYSTEM = """
你是严谨的采购价格抽取助手。只输出 JSON 对象，不要输出多余文本。
请从网页内容中抽取与目标设备相关的价格事实。
如果存在价格区间，取中点作为 price_value，并在 evidence_snippet 中说明。
price_type 仅允许：award/tender/list/dealer_quote/ecommerce/unknown。
matches_target: 1 表示与目标型号/品牌高度匹配，0 表示不匹配。
confidence 在 0-1 之间。
"""


def extract_price(equipment: Dict[str, Any], scraped: Dict[str, Any], url: str) -> Optional[PriceEvidence]:
    content = scraped.get("markdown") or scraped.get("content") or ""
    title = scraped.get("metadata", {}).get("title", "")
    published = scraped.get("metadata", {}).get("publishedDate")

    prompt = {
        "equipment": equipment,
        "page_title": title,
        "page_text": content[:6000],
        "url": url,
        "published_date": published,
    }

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        input=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )

    try:
        data = json.loads(response.output_text)
        if published and not data.get("published_date"):
            data["published_date"] = published
        data.setdefault("url", url)
        return PriceEvidence(**data)
    except (json.JSONDecodeError, ValidationError):
        return None


def fetch_fx_rates() -> Dict[str, float]:
    try:
        resp = requests.get("https://api.exchangerate.host/latest?base=CNY", timeout=10)
        resp.raise_for_status()
        rates = resp.json().get("rates", {})
        fx = {"CNY": 1.0, "RMB": 1.0}
        for cur, rate in rates.items():
            if rate:
                fx[cur.upper()] = 1 / rate
        return fx
    except Exception:
        return STATIC_FX.copy()


FX_RATES = fetch_fx_rates()


def to_wan_rmb(price: float, currency: str) -> Optional[float]:
    if price is None or not currency:
        return None
    fx = FX_RATES.get(currency.upper())
    if not fx:
        return None
    rmb = price * fx
    return round(rmb / 10000, 4)


def score_candidate(candidate: PriceEvidence, budget_wan: float) -> Dict[str, Any]:
    price_wan = to_wan_rmb(candidate.price_value, candidate.currency)
    if price_wan is None:
        diff_ratio = 999
    else:
        diff_ratio = abs(price_wan - budget_wan) / budget_wan

    return {
        "candidate": candidate,
        "price_wan": price_wan,
        "diff_ratio": diff_ratio,
        "priority": SOURCE_PRIORITY.get(candidate.price_type, 6),
    }


def select_top_candidates(candidates: List[PriceEvidence], budget_wan: float) -> (List[Dict[str, Any]], str):
    notes = []
    filtered = [c for c in candidates if c.matches_target == 1 and c.confidence >= MIN_CONFIDENCE]
    if not filtered:
        return [], "无匹配证据（匹配度或置信度不足）"

    scored = [score_candidate(c, budget_wan) for c in filtered]

    def bucket(diff: float) -> int:
        if diff <= 0.10:
            return 0
        if diff <= 0.25:
            return 1
        return 2

    scored.sort(key=lambda x: (bucket(x["diff_ratio"]), x["priority"], x["diff_ratio"]))
    top = scored[:3]

    if any(s["price_wan"] is None for s in top):
        notes.append("存在无法换算币种，已忽略或标记")
    if len(top) < 3:
        notes.append(f"仅找到 {len(top)} 条可用证据")
    if all(s["diff_ratio"] > 0.25 for s in top if s["price_wan"] is not None):
        notes.append("价格偏离预算超过 25%")

    return top, "；".join(notes) if notes else ""


def median_price(prices: List[float]) -> Optional[float]:
    nums = [p for p in prices if p is not None]
    if not nums:
        return None
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2 == 1:
        return nums[mid]
    return round((nums[mid - 1] + nums[mid]) / 2, 4)


def run_pipeline(equipment_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for item in equipment_list:
        queries = build_queries(item)
        search_results = tavily_search(queries)
        urls = filter_urls(search_results)

        evidences = []
        for url in urls:
            try:
                scraped = scrape_url(url)
                extracted = extract_price(item, scraped, url)
                if extracted and extracted.price_value and extracted.currency:
                    evidences.append(extracted)
            except Exception:
                continue

        selected, note = select_top_candidates(evidences, item["budget_wan_rmb"])
        prices_wan = [s["price_wan"] for s in selected]
        median_wan = median_price(prices_wan)

        row = {
            "编号": item["id"],
            "设备名称": item["name_cn"],
            "品牌及型号提示": item.get("model_hint", ""),
            "拟购预算(万RMB)": item["budget_wan_rmb"],
        }

        for idx in range(3):
            prefix = chr(ord('A') + idx)
            if idx < len(selected):
                sel = selected[idx]
                ev = sel["candidate"]
                row.update({
                    f"对比{prefix}_品牌": ev.brand,
                    f"对比{prefix}_型号": ev.model,
                    f"对比{prefix}_价格(万RMB)": sel["price_wan"],
                    f"对比{prefix}_原币种": ev.currency,
                    f"对比{prefix}_原价格": ev.price_value,
                    f"对比{prefix}_来源类型": ev.price_type,
                    f"对比{prefix}_可信度": ev.confidence,
                    f"对比{prefix}_URL": ev.url,
                    f"对比{prefix}_证据片段": ev.evidence_snippet,
                })
            else:
                row.update({
                    f"对比{prefix}_品牌": None,
                    f"对比{prefix}_型号": None,
                    f"对比{prefix}_价格(万RMB)": None,
                    f"对比{prefix}_原币种": None,
                    f"对比{prefix}_原价格": None,
                    f"对比{prefix}_来源类型": None,
                    f"对比{prefix}_可信度": None,
                    f"对比{prefix}_URL": None,
                    f"对比{prefix}_证据片段": None,
                })

        row.update({
            "中位数价格(万RMB)": median_wan,
            "证据数量": len(selected),
            "审计备注/建议": note,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_budget = df["拟购预算(万RMB)"].sum()
    total_median = df["中位数价格(万RMB)"].fillna(0).sum()
    summary = pd.DataFrame([
        {
            "总预算(万RMB)": total_budget,
            "中位数估算总额(万RMB)": total_median,
            "设备数量": len(df),
            "平均证据数量": round(df["证据数量"].mean(), 2),
        }
    ])
    return summary


def main() -> None:
    df = run_pipeline(EQUIPMENT_LIST)
    summary_df = build_summary(df)
    output_path = "/content/equipment_price_verification.xlsx"
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Evidence")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
    print(output_path)
    print(summary_df)


if __name__ == "__main__":
    main()
