import json
import os
import re
import time

# 数据库连接配置
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "dataplatform-backend-pgsql.fftaicorp.com"),
    "port": int(os.getenv("DB_PORT", "30673")),
    "user": os.getenv("DB_USER", "readonly_user"),
    "password": os.getenv("DB_PASSWORD", "fftai2015"),
    "database": os.getenv("DB_NAME", "data_collection"),
}

# Qwen API 配置（官方 OpenAI 兼容方式）
QWEN_API_KEY = os.getenv(
    "QWEN_API_KEY",
    os.getenv("DASHSCOPE_API_KEY", "sk-bc4148025495476badfc33f19968fef1"),
)
QWEN_BASE_URL = os.getenv(
    "QWEN_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen3.5-flash")
QWEN_TIMEOUT_SECONDS = max(10, int(os.getenv("QWEN_TIMEOUT_SECONDS", "120")))
QWEN_MAX_RETRIES = max(0, int(os.getenv("QWEN_MAX_RETRIES", "1")))
QWEN_BATCH_SIZE = max(1, int(os.getenv("QWEN_BATCH_SIZE", "40")))

# 缓存文件：避免重复调用 API
CACHE_FILE_NAME = "qnexo_repo_id_cache.json"


def sanitize_repo_id(raw_text):
    """将 repo_id 规范化为可用作文件夹名的字符串（仅 a-z0-9_）。"""
    if not raw_text:
        return "custom_task"

    text = str(raw_text).strip().lower()

    # 处理模型可能返回的代码块或解释
    text = text.replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        text = "custom_task"

    # 文件夹名首字符避免数字
    if text[0].isdigit():
        text = f"task_{text}"

    # 控制长度，避免路径过长
    text = text[:64].rstrip("_")

    return text or "custom_task"


def strip_code_fence(text):
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_json_from_text(text):
    """从模型返回文本中提取 JSON（兼容夹带文本的情况）。"""
    cleaned = strip_code_fence(text)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        snippet = cleaned[start : end + 1]
        return json.loads(snippet)

    raise ValueError("模型返回中未找到有效 JSON")


def parse_batch_repo_ids(content_text):
    """解析批量返回结果。

    支持两种结构：
    1) {"items": [{"task_id": "1", "repo_id": "xxx"}, ...]}
    2) {"1": "xxx", "2": "yyy"}
    """
    obj = extract_json_from_text(content_text)
    result = {}

    if isinstance(obj, dict) and isinstance(obj.get("items"), list):
        for item in obj["items"]:
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id", "")).strip()
            repo_id = item.get("repo_id")
            if task_id and repo_id:
                result[task_id] = sanitize_repo_id(repo_id)
        return result

    if isinstance(obj, dict):
        for k, v in obj.items():
            task_id = str(k).strip()
            if not task_id:
                continue
            if isinstance(v, str):
                result[task_id] = sanitize_repo_id(v)
            elif isinstance(v, dict) and v.get("repo_id"):
                result[task_id] = sanitize_repo_id(v.get("repo_id"))
        return result

    raise ValueError("模型返回 JSON 结构不符合预期")


def call_qwen_repo_ids_batch(task_map):
    """单个批次调用：输入 task_id->task，输出 task_id->repo_id。"""
    if not task_map:
        return {}

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少依赖 openai，请先安装后再运行脚本。") from exc

    if not QWEN_API_KEY:
        raise RuntimeError("未配置 QWEN_API_KEY 或 DASHSCOPE_API_KEY")

    client = OpenAI(
        api_key=QWEN_API_KEY,
        base_url=QWEN_BASE_URL,
        timeout=QWEN_TIMEOUT_SECONDS,
    )

    input_payload = json.dumps(task_map, ensure_ascii=False)

    messages = [
        {
            "role": "system",
            "content": (
                "你是 repo_id 生成器。"
                "请根据每条任务描述生成 repo_id。"
                "repo_id 必须是小写 snake_case，仅含 a-z0-9_，可作为文件夹名。"
                "长度建议 3~8 个词，尽量体现动作+关键物体+关键目标位置。"
                "必须覆盖所有输入 task_id，不允许遗漏或新增。"
                "只输出 JSON，不要输出解释。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请返回 JSON 对象，key 为 task_id，value 为 repo_id，例如："
                "{\"123\":\"pick_apple_to_basket\"}。\n"
                "以下是输入（JSON 对象，key 为 task_id，value 为 task 描述）：\n"
                f"{input_payload}"
            ),
        },
    ]

    last_error = None
    for retry in range(QWEN_MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=QWEN_MODEL,
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=4000,
            )

            content = (completion.choices[0].message.content or "").strip()
            parsed = parse_batch_repo_ids(content)

            # 只接受输入范围内 task_id
            filtered = {}
            for task_id in task_map:
                if task_id in parsed:
                    filtered[task_id] = sanitize_repo_id(parsed[task_id])

            if not filtered:
                raise RuntimeError("Qwen 批量返回为空")

            return filtered

        except Exception as exc:
            last_error = exc
            if retry < QWEN_MAX_RETRIES:
                time.sleep(1.0 * (retry + 1))

    raise RuntimeError(f"调用 Qwen 批量 API 失败: {last_error}")


def split_task_map(task_map):
    items = list(task_map.items())
    mid = len(items) // 2
    return dict(items[:mid]), dict(items[mid:])


def call_qwen_repo_ids_batch_resilient(task_map, batch_label="batch"):
    """稳定批量调用：超时时自动拆分，尽量保留 API 生成结果。"""
    if not task_map:
        return {}

    try:
        parsed = call_qwen_repo_ids_batch(task_map)

        if len(parsed) == len(task_map):
            print(f"{batch_label} 调用成功: {len(parsed)}/{len(task_map)}")
            return parsed

        missing = {k: v for k, v in task_map.items() if k not in parsed}
        print(
            f"{batch_label} 部分成功: {len(parsed)}/{len(task_map)}，"
            f"缺失 {len(missing)} 条，继续拆分补齐"
        )

        # 只有部分缺失时，仅对缺失部分做递归补齐
        if missing:
            recovered = call_qwen_repo_ids_batch_resilient(
                missing,
                batch_label=f"{batch_label}-missing",
            )
            parsed.update(recovered)

        return parsed

    except Exception as exc:
        if len(task_map) == 1:
            only_task_id = next(iter(task_map.keys()))
            print(
                f"{batch_label} 单条调用失败，task_id={only_task_id}，"
                f"将回退规则生成。原因: {exc}"
            )
            return {}

        left_map, right_map = split_task_map(task_map)
        print(
            f"{batch_label} 调用失败，拆分重试: "
            f"{len(left_map)} + {len(right_map)}。原因: {exc}"
        )

        merged = {}
        merged.update(call_qwen_repo_ids_batch_resilient(left_map, f"{batch_label}-L"))
        merged.update(call_qwen_repo_ids_batch_resilient(right_map, f"{batch_label}-R"))
        return merged


def generate_repo_id_rule_based(task_en):
    """API 失败时的兜底逻辑。"""
    if not task_en or task_en == "Unknown task":
        return "unknown_task"

    text = re.sub(r"[^\w\s]", " ", task_en.lower())

    stop_words = {
        "use",
        "left",
        "right",
        "hand",
        "hands",
        "to",
        "the",
        "a",
        "an",
        "and",
        "from",
        "on",
        "in",
        "into",
        "onto",
        "it",
        "them",
        "back",
        "up",
        "down",
        "bend",
        "stand",
        "twist",
        "body",
        "torso",
        "turn",
        "waist",
        "squat",
        "desktop",
        "table",
        "at",
        "is",
        "if",
        "for",
        "with",
        "then",
        "of",
        "your",
        "person",
        "s",
        "red",
        "blue",
        "green",
        "yellow",
        "white",
        "black",
        "purple",
        "pink",
        "brown",
        "beige",
        "khaki",
        "azure",
        "light",
        "dark",
        "plastic",
        "wooden",
        "metal",
        "glass",
        "paper",
        "cardboard",
        "stainless",
        "steel",
        "toy",
        "small",
        "large",
        "thickened",
        "mini",
    }

    words = [w for w in text.split() if w not in stop_words]

    actions = {
        "pick",
        "put",
        "place",
        "throw",
        "push",
        "pull",
        "wipe",
        "open",
        "close",
        "give",
        "insert",
        "fold",
        "sweep",
        "hold",
        "slide",
        "press",
        "stir",
    }

    repo_words = []

    for i, w in enumerate(words):
        if w in actions:
            repo_words.append(w)
            count = 0
            for j in range(i + 1, len(words)):
                if words[j] not in actions:
                    repo_words.append(words[j])
                    count += 1
                if count >= 3:
                    break
            break

    if not repo_words:
        repo_words = words[:4]

    if not repo_words:
        return "custom_task"

    return sanitize_repo_id("_".join(repo_words))


def load_repo_id_cache(cache_path):
    if not os.path.exists(cache_path):
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_repo_id_cache(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def ensure_unique_repo_id(base_repo_id, task_id, used_repo_ids):
    candidate = sanitize_repo_id(base_repo_id)
    if candidate not in used_repo_ids:
        return candidate

    candidate_with_task = sanitize_repo_id(f"{candidate}_{task_id}")
    if candidate_with_task not in used_repo_ids:
        return candidate_with_task

    idx = 2
    while True:
        final_candidate = sanitize_repo_id(f"{candidate_with_task}_{idx}")
        if final_candidate not in used_repo_ids:
            return final_candidate
        idx += 1


def build_task_en(descriptions):
    task_en = "Unknown task"
    if not descriptions:
        return task_en

    if isinstance(descriptions, str):
        try:
            descriptions = json.loads(descriptions)
        except json.JSONDecodeError:
            descriptions = {}

    if isinstance(descriptions, dict):
        en_content = descriptions.get("en")
        if en_content is not None:
            if isinstance(en_content, list):
                task_en = " ".join([str(item) for item in en_content if item])
            else:
                task_en = str(en_content)

    return task_en


def generate_repo_ids_with_fallback(task_text_by_id, repo_id_cache):
    """批量生成 repo_id：优先缓存，其次一次性调用 Qwen，失败则规则回退。"""
    repo_ids = {}
    pending_for_api = {}

    for task_id, task_en in task_text_by_id.items():
        if not task_en or task_en == "Unknown task":
            repo_ids[task_id] = "unknown_task"
            continue

        cached = repo_id_cache.get(task_en)
        if cached:
            repo_ids[task_id] = sanitize_repo_id(cached)
        else:
            pending_for_api[task_id] = task_en

    api_generated = {}
    if pending_for_api:
        pending_items = list(pending_for_api.items())
        total = len(pending_items)
        total_batches = (total + QWEN_BATCH_SIZE - 1) // QWEN_BATCH_SIZE
        print(
            f"待批量调用 Qwen 生成 repo_id: {total} 条，"
            f"分 {total_batches} 批（每批最多 {QWEN_BATCH_SIZE} 条）"
        )

        for i in range(0, total, QWEN_BATCH_SIZE):
            chunk_items = pending_items[i : i + QWEN_BATCH_SIZE]
            chunk_map = dict(chunk_items)
            batch_no = (i // QWEN_BATCH_SIZE) + 1
            batch_label = f"batch_{batch_no}/{total_batches}"

            chunk_generated = call_qwen_repo_ids_batch_resilient(chunk_map, batch_label)
            api_generated.update(chunk_generated)

        print(f"Qwen 批量生成完成: {len(api_generated)}/{total} 条")

    # 补齐全部 task_id
    for task_id, task_en in pending_for_api.items():
        repo_id = api_generated.get(task_id)
        if not repo_id:
            repo_id = generate_repo_id_rule_based(task_en)

        repo_id = sanitize_repo_id(repo_id)
        repo_ids[task_id] = repo_id
        repo_id_cache[task_en] = repo_id

    return repo_ids


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "gr3_task_configs.json")
    cache_path = os.path.join(project_root, CACHE_FILE_NAME)

    repo_id_cache = load_repo_id_cache(cache_path)
    used_repo_ids = set()

    conn = None
    cursor = None

    try:
        try:
            import psycopg2
        except ImportError:
            print("缺少依赖 psycopg2，请先安装后再运行脚本。")
            return

        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT task_id FROM episodes where equipment = 'qnexo' "
        )
        task_records = cursor.fetchall()

        if not task_records:
            print("未在 episodes 表中找到 equipment = 'qnexo' 的数据")
            return

        task_ids = [str(row[0]) for row in task_records if row[0] is not None]
        results = []

        if task_ids:
            placeholders = ",".join(["%s"] * len(task_ids))
            query_tasks = f"SELECT id, descriptions FROM tasks WHERE id IN ({placeholders})"
            cursor.execute(query_tasks, tuple(task_ids))

            tasks_data = {str(row[0]): row[1] for row in cursor.fetchall()}
            task_text_by_id = {}

            for task_id in task_ids:
                if task_id not in tasks_data:
                    continue
                task_text_by_id[task_id] = build_task_en(tasks_data.get(task_id))

            base_repo_ids = generate_repo_ids_with_fallback(task_text_by_id, repo_id_cache)

            for task_id in task_ids:
                if task_id not in task_text_by_id:
                    continue

                task_en = task_text_by_id[task_id]
                base_repo_id = base_repo_ids.get(task_id, "custom_task")
                repo_id_val = ensure_unique_repo_id(base_repo_id, task_id, used_repo_ids)
                used_repo_ids.add(repo_id_val)

                config_item = {
                    "repo_id": repo_id_val,
                    "query": f"task_id = '{task_id}'",
                    "task": task_en,
                    "mode": "video",
                    "robot_type": "gr3qnexo",
                    "video_config": "480x832",
                    "case_size": 500,
                    "workers": 6,
                    "enable_streaming": True,
                    "timeout": 1000,
                    "auto_cleanup_raw": True,
                    "auto_cleanup_converted": False,
                    "max_disk_usage_gb": 50.0,
                    "disk_check_interval": 30,
                }
                results.append(config_item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

        save_repo_id_cache(cache_path, repo_id_cache)
        print(f"成功将 {len(results)} 条配置写入 {output_path}")
        print(f"repo_id 缓存已写入: {cache_path}")

    except Exception as e:
        print(f"执行过程中发生异常: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
