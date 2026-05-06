import base64
import hashlib
import json
import os
import urllib.error
import urllib.request


class LLMReviewer:
    """
    Handles LLM-based visual critique of PBN results with local LRU-eviction caching.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://openrouter.ai/api/v1",
        model: str = "google/gemini-2.0-flash-001",
        max_cache_size: int = 100,
    ):
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_cache_size = max_cache_size
        self.cache_path = os.path.join(os.path.dirname(__file__), ".llm_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        # TODO: Consider adding time-based expiration (TTL) or automatic file-size
        # limit checks in addition to the current LRU size-based eviction strategy
        # to ensure the cache file remains small and clean.
        with open(self.cache_path, "w") as f:
            json.dump(self.cache, f, indent=2)

    def _get_image_hash(self, img_bytes: bytes) -> str:
        return hashlib.sha256(img_bytes).hexdigest()

    def review_image(
        self, img_bytes: bytes, mime_type: str = "image/webp", prompt_type: str = "pbn_standard"
    ) -> str | None:
        """
        Submits image for review, returns critique text or None on failure.
        Maintains LRU order and size eviction.
        """
        img_hash = self._get_image_hash(img_bytes)
        cache_key = f"{img_hash}_{self.model}_{prompt_type}"

        # If cache hit, update access order (LRU)
        if cache_key in self.cache:
            val = self.cache.pop(cache_key)
            self.cache[cache_key] = val
            self._save_cache()
            return val

        critique = self._query_llm(img_bytes, mime_type, prompt_type)
        if critique:
            # Evict oldest entry if size limit reached
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                self.cache.pop(oldest_key)
            self.cache[cache_key] = critique
            self._save_cache()
            return critique
        return None

    def _query_llm(self, img_bytes: bytes, mime_type: str, prompt_type: str) -> str | None:
        b64_img = base64.b64encode(img_bytes).decode("utf-8")

        prompts = {
            "pbn_standard": (
                "You are an expert Paint-By-Number quality inspector. "
                "Evaluate this generated PBN template. "
                "1. Region Clarity: Are regions well-defined or fragmented into tiny 'specks'? "
                "2. Labels: Are numeric labels readable and correctly placed in center of regions? "
                "3. Artistic Faithfulness: Does it capture the essence of the original subjects? "
                "4. Defects: Identify any 'leaking' colors, gaps, or jagged edges. "
                "Provide a concise critique and a score out of 10."
            )
        }

        prompt = prompts.get(prompt_type, prompts["pbn_standard"])

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64_img}"}},
                    ],
                }
            ],
            "max_tokens": 500,
            "temperature": 0.2,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/tazztone/pbn_node",  # Required by OpenRouter
            "X-Title": "PBN Node Quality Bot",
        }

        try:
            req = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                return res_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LLM API Error: {e}")
            return None
