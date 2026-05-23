import asyncio
import json
import re

import litellm
from rich.console import Console

from iso_labeller.taxonomy import ISO_TAXONOMY

console = Console(force_terminal=True)

_NON_RETRYABLE = (
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    litellm.NotFoundError,
    litellm.UnprocessableEntityError,
)

_CHAR_DESCRIPTIONS: dict[str, str] = {
    "Functional suitability": "does what is expected",
    "Performance efficiency": "speed and resource use",
    "Compatibility": "works alongside other systems",
    "Interaction capability": "quality of user interaction",
    "Reliability": "operates without failure",
    "Security": "protects against threats",
    "Maintainability": "ease of modification",
    "Flexibility": "adapts to change",
    "Safety": "avoids harm to people or environment",
}

_TAXONOMY_TEXT = "\n".join(
    f"\n{char} ({_CHAR_DESCRIPTIONS[char]}):\n"
    + "\n".join(f"  - {sub}" for sub in subs)
    for char, subs in ISO_TAXONOMY.items()
)

_PROMPT_TEMPLATE = """\
You are a software quality requirements analyst.

Classify the following software requirement according to ISO/IEC 25010:2023 quality sub-characteristics.

REQUIREMENT:
{requirement_text}

ISO/IEC 25010:2023 TAXONOMY — assign ALL applicable sub-characteristics:
{taxonomy}

INSTRUCTIONS:
- Assign every sub-characteristic that the requirement addresses or constrains.
- Multi-label: assign more than one if applicable.
- Use the EXACT sub-characteristic names listed above.
- If the requirement describes purely what the system must do (a functional requirement), \
assign Functional suitability sub-characteristics.
- Do NOT invent sub-characteristic names; only use the names above.

Respond with JSON only, no prose:
{{"iso_sub_chars": ["sub-char 1", "sub-char 2"]}}
"""


def _build_prompt(requirement_text: str) -> str:
    return _PROMPT_TEMPLATE.format(
        requirement_text=requirement_text,
        taxonomy=_TAXONOMY_TEXT,
    )


def _parse_sub_chars(raw: str) -> list[str]:
    try:
        data = json.loads(raw)
        return data.get("iso_sub_chars", [])
    except json.JSONDecodeError:
        match = re.search(r'"iso_sub_chars"\s*:\s*\[([^\]]*)\]', raw, re.DOTALL)
        if match:
            return re.findall(r'"([^"]+)"', match.group(1))
    return []


async def _classify_one(
    requirement_text: str,
    model: str,
    temperature: float,
    sem: asyncio.Semaphore,
    max_retries: int = 5,
    base_delay: float = 0.3,
) -> tuple[list[str], str | None]:
    prompt = _build_prompt(requirement_text)
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                try:
                    response = await litellm.acompletion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        response_format={"type": "json_object"},
                    )
                except litellm.BadRequestError:
                    response = await litellm.acompletion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                    )
                raw = response.choices[0].message.content.strip()
                return _parse_sub_chars(raw), raw
            except _NON_RETRYABLE:
                raise
            except Exception as e:
                if attempt == max_retries:
                    raise
                delay = base_delay * (2**attempt)
                console.print(
                    f"  [yellow]Attempt {attempt + 1} failed ({type(e).__name__}), "
                    f"retrying in {delay:.1f}s…[/yellow]"
                )
                await asyncio.sleep(delay)
    return [], None


async def classify_all(
    rows: list[dict],
    model: str,
    temperature: float,
    concurrency: int,
) -> list[tuple[list[str], str | None]]:
    sem = asyncio.Semaphore(concurrency)
    completed = 0

    async def _tracked(row: dict) -> tuple[list[str], str | None]:
        nonlocal completed
        result = await _classify_one(row["RequirementText"], model, temperature, sem)
        completed += 1
        if completed % 50 == 0 or completed == len(rows):
            console.print(f"  [dim]Classified {completed}/{len(rows)} requirements[/dim]")
        return result

    return await asyncio.gather(*[_tracked(row) for row in rows])
