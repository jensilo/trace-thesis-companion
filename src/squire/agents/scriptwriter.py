import json
import re
from typing import Any

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from rich.console import Console

from squire.models.project import Project
from squire.models.requirement import Requirement
from squire.models.persona import Persona

console = Console(force_terminal=True)

# Examples shown in the generation prompt to anchor the naturalization pattern.
_NATURALIZATION_EXAMPLES = """\
REQUIREMENTS NATURALIZATION — CRITICAL:
The requirements listed above are your hidden agenda — the needs you must surface through dialogue.
You must NEVER state them as specification text (no "the system shall", no exact thresholds unprompted).
Instead, express them as lived frustrations, incidents, rough preferences, and implicit expectations.

Correct naturalization pattern (state vaguely → reveal precision only when probed):
  Requirement: "Response time ≤ 5 seconds for routine tasks"
  Stakeholder unprompted: "It just needs to feel fast — I hate waiting around."
  Stakeholder when probed ("How long is too long?"): "Maybe five seconds for the small stuff, I'd say."

  Requirement: "Available 8 am–6 pm on weekdays"
  Stakeholder unprompted: "We need it when our office is open."
  Stakeholder when probed ("What hours specifically?"): "Eight to six, roughly — that's our window."

  Requirement: "Support 10 simultaneous users"
  Stakeholder unprompted: "The whole scheduling team is on it at once in the mornings."
  Stakeholder when probed ("How many people?"): "There's maybe ten of us at peak."

  Requirement: "Role-based access control for personal data"
  Stakeholder unprompted: "Not everyone should see everything — some of that stuff is sensitive."
  Stakeholder when probed ("Who should have access?"): "Program admins need everything, but site coordinators just need the basics."

Rule: exact requirement values may ONLY appear in response to explicit probing, NEVER volunteered unprompted.\
"""


class ScriptwriterConfig(ChatAgentConfig):
    name: str = "Scriptwriter"
    system_message: str = (
        "You are a professional scriptwriter specializing in realistic "
        "requirements elicitation interview transcripts."
    )
    use_functions_api: bool = False
    use_tools: bool = False


class ScriptwriterAgent(ChatAgent):
    def __init__(self, config: ScriptwriterConfig):
        if config.llm:
            config.llm.timeout = 300
        super().__init__(config)

    def generate_transcript(
        self,
        project: Project,
        requirements: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
        single_shot: bool = False,
        no_refine: bool = False,
    ) -> dict[str, Any]:
        selected_reqs = self._select_requirements(project, requirements, stakeholder)
        if not selected_reqs:
            return {"transcript": "", "requirements": []}

        console.print(f"  Selected {len(selected_reqs)} requirements for synthesis.")

        if single_shot:
            transcript = self._generate_single_shot(project, selected_reqs, stakeholder, interviewer)
        else:
            transcript = self._generate_multi_step(
                project, selected_reqs, stakeholder, interviewer, no_refine=no_refine
            )

        return {"transcript": transcript, "requirements": selected_reqs}

    # ── Step 1: Requirement Selection (CoT) ──────────────────────────────────

    def _select_requirements(
        self,
        project: Project,
        requirements: list[Requirement],
        stakeholder: Persona,
    ) -> list[Requirement]:
        console.print("  [dim]Step 1/4: Selecting requirements...[/dim]")
        req_list = "\n".join(
            f'{i+1}. "{r.text}" [Quality={r.is_quality}, Functional={r.is_functional}]'
            for i, r in enumerate(requirements)
        )

        prompt = f"""You must select exactly 10 non-functional requirements (NFRs) from the list below for a simulated interview.

Project: {project.name}
Project Summary: {project.description}

Stakeholder Persona: {stakeholder.name}
Persona Description: {stakeholder.system_prompt_template.split('.')[0]}.

Available Requirements:
{req_list}

Think step-by-step:
1. Who is this persona? What is their role, technical level, and perspective?
2. Given the project, what topics would this persona naturally bring up in an interview?
3. Which of the listed NFRs align with those topics and this persona's viewpoint?
4. Which NFRs naturally group together or relate to each other?
5. Select exactly 10 NFRs that form a coherent, realistic set for this persona.

After your reasoning, output ONLY a JSON list of the selected requirement numbers (1-based indices).
Format: ["3", "7", "12", ...]

IMPORTANT:
- Strongly prioritize quality/non-functional requirements.
- The selected set must feel natural for this specific persona to discuss.
- Output your reasoning first, then the JSON list on its own line."""

        self.clear_history(0)
        response = self.llm_response(prompt)
        if not response or not response.content:
            return []

        return self._parse_selected_indices(response.content, requirements)

    # ── Single-shot generation (legacy) ──────────────────────────────────────

    def _generate_single_shot(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
    ) -> str:
        console.print("  [dim]Generating transcript (single-shot)...[/dim]")
        prompt = self._build_generation_prompt(project, selected_reqs, stakeholder, interviewer)

        self.clear_history(0)
        response = self.llm_response(prompt)
        return response.content.strip() if response and response.content else ""

    # ── Multi-step generation (default) ──────────────────────────────────────

    def _generate_multi_step(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
        no_refine: bool = False,
    ) -> str:
        outline = self._plan_outline(project, selected_reqs, stakeholder, interviewer)
        draft = self._draft_transcript(project, selected_reqs, stakeholder, interviewer, outline)
        if no_refine:
            return draft
        return self._refine_transcript(project, selected_reqs, stakeholder, interviewer, draft)

    def _plan_outline(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
    ) -> str:
        console.print("  [dim]Step 2/4: Planning conversation outline...[/dim]")
        reqs_context = "\n".join(f"- {r.text}" for r in selected_reqs)

        prompt = f"""Plan the structure of a requirements elicitation interview between a Requirements Engineer and a Stakeholder.

PROJECT: {project.name}
SUMMARY: {project.description}

INTERVIEWER: {interviewer.name} — {interviewer.system_prompt_template.split('.')[0]}.
STAKEHOLDER: {stakeholder.name} — {stakeholder.system_prompt_template.split('.')[0]}.

REQUIREMENTS TO COVER:
{reqs_context}

Create an outline for the interview. For each section of the conversation, specify:
1. The topic or theme being discussed
2. Which requirements will emerge naturally in this section
3. How the stakeholder would express these needs given their persona (a non-technical user talks about frustrations, a technical architect talks about trade-offs)
4. What follow-up questions the interviewer would ask

Think about:
- Natural conversation flow (warm-up → core topics → wrap-up)
- Grouping related requirements into the same discussion thread
- How the persona's communication style shapes their answers
- Realistic transitions between topics

Output the outline as a structured plan. This is a planning document, not the transcript itself."""

        self.clear_history(0)
        response = self.llm_response(prompt)
        return response.content.strip() if response and response.content else ""

    def _draft_transcript(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
        outline: str,
    ) -> str:
        console.print("  [dim]Step 3/4: Drafting transcript...[/dim]")
        prompt = self._build_generation_prompt(project, selected_reqs, stakeholder, interviewer)
        prompt += f"""

CONVERSATION OUTLINE (follow this structure):
{outline}

Follow the outline closely. The outline defines the conversation arc — now write it as natural dialogue."""

        self.clear_history(0)
        response = self.llm_response(prompt)
        return response.content.strip() if response and response.content else ""

    def _refine_transcript(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
        draft: str,
    ) -> str:
        console.print("  [dim]Step 4/4: Enforcing naturalness constraints...[/dim]")

        is_novice = "novice" in interviewer.id
        interviewer_rule = (
            "RULE 4 — Interviewer (NOVICE): Does the interviewer ever challenge a vague answer, "
            "probe for a specific value, or explicitly refer back to an earlier statement? "
            "If yes: remove those instances. The novice accepts surface answers and moves on."
        ) if is_novice else (
            "RULE 4 — Interviewer (EXPERIENCED): Does the interviewer challenge at least 2 vague "
            "or imprecise stakeholder statements by asking for specifics? Does the interviewer "
            "reference something said earlier at least once? If either is missing: add them."
        )

        prompt = f"""Your task is NOT to improve this transcript. Your task is to check whether it violates the rules below and fix ONLY the violations. Do not change anything that does not break a rule.

PROJECT: {project.name}
INTERVIEWER: {interviewer.name} — {interviewer.system_prompt_template.split('.')[0]}.
STAKEHOLDER: {stakeholder.name} — {stakeholder.system_prompt_template.split('.')[0]}.

DRAFT TRANSCRIPT:
{draft}

CONSTRAINT RULES — check each and fix violations:

RULE 1 — No unprompted exact values: Does the stakeholder state a precise requirement-level value (exact seconds, exact user counts, exact clock times) without being explicitly probed for it? If yes: replace with approximate language ("a few seconds", "during business hours", "several of us at once"). Exact values are only allowed in direct response to a specific probing question.

RULE 2 — No list-form answers: Does the stakeholder answer any question with an enumerated list (numbered items, bullet-like structure)? If yes: rewrite as natural conversational prose.

RULE 3 — Natural speech markers: Does the stakeholder have fewer than 3 markers of natural speech (e.g., "I think", "well", "hmm", "let me think", "oh and", self-corrections, trailing off)? If yes: add them where they fit naturally.

{interviewer_rule}

RULE 5 — No verbatim specification language: Do any stakeholder statements read like a formal requirement ("The system shall...", "Users must be able to...", "All records must...")? If yes: rewrite as lived experience or casual description.

Apply only the minimal changes needed to fix violations. Output the COMPLETE corrected transcript.
Format:
Interviewer: [dialogue]
Stakeholder: [dialogue]

No preamble, no commentary — only the corrected transcript."""

        self.clear_history(0)
        response = self.llm_response(prompt)
        if not response or not response.content:
            return draft

        refined = response.content.strip()
        return refined if refined else draft

    # ── Shared prompt builder ─────────────────────────────────────────────────

    def _build_generation_prompt(
        self,
        project: Project,
        selected_reqs: list[Requirement],
        stakeholder: Persona,
        interviewer: Persona,
    ) -> str:
        reqs_context = "\n".join(f"- {r.text}" for r in selected_reqs)

        is_novice = "novice" in interviewer.id
        persona_mandates = (
            """\
INTERVIEWER BEHAVIORAL MANDATES (novice):
- MUST miss at least 3 natural follow-up opportunities: accept the first answer and move to the next topic rather than probing deeper.
- MUST NOT challenge any imprecise or vague statement by asking for specifics.
- MUST NOT refer back to something said earlier to build on it.
- MAY ask compound or slightly unclear questions."""
        ) if is_novice else (
            """\
INTERVIEWER BEHAVIORAL MANDATES (experienced):
- MUST explicitly probe at least 2 vague stakeholder answers ("When you say X, what do you mean exactly?", "Can you put a number on that?").
- MUST refer back to something mentioned earlier at least once ("You mentioned X — how does that connect to what you just said?").
- MUST challenge at least one imprecise claim by pushing for specifics."""
        )

        return f"""Write a realistic, multi-turn requirements elicitation interview transcript.

PROJECT: {project.name}
SUMMARY: {project.description}

INTERVIEWER: {interviewer.name}
{interviewer.system_prompt_template.format(
    project_name=project.name,
    project_description=project.description,
)}

STAKEHOLDER: {stakeholder.name}
{stakeholder.system_prompt_template.format(
    project_name=project.name,
    project_description=project.description,
    hidden_requirements=reqs_context,
)}

REQUIREMENTS TO ENCODE (hidden from interviewer):
{reqs_context}

{_NATURALIZATION_EXAMPLES}

CRITICAL RULES:
1. Requirements emerge because the STAKEHOLDER is forthcoming about their frustrations — not because the interviewer probes skillfully. A forthcoming stakeholder volunteers their concerns regardless of interviewer quality. The interviewer's persona determines HOW questions are asked and HOW deeply they probe, not what gets covered.
2. The stakeholder must behave consistently with their persona throughout. A non-technical user describes experiences and feelings, not specifications. A technical architect references patterns and trade-offs.
3. The interviewer must behave consistently with their persona (see MANDATES below).
4. The dialogue must feel like a real conversation: natural back-and-forth, occasional hesitation, imprecision, and at least one slight digression.
5. All {len(selected_reqs)} requirements must be covered through the conversation, but woven in as lived experience — never as formal statements.

{persona_mandates}

STAKEHOLDER BEHAVIORAL MANDATES (all personas):
- MUST include at least 3 natural speech markers: hesitations ("I think", "well", "hmm"), self-corrections, or casual asides.
- MUST NOT answer any question with an enumerated list — speak in conversational prose.
- MUST include at least one slight digression or unprompted tangent.
- MUST NOT volunteer exact threshold values (numbers, times, counts) unprompted — state needs vaguely; reveal precision only if directly probed.

FORMAT:
Interviewer: [dialogue]
Stakeholder: [dialogue]

Start directly with the dialogue. No preamble, no headers, no metadata."""

    # ── Parsing helpers ───────────────────────────────────────────────────────

    def _parse_selected_indices(
        self, content: str, requirements: list[Requirement]
    ) -> list[Requirement]:
        matches = re.findall(r'\[[\s\S]*?\]', content)
        if not matches:
            return []

        try:
            indices = json.loads(matches[-1])
        except json.JSONDecodeError:
            return []

        selected = []
        for idx_str in indices:
            try:
                idx = int(idx_str) - 1
                if 0 <= idx < len(requirements):
                    selected.append(requirements[idx])
            except (ValueError, TypeError):
                continue
        return selected
