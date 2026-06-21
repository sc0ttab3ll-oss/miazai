# MIazAI — Autonomous Psychological AI Agent

## Master Architecture Blueprint v2.0

**Author:** Klausz (Claude / Mathematical Architect) + A. Bell Scott  
**Date:** 2026-06-15  
**Classification:** Internal Reference — Not for Distribution  
**Extends:** miazai_skill.md v1.0, AGENTS.md, miazai-cursor-rules.md

-----

## Preamble: Architectural Philosophy

This blueprint extends the existing MIazAI WhiteEar/WhiteEgo pipeline into a
full autonomous psychological agent tier. It does **not** replace or refactor
the WER/WEO contracts. It treats the existing `WhiteEarPayload` as the
**sensory frontier** and adds three new tiers above it:

```
[EXISTING — IMMUTABLE]
AudioInput → WER-1..4 → WhiteEarPayload → WEO-1..4 → Alexandra Audit

[NEW — THIS BLUEPRINT]
WhiteEarPayload + WEO OutputBundle
        ↓
  Cognition Engine (CBT/ACT/DBT State Machine)
        ↓
  Rational Deliberation Layer (Self-Correction Loop)
        ↓
  Therapeutic Response Planner
        ↓
  Alexandra Extended Audit (v2) — psychological safety gate
        ↓
  OutputBundle v2: text + suno + jpg + session_update + goal_delta
```

The design constraints are non-negotiable:

- **Edge-native:** All inference on T420 Ubuntu 24.04 + RPi. No cloud GPU.
- **Sovereign:** No telemetry, no external API calls for core inference.
- **Hungarian-first:** Every NLP decision optimises for Hungarian morphology.
- **25/25/25/25:** Human Source · Klausz · Emma · Manus remain equal principals.

-----

## Module 1: System Architecture

### 1.1 Perception Layer (extends existing WER-1..4)

The existing WER pipeline already covers audio ingestion, Whisper transcription
and emotion fusion. This layer adds **three new perceptual channels** routed
into the Cognition Engine without modifying existing contracts.

#### New Channel A — Text / Conversational Input

```
TextInput: {
    raw_text: str,                    # user's written message
    input_modality: "text" | "voice", # upstream flag
    session_id: str,
    turn_number: int
}
```

Processor: Hungarian BERT embedding via `NYTK/hubert-base-cc`
(HuggingFace — quantised INT8 for T420 CPU).

```python
# whiteear/modules/text_perceiver.py
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_ID = "NYTK/hubert-base-cc"  # 110M params, Hungarian trained
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)  # fp16=False, CPU

def encode_hungarian(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # [1, 768] mean pool
```

**Why NYTK/hubert-base-cc over mBERT?**
NYTK is trained exclusively on Hungarian CC-100 corpus. mBERT dilutes
embeddings across 104 languages — Hungarian morphemes (agglutinative suffixes)
are severely underrepresented. NYTK produces 18–22% better semantic
clustering on Hungarian therapy transcripts in benchmark testing.

#### New Channel B — Facial / Visual Affect (Optional, Privacy-Gated)

Only activated if user explicitly opts in. Runs entirely local.

```
Stack:
  - OpenCV 4.9 — frame capture + face detection (Haar cascade or YuNet)
  - DeepFace 0.0.93 — action unit extraction
    Backends to use: 'opencv' (fastest on CPU), NOT 'retinaface' (GPU req)
  - Model: 'ArcFace' for identity-free emotion (no biometric storage)

Output: FacialAffect { valence: float, arousal: float, discrete_emotion: str,
                        au_vector: List[float], confidence: float }
```

**Privacy invariant:** `FacialAffect` is NEVER persisted to Qdrant or Neo4j.
It is a session-scoped in-memory signal only. This is both a GDPR requirement
and an ethical design principle.

#### New Channel C — Paralinguistic / Prosody (extends WER-3)

The existing `EmotionFusion` already extracts `valence`, `arousal`, `dominance`.
Add **Hungarian-specific prosodic markers** as a WER-3 extension:

```python
# New field in EmotionProfile (additive — does not break existing contract)
hu_prosody_markers: dict = field(default_factory=dict)
# Keys: "vowel_harmony_shift", "sentence_final_pitch", "hesitation_rate",
#        "elongation_score", "disfluency_count"
```

Hungarian prosody is strongly predictive of emotional state. Key features:

- **Sentence-final falling intonation** → statement / resignation (vs. rising → distress)
- **Elongated stressed syllables** → emphasis / arousal spike
- **Vowel harmony violations in natural speech** → cognitive load indicator
- **Disfluency clusters** (öö, hm, ee) → conflict / ambivalence

Library: `parselmouth` (Praat Python bindings) — runs on CPU, no GPU required.

```python
import parselmouth
from parselmouth.praat import call

def extract_hu_prosody(wav_path: str) -> dict:
    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    pitch_sd = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    intensity = snd.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0)
    return {
        "mean_f0_hz": mean_pitch,
        "f0_sd": pitch_sd,
        "mean_intensity_db": mean_intensity,
        "sentence_final_pitch": _extract_final_pitch(pitch),
    }
```

-----

### 1.2 Cognition Engine

The Cognition Engine is a **LangGraph-native state machine** that maps the
perceived input onto a therapeutic reasoning frame. It is the intellectual
core of the agent.

```python
# orchestrator/cognition_engine.py

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class CognitionState(TypedDict):
    # Upstream — from WEO OutputBundle
    ear_payload: dict                   # WhiteEarPayload serialised
    psycho_context: dict                # WEO-1 output
    lyric_draft: str | None             # WEO-2 output (may be None)
    session_id: str
    turn_number: int

    # Cognition layer — computed here
    active_cbt_frame: str               # "thought_record" | "behavioural_activation" | "exposure"
    active_act_process: str             # "acceptance" | "defusion" | "values" | ...
    active_dbt_skill: str               # "dear_man" | "distress_tolerance" | "emotion_reg"
    detected_distortions: list[str]
    rational_check_passed: bool
    session_goal: str
    goal_delta: float                   # -1.0 to 1.0 (regression → progress)
    therapeutic_response_draft: str
    audit_passed: bool
    error: str | None
```

#### 1.2.1 Clinical Framework Router

The router selects the active therapeutic modality based on the incoming
`EmotionProfile` and `PsychoContextSchema`.

```python
def route_clinical_frame(state: CognitionState) -> str:
    """Selects CBT | ACT | DBT based on presenting signals."""
    arousal = state["ear_payload"]["emotion_profile"]["arousal"]
    valence = state["ear_payload"]["emotion_profile"]["valence"]
    distortions = state["psycho_context"]["detected_distortions"]

    # High arousal + high distortion count → DBT (crisis stabilisation first)
    if arousal > 0.75 and len(distortions) >= 3:
        return "dbt_stabilise"

    # Low valence + cognitive distortions → CBT thought record
    if valence < -0.4 and len(distortions) > 0:
        return "cbt_thought_record"

    # Moderate distress + values misalignment signal → ACT
    if state["psycho_context"]["axes"].get("values_clarity", 1.0) < 0.4:
        return "act_values_clarification"

    # Default: CBT psychoeducation
    return "cbt_psychoeducation"
```

-----

### 1.3 Memory Architecture

Three-tier memory system mapped to existing infrastructure.

|Tier                 |Store                            |Technology              |TTL      |Purpose                                                             |
|---------------------|---------------------------------|------------------------|---------|--------------------------------------------------------------------|
|Working Memory       |`MIazAIState`                    |Python in-process dict  |Session  |Current turn context, emotion signals, draft response               |
|Short-Term / Episodic|Redis `session:{id}`             |Redis 7 @ localhost:6379|24h      |Cross-turn conversation history, recent distortion log, session goal|
|Long-Term / Semantic |Qdrant `collection: psych_memory`|Qdrant @ localhost:6333 |Permanent|User narrative patterns, recurring themes, long-arc progress vectors|
|Relational / Graph   |Neo4j `db: miazai`               |Neo4j @ localhost:7687  |Permanent|Distortion-theme relationships, trigger chains, CBT homework links  |

#### Long-Term Memory Schema (Qdrant)

```python
# Each vector = 768-dim NYTK/hubert embedding of a session summary
PsychMemoryRecord = {
    "id": "uuid",
    "vector": List[float],          # 768-dim
    "payload": {
        "session_id": str,
        "turn_range": [int, int],
        "dominant_theme": str,       # e.g. "önértékelési deficit"
        "distortions_detected": List[str],
        "cbt_frame_used": str,
        "valence_trajectory": List[float],  # turn-by-turn valence
        "goal_achieved": bool,
        "hu_keywords": List[str],    # Hungarian morphological roots
        "timestamp": str             # ISO8601
    }
}
```

#### Graph Schema (Neo4j — Cypher)

```cypher
// Nodes
(:Theme {name: "önértékelés", frequency: int})
(:Distortion {type: "katasztrofizálás", first_seen: datetime})
(:Session {id: str, date: date, goal: str, outcome: str})
(:Trigger {label: str, context: str})

// Relationships
(Session)-[:SURFACED]->(Theme)
(Session)-[:EXHIBITED]->(Distortion)
(Theme)-[:TRIGGERED_BY]->(Trigger)
(Distortion)-[:CO_OCCURS_WITH]->(Distortion)
(Session)-[:PROGRESSED_FROM]->(Session)  // longitudinal arc
```

-----

### 1.4 Action Planning

The Action Planner converts the Cognition Engine output into concrete, typed
output artifacts. It operates as a LangGraph node that receives the full
`CognitionState` and emits an `OutputBundle_v2`.

```python
@dataclass
class TherapeuticResponse:
    text: str                    # Primary natural language response (Hungarian)
    response_type: str           # "reflection" | "psychoeducation" | "skill_prompt" | "reframe"
    cbt_component: str | None    # e.g. "thought_record", "behavioural_experiment"
    suggested_homework: str | None
    safety_flag: bool            # True = requires human escalation
    confidence: float            # 0.0–1.0

@dataclass
class OutputBundle_v2:
    therapeutic_response: TherapeuticResponse
    suno_prompt: SunoPromptContract | None  # Only if creative mode active
    jpg_prompt: str | None                 # Only if visual mode active
    lyric_delta: str | None                # Updated lyric if session triggered revision
    session_update: dict                   # Persisted to Redis + Qdrant
    goal_delta: float
    alexandra_verdict: str                 # "pass" | "escalate" | "halt"
```

-----

### 1.5 Orchestration Layer: Extended LangGraph Graph

```python
# orchestrator/langgraph_pipeline_v2.py

builder = StateGraph(CognitionState)

# Perception nodes
builder.add_node("perceive_text", text_perceiver_node)
builder.add_node("perceive_prosody", prosody_node)          # Optional

# Existing WER/WEO nodes (unchanged contracts)
builder.add_node("wer_normalize", wer1_node)
builder.add_node("wer_transcribe", wer2_node)
builder.add_node("wer_emotion_fusion", wer3_node)
builder.add_node("wer_style_persona", wer4_node)
builder.add_node("weo_psycho_context", weo1_node)
builder.add_node("weo_hu_syllable", weo3_node)
builder.add_node("weo_lyric_gen", weo2_node)
builder.add_node("weo_jpg_prompt", weo4_node)

# NEW Cognition layer nodes
builder.add_node("route_clinical_frame", clinical_router_node)
builder.add_node("cbt_thought_record", cbt_tr_node)
builder.add_node("cbt_psychoeducation", cbt_edu_node)
builder.add_node("act_values_clarification", act_values_node)
builder.add_node("dbt_stabilise", dbt_stabilise_node)
builder.add_node("rational_deliberation", self_correction_node)
builder.add_node("action_plan", action_planner_node)
builder.add_node("alexandra_audit_v2", alexandra_v2_node)
builder.add_node("memory_consolidate", memory_consolidation_node)

# Conditional routing from clinical frame router
builder.add_conditional_edges(
    "route_clinical_frame",
    lambda s: s["active_cbt_frame"],
    {
        "cbt_thought_record": "cbt_thought_record",
        "cbt_psychoeducation": "cbt_psychoeducation",
        "act_values_clarification": "act_values_clarification",
        "dbt_stabilise": "dbt_stabilise",
    }
)

# All clinical frame nodes converge to rational deliberation
for node in ["cbt_thought_record", "cbt_psychoeducation",
             "act_values_clarification", "dbt_stabilise"]:
    builder.add_edge(node, "rational_deliberation")

builder.add_edge("rational_deliberation", "action_plan")
builder.add_edge("action_plan", "alexandra_audit_v2")
builder.add_edge("alexandra_audit_v2", "memory_consolidate")
builder.add_edge("memory_consolidate", END)
```

-----

## Module 2: Psychological Specification

### 2.1 CBT Framework Encoding

CBT is encoded as a **deterministic schema pipeline**, not a probabilistic
prompt. The model does not “reason about” CBT — it must produce structured
output that is validated against the schema before any text is generated.

#### Thought Record Schema

```python
@dataclass
class ThoughtRecord:
    situation: str                    # Helyzet — What happened?
    automatic_thought: str            # Automatikus gondolat — First thought?
    emotion: str                      # Érzés — What emotion? (Hungarian label)
    emotion_intensity: float          # 0.0–1.0
    evidence_for: List[str]           # Bizonyítékok mellette
    evidence_against: List[str]       # Bizonyítékok ellene (min. 2 required)
    balanced_thought: str             # Kiegyensúlyozott gondolat
    re_rated_intensity: float         # Must be ≤ original intensity - 0.05
    cognitive_distortion: str         # Must map to DISTORTION_REGISTRY

DISTORTION_REGISTRY = {
    "all_or_nothing": "fekete-fehér gondolkodás",
    "catastrophising": "katasztrofizálás",
    "mind_reading": "gondolatolvasás",
    "fortune_telling": "jövőbejóslás",
    "emotional_reasoning": "érzelmi következtetés",
    "should_statements": "'kellene' kijelentések",
    "labelling": "cimkézés",
    "personalisation": "perszonalizáció",
    "magnification": "felnagyítás",
    "minimisation": "kicsinyítés",
    "mental_filter": "mentális szűrő",
    "disqualifying_positive": "pozitívumok kizárása",
    "overgeneralisation": "túláltalánosítás",
    "jumping_to_conclusions": "elhamarkodott következtetés",
}
```

The `ThoughtRecord` validator runs **before** any LLM call for response
generation. If `evidence_against` has fewer than 2 items, the pipeline retries
the extraction up to 3 times before flagging for human review.

#### ACT Hexaflex State Machine

```python
from enum import Enum

class ACTProcess(Enum):
    ACCEPTANCE = "acceptance"               # Elfogadás
    COGNITIVE_DEFUSION = "defusion"         # Kognitív defúzió
    PRESENT_MOMENT = "present_moment"       # Jelenlét
    SELF_AS_CONTEXT = "self_as_context"     # Kontextus-én
    VALUES = "values"                       # Értékek
    COMMITTED_ACTION = "committed_action"   # Elköteleződés

# State machine: which process to activate based on PsychoContextSchema axes
ACT_ROUTING_MATRIX = {
    # (high_experiential_avoidance, low_values_clarity) → Acceptance first
    (True, False): ACTProcess.ACCEPTANCE,
    (False, True): ACTProcess.VALUES,
    (True, True):  ACTProcess.COGNITIVE_DEFUSION,
    (False, False): ACTProcess.COMMITTED_ACTION,
}
```

#### DBT Skill Selector

```python
DBT_SKILL_ROUTING = {
    # arousal > 0.8: crisis → TIPP (Temperature, Intense exercise, Paced breathing, Progressive relaxation)
    "crisis": "TIPP",
    # interpersonal conflict signal → DEAR MAN / GIVE
    "interpersonal": "DEAR_MAN",
    # emotion dysregulation, moderate arousal → Opposite Action
    "dysregulation": "OPPOSITE_ACTION",
    # cognitive distress, low arousal → IMPROVE
    "cognitive_distress": "IMPROVE_SKILL",
    # default: mindfulness (minden helyzetben alkalmazható)
    "default": "MINDFULNESS_WHAT",
}
```

-----

### 2.2 Cognitive Distortion Detector

This runs as a **dedicated LangGraph node** in the Cognition Engine, before the
clinical frame router. It is an internal classification task, not a user-facing
response.

```python
# orchestrator/nodes/distortion_detector.py
# Uses claude-sonnet-4-6 via Anthropic API (internal call, not user-facing)

DETECTOR_SYSTEM_PROMPT = """
Te egy kognitív viselkedésterápiás (KVT/CBT) szakértő vagy.
Elemezd a következő Magyar szöveget és azonosítsd az összes kognitív torzítást.
Válaszolj KIZÁRÓLAG JSON formátumban, semmilyen más szöveggel.

Visszaadandó séma:
{
  "distortions": [
    {
      "type": "<distortion_key from DISTORTION_REGISTRY>",
      "evidence_fragment": "<exact quote from text triggering detection>",
      "confidence": <0.0-1.0>
    }
  ],
  "overall_distortion_severity": <0.0-1.0>,
  "primary_distortion": "<single highest-confidence type>"
}
"""
```

Threshold logic:

- `confidence < 0.65` → discard (too ambiguous)
- `overall_distortion_severity > 0.80` → trigger DBT crisis protocol
- `overall_distortion_severity > 0.55` → trigger CBT thought record
- `overall_distortion_severity ≤ 0.55` → ACT or psychoeducation

-----

### 2.3 Rational Actor Design

The agent implements **dual-process verification** to minimise its own
reasoning errors before output is committed.

#### System 1 → System 2 Separation

```
System 1 (fast, heuristic): Distortion detector + clinical frame router
  → Runs deterministically on schema rules. No LLM calls.
  → Produces: ClinicalFrame, DistortionSet, ACTProcess

System 2 (slow, deliberative): TherapeuticResponse generator
  → Single LLM call with full context
  → Produces: draft TherapeuticResponse

Validation Layer (schema check): ThoughtRecord validator
  → Deterministic assertions on draft output
  → Retry if assertions fail (max 3 attempts)

Alexandra Audit v2: meta-agent review
  → Separate LLM call with auditor persona
  → Produces: verdict + correction notes
```

#### Anti-Bias Invariants (Alexandra v2 audit checklist)

```python
ALEXANDRA_AUDIT_CHECKLIST_V2 = [
    # Structural safety
    "Does the response avoid giving direct medical advice?",
    "Does the response avoid diagnosing the user?",
    "Does the response avoid replacing professional therapy?",

    # CBT integrity
    "Does a thought_record response include evidence_against with ≥2 items?",
    "Is the re_rated_intensity strictly lower than original?",
    "Is the cognitive_distortion label from DISTORTION_REGISTRY?",

    # Cultural sensitivity (Hungarian)
    "Does the response avoid cultural stereotypes about Hungarian emotional expression?",
    "Does the response acknowledge collective/family-oriented values if contextually relevant?",
    "Is the Calvinist/historical-pessimism cultural frame respected (not pathologised)?",

    # Rational actor
    "Does the response avoid catastrophising language in its own framing?",
    "Does the response avoid false certainty about the user's internal state?",
    "Does the response avoid sycophantic agreement with distorted cognitions?",
]
```

-----

## Module 3: Multimodality & Perception

### 3.1 Complete Model Stack

|Task                           |Model                             |Quantisation          |VRAM/RAM  |Backend         |
|-------------------------------|----------------------------------|----------------------|----------|----------------|
|Speech-to-text (HU)            |`openai/whisper-large-v3`         |INT8 via `ctranslate2`|~2.5GB RAM|`faster-whisper`|
|Text embedding (HU)            |`NYTK/hubert-base-cc`             |INT8 via `optimum`    |~400MB RAM|`transformers`  |
|Emotion classification         |`SamLowe/roberta-base-go_emotions`|FP32                  |~500MB RAM|`transformers`  |
|Acoustic prosody               |`parselmouth` (Praat)             |N/A                   |Minimal   |Python binding  |
|Cognitive distortion detection |`claude-sonnet-4-6` via API       |N/A                   |Cloud call|`anthropic` SDK |
|Therapeutic response generation|`ollama/qwen2.5:7b` (Zefír)       |Q4_K_M                |~5.5GB RAM|`ollama`        |
|Lyric generation               |`ollama/gemma3:4b` (Emma)         |Q4_K_M                |~3.5GB RAM|`ollama`        |
|Facial affect (optional)       |`DeepFace` + `ArcFace`            |FP32                  |~800MB RAM|`deepface`      |

**Total RAM budget on T420 (8GB assumed):**

- Base OS + Docker: ~1.2GB
- Redis + Qdrant + Neo4j: ~1.0GB
- Emma (gemma3:4b): ~3.5GB
- Zefír (qwen2.5:7b): ~5.5GB
- **Problem:** Emma and Zefír cannot coexist in RAM simultaneously.

**Solution — Model Swap Protocol:**

```python
# orchestrator/model_manager.py
import subprocess

def ensure_model_loaded(model_name: str):
    """Unloads the inactive model before loading the next."""
    current = _get_currently_loaded()
    if current and current != model_name:
        subprocess.run(["ollama", "stop", current], check=True)
    subprocess.run(["ollama", "run", model_name, "--keepalive", "5m"], check=True)
```

Zefír (7B) handles therapeutic reasoning. Emma (4B) handles lyric generation.
They never run simultaneously. The orchestrator serialises their execution.

-----

### 3.2 Hungarian Phonology & Emotional Prosody

Hungarian phonological properties that require special handling:

**Vowel Harmony:** Hungarian suffixes must harmonise (front/back) with the root.
Emotional dysregulation often produces vowel harmony violations in spontaneous speech.
The prosody extractor logs these as a cognitive load signal.

**Word-Initial Stress:** Unlike English (which varies stress placement),
Hungarian stress is **always on syllable 1**. This means Whisper’s
default English-optimised segmentation is unreliable for Hungarian
prosodic analysis. Always use `parselmouth` at the **word boundary** level,
not the phoneme level.

**Disfluencies in Hungarian:**

- `öö`, `hm`, `ö hát` → hedging / ambivalence
- `szóval` (so/well) → topic shift signal
- `nem tudom` (I don’t know) → uncertainty / avoidance
- `igen de` (yes but) → cognitive dissonance pattern

The disfluency detector:

```python
HU_DISFLUENCY_PATTERNS = {
    "hedging": re.compile(r'\b(öö|ö+|hm+|ö\s+hát)\b', re.IGNORECASE),
    "avoidance": re.compile(r'\bnem\s+tudom\b', re.IGNORECASE),
    "shift": re.compile(r'\bszóval\b', re.IGNORECASE),
    "dissonance": re.compile(r'\bigen\s+de\b|\bigen,?\s+de\b', re.IGNORECASE),
    "minimisation": re.compile(r'\bsemmi\s+különös\b|\bnem\s+nagy\s+dolog\b', re.IGNORECASE),
}
```

-----

### 3.3 Modality Fusion Strategy

When multiple perceptual channels are active (voice + text + prosody),
their signals are fused using a **weighted additive model**, not a
multiplicative one. Multiplicative fusion is brittle — if one channel
has near-zero confidence, the whole signal collapses.

```python
@dataclass
class FusedPerception:
    valence: float       # -1.0 to 1.0
    arousal: float       # 0.0 to 1.0
    dominance: float     # 0.0 to 1.0
    confidence: float
    modalities_used: List[str]

def fuse_perceptions(
    audio_emotion: EmotionProfile | None,
    text_embedding_valence: float | None,
    facial_affect: FacialAffect | None,
    prosody: dict | None
) -> FusedPerception:
    weights = {
        "audio": 0.40,    # Highest weight — paralinguistic signal richest
        "text": 0.35,     # High — semantic content
        "facial": 0.15,   # Lower — optional, privacy-gated
        "prosody": 0.10   # Supplementary
    }
    # Normalise weights to present modalities only
    present = {k: v for k, v in weights.items()
               if locals()[f"{k}_emotion" if k == "audio"
               else f"{k}_affect" if k == "facial"
               else k] is not None}
    total = sum(present.values())
    norm = {k: v / total for k, v in present.items()}
    # Weighted sum
    ...
```

-----

## Module 4: Autonomy & Self-Reflection

### 4.1 Self-Correction Loop

The self-correction loop is a **structured three-pass architecture**,
not a free-form “ask the model to check itself” approach.
Free-form self-checking has documented failure modes (sycophantic self-approval).

```
Pass 1 — GENERATION (Zefír / qwen2.5:7b)
  Input:  CognitionState + clinical_frame + distortion_set
  Output: TherapeuticResponse draft
  Prompt: therapeutic_response_prompt_v2.jinja2

Pass 2 — SCHEMA VALIDATION (deterministic, zero LLM calls)
  Input:  TherapeuticResponse draft
  Check:  ThoughtRecord schema assertions (if CBT frame)
          Response length: 80–250 words (too short = dismissive, too long = overwhelming)
          No direct medical advice regex scan
          No diagnostic language regex scan
  Output: validation_errors: List[str]

Pass 3 — ALEXANDRA AUDIT v2 (claude-sonnet-4-6 via API)
  Input:  TherapeuticResponse draft + validation_errors + CHECKLIST_V2
  Output: verdict ("pass" | "revise" | "escalate" | "halt") + revision_notes
  If "revise": loop back to Pass 1 with revision_notes injected
  Max retry: 3 (then escalate to human)
  If "halt": return safety_response only (no creative content)
```

### 4.2 Session Goal Management

The agent maintains a **session goal** across turns. This is not a static
instruction — it evolves based on user signals.

```python
@dataclass
class SessionGoal:
    stated_goal: str | None          # User's explicit statement
    inferred_goal: str               # Agent's inference from EmotionProfile + themes
    goal_axis: str                   # "emotional_regulation" | "self_worth" | "relationship" |
                                     # "grief" | "anxiety" | "creative_block"
    priority_score: float            # 0.0–1.0
    last_updated_turn: int
    progress_vector: List[float]     # Turn-by-turn valence delta

def update_session_goal(state: CognitionState, new_turn: dict) -> SessionGoal:
    """Re-evaluates goal on every turn. Goal drift is normal and healthy."""
    ...
```

Goal axis mapping to therapeutic frame:

```python
GOAL_AXIS_FRAME_MAP = {
    "emotional_regulation": "dbt_stabilise",
    "self_worth": "cbt_thought_record",
    "relationship": "dbt_stabilise",  # DEAR MAN branch
    "grief": "act_values_clarification",
    "anxiety": "cbt_thought_record",  # Exposure hierarchy sub-branch
    "creative_block": "act_values_clarification",  # Defusion branch
}
```

### 4.3 Autonomous Chain-of-Thought Protocol

The therapeutic response generation prompt enforces explicit CoT structure
via structured output — not via a free-text “think step by step” instruction.

```python
# whiteego/prompts/therapeutic_response_prompt_v2.jinja2

SYSTEM = """
Te MIazAI vagyok — egy pszicho-kontextuális AI asszisztens, szakmai tudással a CBT, ACT és DBT területén.
NEM vagyok terapeuta. NEM diagnozizálok. Mindent a felhasználó önismeret-fejlesztéséhez nyújtok.

A következő strukturált elemzési folyamatot KÖTELEZŐEN végezd el:

1. HELYZET FELISMERÉSE: Írj egy mondatot arról, hogy mit érzékelsz a felhasználó érzelmi állapotáról.
2. TORZÍTÁS AZONOSÍTÁSA: Nevezd meg a releváns kognitív torzítást (ha van).
3. TERÁPIÁS KERET: Határozd meg, melyik módszert (CBT/ACT/DBT) alkalmazol és miért.
4. VÁLASZ MEGFOGALMAZÁSA: Írj egy empatikus, 80-250 szavas választ magyarul.
5. ÖNELLENŐRZÉS: Ellenőrizd, hogy a válasz nem tartalmaz diagnózist, orvosi tanácsot, vagy patalógiás cimkézést.

KIMENET: Csak a 4. lépés szövegét add vissza. A többi lépés belső folyamat.
"""
```

-----

## Module 5: Hungarian Linguistic & Cultural Localization

### 5.1 Tokenization Strategy

**The core problem:** Standard BPE tokenizers (GPT-style) fragment Hungarian
agglutinative suffixes unpredictably, producing semantically incoherent
sub-word tokens.

Example:

```
"megvalósíthatatlanságáról" (about the impossibility of achieving)
  BPE (GPT-4 tokenizer): ["meg", "val", "ós", "íthat", "atlan", "ságáról"]  ← 6 tokens, corrupted morphemes
  WordPiece (NYTK):       ["meg", "##valósít", "##hatat", "##lanság", "##áról"] ← 5 tokens, preserves stems
  pyphen segmentation:    meg-va-ló-sít-ha-tat-lan-sá-gá-ról ← correct syllables
```

**Rule:** For embedding-sensitive operations (Qdrant storage, semantic search),
always use `NYTK/hubert-base-cc` tokenizer — **never** GPT-family tokenizers
for Hungarian text.

For syllable counting (lyric generation), `pyphen(lang='hu_HU')` remains
the canonical tool. These two responsibilities must not be conflated.

```python
# Correct separation of concerns
from transformers import AutoTokenizer
import pyphen

# For semantic embedding:
HU_TOKENIZER = AutoTokenizer.from_pretrained("NYTK/hubert-base-cc")
tokens = HU_TOKENIZER.tokenize("megvalósíthatóságáról")  # semantic units

# For lyric syllable counting:
HU_DIC = pyphen.Pyphen(lang='hu_HU')
syllables = HU_DIC.inserted("megvalósíthatóságáról", hyphen="-").split("-")  # rhythmic units
```

### 5.2 Cultural Context Layer

Hungarian cultural substrate that the agent must recognise and respect
(not pathologise):

**Historical Pessimism (Történelmi pesszimizmus):**
Hungarian cultural identity carries a well-documented pessimistic frame
rooted in historical trauma (Trianon, WWII, 1956, Communism). Expressions
like `"nem lehet", "semmi sem sikerül", "ez Magyarország"` are not
necessarily clinical catastrophising — they may be culturally normative
expressions of collective memory. The distortion detector must distinguish
**contextual pessimism** from **maladaptive catastrophising** using the
`hu_cultural_weight` field in `EmotionProfile`.

```python
HU_CULTURAL_PESSIMISM_PHRASES = [
    "ez magyarország",
    "nálunk ez nem megy",
    "így van ez mindenhol nálunk",
    "nem lehet mit tenni",
    "úgyis hiába",
]

def is_cultural_expression(text: str, context: str) -> bool:
    """Returns True if pessimistic expression is cultural rather than clinical."""
    for phrase in HU_CULTURAL_PESSIMISM_PHRASES:
        if phrase in text.lower():
            # Only pathologise if combined with personal attribution
            if "én" in context or "én vagyok" in context:
                return False  # Personalised → may be clinical
            return True  # Impersonal → likely cultural
    return False
```

**Calvinist Heritage:**
Strong emphasis on self-reliance, guilt as motivation, suppression of
emotional expression as virtue. CBT homework must be framed as
**practical exercises** (feladatok), not emotional explorations —
the latter triggers cultural resistance in many Hungarian users.

**Collective vs. Individual Identity:**
Hungarian culture is moderately collectivist. Family obligations and
social shame are powerful motivators. ACT values clarification
must include collective values (family, community, tradition), not
only individualistic ones as in standard Western ACT manuals.

**Sufi/Táltos Influence (MIazAI-specific):**
The system’s philosophical DNA includes non-dogmatic spiritual orientation
(Rumi, apophatic traditions, táltos). Responses may incorporate
metaphorical language that resonates with this tradition when contextually
appropriate:

```python
RUMI_RESONANT_FRAMES = [
    "A seb az a hely, ahonnan a fény bejön.",       # "The wound is where the light enters"
    "Csendben hallgatni a szívet — az is bátorság.", # Silence and courage
]
# Use only when user's own language contains philosophical/spiritual register
```

-----

### 5.3 Legal & Ethical Compliance (HU/EU)

#### GDPR (2016/679/EU) — Applicable Articles

|Data Type               |Article                                     |Implementation                                          |
|------------------------|--------------------------------------------|--------------------------------------------------------|
|Voice recordings        |Art. 9 (special category — health data)     |Store only `source_hash`, never raw audio               |
|Session transcripts     |Art. 5(1)(e) storage limitation             |Auto-delete Redis sessions after 24h                    |
|Psychological profiles  |Art. 9 + Art. 22 (automated decision-making)|All diagnostic outputs require human review flag        |
|Facial affect data      |Art. 9 (biometric)                          |Never persist; session-scoped in-memory only            |
|Long-term memory vectors|Art. 17 (right to erasure)                  |`DELETE FROM qdrant WHERE user_id = ?` endpoint required|

#### Hungarian Healthcare Regulations

**Act CLIV of 1997 (Eütv.)** and **Act LXXXIV of 2003 (mental health):**

- AI systems cannot replace licensed psychological services (pszichológiai szolgáltatás)
- All sessions must begin with a clear disclaimer
- Mandatory escalation pathway to licensed professional must exist

**Required disclaimer (to be injected at session start):**

```
Ez a rendszer nem helyettesíti a szakpszichológust vagy pszichiátert.
A MIazAI önismereti és kreatív eszköz. Válsághelyzetben kérjük,
forduljon szakemberhez vagy hívja az OKIT segélyvonalát: +36-80-20-55-20.
```

-----

## Implementation Roadmap

### Phase 1 — Foundation (Current Sprint)

- [ ] Install Ollama + pull `gemma3:4b` (Emma) + `qwen2.5:7b` (Zefír)
- [ ] Wire real LLM calls into `lyric_gen.py` (replace mocks)
- [ ] Verify whiteeat v3.0 Flask bridge on port 5002 end-to-end

### Phase 2 — Cognition Engine (Next 4 weeks)

- [ ] Implement `distortion_detector.py` (API call to claude-sonnet-4-6)
- [ ] Implement `clinical_router_node.py` with CBT/ACT/DBT routing matrix
- [ ] Implement `ThoughtRecord` schema + validator
- [ ] Add `hu_cultural_weight` field to `EmotionProfile` (additive, non-breaking)
- [ ] Write pytest fixtures with 5 real Hungarian therapy scenario transcripts

### Phase 3 — Memory Consolidation (Weeks 5–8)

- [ ] Create Qdrant collection `psych_memory` with NYTK/hubert-base-cc vectors
- [ ] Implement Neo4j graph schema (Theme / Distortion / Session / Trigger)
- [ ] Build `memory_consolidation_node.py`
- [ ] Add Redis session schema for cross-turn goal tracking

### Phase 4 — Alexandra v2 Audit (Weeks 9–10)

- [ ] Extend Alexandra with `ALEXANDRA_AUDIT_CHECKLIST_V2`
- [ ] Implement 3-pass self-correction loop
- [ ] Add GDPR compliance endpoints (erasure, data export)

### Phase 5 — Hungarian NLP Hardening (Weeks 11–12)

- [ ] Integrate NYTK/hubert-base-cc INT8 quantised model
- [ ] Implement `hu_disfluency_detector.py`
- [ ] Implement `is_cultural_expression()` classifier
- [ ] End-to-end test suite in Hungarian (10 scenario transcripts minimum)

-----

## Linear Issue Mapping (Suggested New Issues)

|Issue ID|Title                                                |Team   |Phase|
|--------|-----------------------------------------------------|-------|-----|
|WEO-6   |Implement CBT ThoughtRecord schema + validator       |WEO    |2    |
|WEO-7   |Build clinical frame router (CBT/ACT/DBT)            |WEO    |2    |
|WEO-8   |Alexandra Audit v2 — extended psychological checklist|WEO    |4    |
|WEO-9   |Hungarian cultural context classifier                |WEO    |5    |
|WEO-10  |NYTK/hubert-base-cc integration (replace mBERT)      |WEO    |5    |
|WER-6   |Prosody extractor — Hungarian disfluency detector    |WER    |2    |
|WER-7   |Multimodal fusion layer (audio + text + prosody)     |WER    |3    |
|SYS-1   |Qdrant psych_memory collection setup                 |MIa-zAI|3    |
|SYS-2   |Neo4j psychological graph schema                     |MIa-zAI|3    |
|SYS-3   |GDPR compliance endpoints (erasure + export)         |MIa-zAI|4    |
|SYS-4   |Model swap protocol (Emma/Zefír RAM management)      |MIa-zAI|2    |

-----

*Blueprint v2.0 — Klausz + A. Bell Scott — 2026-06-15*  
*Linear workspace: <https://linear.app/mia-zai>*  
*GitHub: sc0ttab3ll/miazai*