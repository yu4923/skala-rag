"""수집·계획·보고서·검토의 검증 가능한 입출력 계약."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from urllib.parse import urlsplit

ROLES = ('founder_insight', 'market_scout', 'tech_brief')


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class Evidence(Contract):
    evidence_id: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)
    source_title: str = Field(min_length=1)
    source: str = Field(min_length=1)
    source_type: Literal['pdf', 'web', 'input']
    page: int | None = Field(default=None, ge=1)
    provenance: Literal['company', 'independent', 'unknown'] = 'unknown'

    @model_validator(mode='after')
    def location(self):
        if self.source_type == 'web':
            url = urlsplit(self.source)
            if url.scheme not in ('http', 'https') or not url.hostname:
                raise ValueError('Invalid source URL')
        if self.source_type == 'pdf' and self.page is None:
            raise ValueError('PDF page required')
        return self


class Finding(Contract):
    topic: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    excerpts: dict[str, str] = Field(default_factory=dict)


class Collection(Contract):
    company_name: str
    agent: Literal['founder_insight', 'market_scout', 'tech_brief']
    findings: list[Finding]
    evidence: list[Evidence]
    missing_items: list[str] = Field(default_factory=list)
    queries_used: list[str]
    search_errors: list[str] = Field(default_factory=list)
    raw_documents: dict[str, dict] = Field(default_factory=dict)  # 근거 ID -> 원본 검색 데이터/질의

    @model_validator(mode='after')
    def grounding(self):
        ids = {e.evidence_id for e in self.evidence}
        if len(ids) != len(self.evidence) or any(set(f.evidence_ids) - ids for f in self.findings):
            raise ValueError('Unknown or duplicated evidence ID')
        if set(self.raw_documents) - ids:
            raise ValueError('raw 데이터가 수집 근거에 없는 ID를 참조합니다.')
        registry = {e.evidence_id: e for e in self.evidence}
        for finding in self.findings:
            if finding.excerpts and set(finding.excerpts) != set(finding.evidence_ids):
                raise ValueError('발췌와 연결 근거 ID 불일치')
            for eid, excerpt in finding.excerpts.items():
                if not excerpt.strip() or excerpt not in registry[eid].excerpt:
                    raise ValueError('검색 원문에 없는 발췌')
        return self


class ResearchTask(Contract):
    agent: Literal['founder_insight', 'market_scout', 'tech_brief']
    instruction: str = Field(min_length=1, max_length=2000)
    queries: list[str] = Field(min_length=1, max_length=3)


class SupervisorDecision(Contract):
    action: Literal['collect', 'write', 'review', 'finish', 'blocked']
    reason: str = Field(min_length=1)
    evidence_sufficient: bool
    collected_perspectives: list[Literal['founder_insight', 'market_scout', 'tech_brief']]
    missing_items: list[str]
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def plan(self):
        if self.action == 'collect' and not self.tasks:
            raise ValueError('Collection tasks required')
        if self.action != 'collect' and self.tasks:
            raise ValueError('Unexpected tasks')
        if len({t.agent for t in self.tasks}) != len(self.tasks):
            raise ValueError('Only one task per role in each fan-out')
        return self


class ReportDraft(Contract):
    summary: str = Field(min_length=1)
    business_overview: str = Field(min_length=1)
    market_analysis: str = Field(min_length=1)
    product_technology_and_team: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    used_evidence_ids: list[str]


class ReviewItem(Contract):
    passed: bool
    reason: str = Field(min_length=1)
    issues: list[str]


class ReportReview(Contract):
    groundedness: ReviewItem
    neutrality: ReviewItem
    bias_control: ReviewItem
    required_actions: list[str]

    @property
    def passed(self):
        return all(x.passed and not x.issues for x in (self.groundedness, self.neutrality, self.bias_control)) and not self.required_actions


class ResearchDecision(Contract):
    """자료 수집 및 보고서 작성/검토/종료 계획. 투자 평가는 하지 않는다."""
    action: Literal['collect', 'write', 'review', 'finish', 'blocked']
    reason: str = Field(min_length=1)
    evidence_sufficient: bool
    collected_perspectives: list[Literal['founder_insight', 'market_scout', 'tech_brief']]
    missing_items: list[str]
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=3)

    @model_validator(mode='after')
    def plan(self):
        if (self.action == 'collect') != bool(self.tasks):
            raise ValueError('collect에는 작업이 필요하고 종료에는 작업이 없어야 합니다.')
        if len({t.agent for t in self.tasks}) != len(self.tasks):
            raise ValueError('같은 관점의 중복 fan-out은 허용하지 않습니다.')
        if self.action == 'finish' and (not self.evidence_sufficient or self.missing_items):
            raise ValueError('미해결 필수 정보가 있으면 finish할 수 없습니다.')
        if self.action == 'blocked' and self.evidence_sufficient:
            raise ValueError('blocked와 충분성 판정 불일치')
        return self
