# -*- coding: utf-8 -*-
"""5차 발표 deck 생성 — 70B 전이헤드 confound 정정 + 세대축(Qwen3-8B) 교차검증.

내용은 IPI_Head_PoC_5th_script.md(S1~S13)를 그대로 옮긴 것. 디자인 토큰/헬퍼는
build_deck_4th.py와 동일 — import하면 그쪽 deck이 재빌드되므로 의도적으로 복제했다
(3rd/4th와 같은 이유).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

OUT = (r"C:\Users\Won\Desktop\대학교\AI Secure Lab\내부과제\atlas_poc"
       r"\docs\presentation\IPI_Head_Separation_PoC_5th [26-09-XX].pptx")

# ---------------------------------------------------------------- design tokens
BG        = RGBColor(0xFF, 0xFF, 0xFF)
INK       = RGBColor(0x16, 0x18, 0x1D)   # 제목
BODY      = RGBColor(0x56, 0x5C, 0x66)   # 본문
MUTED     = RGBColor(0x8B, 0x90, 0x99)   # 캡션
ORANGE    = RGBColor(0xC0, 0x6A, 0x1F)   # kicker / 강조
BLUE      = RGBColor(0x2F, 0x6F, 0xCE)   # 소제목
RED       = RGBColor(0xC1, 0x44, 0x2A)   # 공격 / 경고
CARD      = RGBColor(0xF2, 0xF2, 0xF4)
CARD_HL   = RGBColor(0xDB, 0xE9, 0xF8)   # 강조 카드
TH_FILL   = RGBColor(0xE9, 0xEA, 0xED)   # 표 헤더
ROW_ALT   = RGBColor(0xF7, 0xF7, 0xF9)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
LINE      = RGBColor(0xE3, 0xE5, 0xEA)

KR = "Malgun Gothic"
MONO = "Consolas"

M_L, M_W = 0.70, 11.93           # 좌측 마진 / 콘텐츠 폭
Y_KICK, Y_TITLE, Y_BODY = 0.45, 0.85, 1.85
Y_FOOT = 7.05

prs = Presentation()
prs.slide_width = Emu(12191695)
prs.slide_height = Emu(6858000)
BLANK = prs.slide_layouts[6]


# ---------------------------------------------------------------- helpers
def set_font(run, size, bold=False, color=BODY, font=KR):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = font
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        el = rPr.find(qn(tag))
        if el is None:
            el = rPr.makeelement(qn(tag), {})
            rPr.append(el)
        el.set("typeface", font)


def para(tf, spec, size=11.5, color=BODY, bold=False, font=KR,
         first=False, space_before=0, space_after=0, align=PP_ALIGN.LEFT,
         line_spacing=1.25):
    """spec: str 또는 [(text, {size,color,bold,font}), ...]"""
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    if space_before:
        p.space_before = Pt(space_before)
    if space_after:
        p.space_after = Pt(space_after)
    if line_spacing:
        p.line_spacing = line_spacing
    chunks = [(spec, {})] if isinstance(spec, str) else spec
    for text, opt in chunks:
        r = p.add_run()
        r.text = text
        set_font(r, opt.get("size", size), opt.get("bold", bold),
                 opt.get("color", color), opt.get("font", font))
    return p


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def card(slide, x, y, w, h, fill=CARD):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.09
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    sh.text_frame.text = ""
    return sh


def rect(slide, x, y, w, h, fill, alpha=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    if alpha is not None:
        _alpha(sh, alpha)
    return sh


def _alpha(shape, pct):
    clr = shape.fill.fore_color._xFill.find(qn("a:srgbClr"))
    el = clr.makeelement(qn("a:alpha"), {"val": str(int(pct * 1000))})
    clr.append(el)


def new_slide(kicker, title, title_size=26):
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 13.33, 7.5, BG)
    para(textbox(s, M_L, Y_KICK, M_W, 0.40), kicker,
         size=12, bold=True, color=ORANGE, font=MONO, first=True)
    para(textbox(s, M_L, Y_TITLE, M_W, 0.90), title,
         size=title_size, bold=True, color=INK, first=True, line_spacing=1.1)
    return s


def foot(slide, spec, y=Y_FOOT):
    para(textbox(slide, M_L, y, M_W, 0.40), spec,
         size=10.5, color=MUTED, first=True, line_spacing=1.2)


def table(slide, x, y, w, rows, col_w, row_h=0.36, head_h=0.38,
          sizes=None, aligns=None):
    """rows[0]=헤더. 각 셀은 str 또는 (text, {opt})리스트."""
    n_r, n_c = len(rows), len(col_w)
    h = head_h + row_h * (n_r - 1)
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    tbl = gf.table
    tbl.first_row = False
    tbl.horz_banding = False
    tbl._tbl.find(qn("a:tblPr")).set("bandRow", "0")

    total = sum(col_w)
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Emu(int(Inches(w) * cw / total))
    tbl.rows[0].height = Inches(head_h)
    for i in range(1, n_r):
        tbl.rows[i].height = Inches(row_h)

    for ri, row in enumerate(rows):
        for ci, cell_spec in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.045)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = TH_FILL
            else:
                cell.fill.fore_color.rgb = WHITE if ri % 2 else ROW_ALT
            tf = cell.text_frame
            tf.word_wrap = True
            al = (aligns or ["l"] * n_c)[ci]
            alignment = {"l": PP_ALIGN.LEFT, "r": PP_ALIGN.RIGHT,
                         "c": PP_ALIGN.CENTER}[al]
            sz = (sizes or [11] * n_c)[ci]
            if ri == 0:
                para(tf, cell_spec, size=sz, bold=True, color=MUTED,
                     first=True, align=alignment, line_spacing=1.0)
            else:
                para(tf, cell_spec, size=sz, color=BODY,
                     first=True, align=alignment, line_spacing=1.15)
    return gf


def card_head(tf, text, first=False):
    para(tf, text, size=13.5, bold=True, color=BLUE, first=first,
         space_before=0 if first else 9, line_spacing=1.15)


B = lambda t, c=INK: (t, {"bold": True, "color": c})     # noqa: E731
M = lambda t: (t, {"font": MONO, "size": 11})            # noqa: E731


# ================================================================ S1 타이틀
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.33, 7.5, BG)
rect(s, 0.70, 2.28, 1.10, 0.055, ORANGE)
para(textbox(s, 0.70, 2.50, 10.5, 0.40), "IPI DEFENSE · SCALE FIX & GENERATION CROSS-CHECK",
     size=13, bold=True, color=ORANGE, font=MONO, first=True)
tf = textbox(s, 0.70, 2.95, 11.7, 2.20)
para(tf, "Read Head, Control Head 분리 PoC", size=34, bold=True,
     color=INK, first=True, line_spacing=1.2)
para(tf, "— 스케일축 정정(Llama-70B) + 세대축(Qwen3-8B) 교차검증 (5차)", size=24, bold=True,
     color=INK, line_spacing=1.2)
para(textbox(s, 0.70, 4.75, 11.4, 1.00),
     "70B 전이헤드 confound 규명  ·  heldout 표본 확대(15→60)  ·  Qwen3-8B 세대 축",
     size=15, color=BODY, first=True)
para(textbox(s, 0.70, 6.50, 11.4, 0.40),
     "4차 발표 09-XX  ·  P12/P11 실험 09-08~09-12  ·  5차 발표 2026-09-XX       원종빈",
     size=11.5, color=MUTED, font=MONO, first=True)

# ================================================================ S2 서론
s = new_slide("01 · 서론", "4차 발표 요약")

card(s, M_L, Y_BODY, M_W, 1.55)
tf = textbox(s, 0.96, 2.04, 11.3, 1.30)
para(tf, [B("파서", INK), (" confound 아님 확정, ", {}), M("agentdojo_default"),
          (" 기본값 채택.", {})], size=12.5, first=True, line_spacing=1.3)
para(tf, [B("표본 확대(n=148)", INK), ("로 \"스케일업 반례\" 재확정 — ", {}),
          B("Qwen2.5-7B/Llama-8B는 slack knockout 전량 억제, Qwen2.5-32B만 절반 이상 "
            "persist", RED), (" (8~9→5, backfire 1).", {})],
     size=12.5, space_before=9, line_spacing=1.3)
para(tf, [B("양자화", INK), (": ", {}), M("fp4"), (" 아티팩트 확정(", {}), M("nf4+dq"),
          ("로 해소). 32B bf16(양자화 배제)에서도 slack knockout 불완전 — "
           "\"32B 스케일 효과\"로 읽었음.", {})],
     size=12.5, space_before=9, line_spacing=1.3)

card(s, M_L, 4.95, M_W, 1.30, CARD_HL)
tf = textbox(s, 1.00, 5.13, 11.33, 1.00)
para(tf, [B("다음 피드백(A)", ORANGE),
          (" — \"다른 모델 교차검증: Qwen3-8B · Llama-70B\" — \"다른 모델·다른 스케일에서도 "
           "같은 패턴인가?\"", {})], size=13, first=True, line_spacing=1.3)

# ================================================================ S3 이번 사이클 배경
s = new_slide("02 · 배경", "Llama-70B 1차 결과가 \"스케일업 반례\"를 더 강하게 재현했다")

para(textbox(s, M_L, Y_BODY - 0.05, M_W, 0.35),
     "results/2026-09-08_p12_llama70b/ — Llama-3.1-70B, 8B에서 찾은 헤드를 그대로 전이해 "
     "slack held-out 35쌍 knockout", size=10.5, color=MUTED, first=True)

table(s, M_L, 2.30, M_W,
      [["모델", "k0 성공", "knockout 효과", "backfire", "net"],
       ["Qwen2.5-7B / Llama-8B bf16", "6", [B("6 → 0  전량 억제", BLUE)], "0", "0"],
       ["Qwen2.5-32B bf16 (자체 헤드)", "8~9", [B("절반 이상 persist", RED)], "1", "5"],
       [[B("Llama-3.1-70B nf4dq (8B 헤드 전이)", RED)], [B("5", RED)],
        [B("1 억제 / 4 persist", RED)], [B("1", RED)], [B("5 (순 억제 0)", RED)]]],
      col_w=[4.3, 1.5, 3.0, 1.5, 2.03], row_h=0.50, head_h=0.38,
      sizes=[10.5, 10.5, 10, 10.5, 10.5], aligns=["l", "c", "c", "c", "c"])

card(s, M_L, 4.55, M_W, 2.15, CARD_HL)
tf = textbox(s, 1.00, 4.73, 11.33, 1.85)
para(tf, [("언뜻 \"대형 모델일수록 knockout에 저항한다\"는 결론을 강화하는 것처럼 보였다 — "
           "70B가 세 지점 중 가장 극단.", {})], size=12, first=True, line_spacing=1.3)
para(tf, [B("그런데 이 70B 행은 70B 자신의 헤드가 아니라 8B에서 찾은 헤드를 그대로 썼다", RED),
          (" (70B 자체 Track A는 이 시점까지 ", {}), M("--device_map auto"),
          ("의 backward OOM으로 실행 불가) → \"70B가 저항하는가\" vs \"8B의 헤드를 껐을 "
           "뿐인가\"를 구분할 수 없는 상태로 다음 사이클에 들어감.", {})],
     size=12, space_before=8, line_spacing=1.3)

# ================================================================ S4 실험① 배경
s = new_slide("03 · 실험 ①  70B 헤드 탐색", "막던 것은 하드웨어가 아니라 배선이었다")

pts = [
    [("기존 결론(\"70B AttnLRP backward는 이 하드웨어에서 불가\")은 ", {}), M("--device_map auto"),
     (" 경로 한정이었다 — 타이트한 ", {}), M("--max_memory"),
     ("면 CPU/disk 분산 에러, 느슨하면 레이어가 한 GPU에 몰려 backward OOM. 그 사이에 "
      "열리는 창이 없었다.", {})],
    [B("해결: ", BLUE), M('--device_map_plan "0:32,1:30,2:18"'),
     (" (수동 device_map) 신설 — embed/norm/lm_head는 첫 GPU에 몰아두고(backward가 양 끝에서 "
      "시작·수렴), 레이어는 순서대로 분배.", {})],
    [B("3-GPU 배선(A6000 48G root / Blackwell 32G / 4090 24G)으로 ", INK),
     B("130/130 성공, 0 oom, 0 nan.", BLUE)],
    [("부수 발견: Blackwell nf4 커널이 32B greedy 출력을 안 건드리는 데 이어 70B backward도 "
      "정상 — \"4bit=A6000 고정\" 관례가 계속 완화됨.", {})],
]
tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 4.4)
for i, sp in enumerate(pts):
    para(tf, [("•  ", {"color": ORANGE, "bold": True})] + sp,
         size=13, first=(i == 0), space_before=0 if i == 0 else 16, line_spacing=1.3)

# ================================================================ S5 실험① 결과
s = new_slide("03 · 실험 ①  70B 헤드 탐색", "70B 자체 헤드는 8B와 같은 상대 깊이에서 나온다")

para(textbox(s, M_L, Y_BODY - 0.05, M_W, 0.35),
     "results/2026-09-08_p12_llama70b/heads_agentdojo.json", size=10.5, color=MUTED,
     first=True, font=MONO)

table(s, M_L, 2.30, M_W,
      [["항목", "Llama-8B", "Llama-70B"],
       ["head 위치", "layer 11–22 / 32", "layer 26–44 / 80"],
       ["상대 깊이", "≈ 38–47%", "≈ 35–44%"],
       ["재현성", "—", "smoke(16) ∩ full(130) = 17/20"]],
      col_w=[2.6, 4.6, 4.73], row_h=0.55, head_h=0.40,
      sizes=[11, 11, 11], aligns=["l", "c", "c"])

card(s, M_L, 4.55, M_W, 1.35, CARD_HL)
tf = textbox(s, 1.00, 4.73, 11.33, 1.05)
para(tf, [B("같은 상대 깊이 대역", BLUE),
          (" — 스케일이 8배 커져도 \"주입 신호를 담당하는 레이어\"의 상대 위치는 유지된다는 "
           "신호.", {})], size=13, first=True, line_spacing=1.3)

# ================================================================ S6 실험② important_instructions
s = new_slide("04 · 실험 ②  자체 vs 전이", "\"70B 자체 헤드\" vs \"8B 전이 헤드\" — important_instructions")

para(textbox(s, M_L, Y_BODY - 0.08, M_W, 0.55),
     "같은 모델(Llama-3.1-70B nf4dq)·같은 slack 105쌍(--eval_split all)에서 knockout에 쓰는 "
     "헤드 집합만 8B 전이 ↔ 70B 자체로 교체", size=10, color=MUTED, first=True, line_spacing=1.2)

table(s, M_L, 2.35, M_W,
      [["knockout 헤드", "k0 sec", "kN sec", "억제/backfire", "net", "kN util", "parse_ok"],
       ["8B 전이 (S3과 동일 헤드)", "0.276", "0.257", "4 / 2", [B("−2 (효과 없음)", RED)],
        "0.190", "0.782"],
       [[B("70B 자체", BLUE)], "0.276", [B("0.181", BLUE)], [B("13 / 3")],
        [B("−10 (ASR 34%↓)", BLUE)], "0.190", "0.783"],
       ["70B 자체 (누수 없는 heldout 15쌍)", "0.533", [B("0.200", BLUE)], "5 / 0",
        "−5", [B("0.333 (무손상)", BLUE)], "0.853"]],
      col_w=[3.15, 1.15, 1.15, 1.55, 2.0, 1.85, 1.13], row_h=0.48, head_h=0.42,
      sizes=[9.5, 9.5, 9.5, 9.5, 9.5, 9.5, 9.5], aligns=["l", "r", "r", "c", "c", "r", "r"])

card(s, M_L, 4.55, M_W, 2.15, CARD_HL)
tf = textbox(s, 1.00, 4.73, 11.33, 1.85)
para(tf, [("parse_ok율 두 조건이 동일(0.78) → security 하락이 \"tool-call을 못 뱉어서\"가 "
           "아님.", {})], size=11.5, first=True, line_spacing=1.3)
para(tf, [("utility는 오히려 소폭 상승 → 모델 손상이 아니라 ", {}),
          B("injection-following만 선택적으로 억제.", BLUE)],
     size=11.5, space_before=6, line_spacing=1.3)
para(tf, [B("누수 없는 heldout이 누수 있는 all105보다 더 강한 효과", INK),
          (" → all105의 결과가 누수로 부풀려진 게 아님.", {})],
     size=11.5, space_before=6, line_spacing=1.3)

# ================================================================ S7 실험② tool_knowledge
s = new_slide("04 · 실험 ②  자체 vs 전이", "2번째 공격 축(tool_knowledge)으로 교차 확인")

para(textbox(s, M_L, Y_BODY - 0.08, M_W, 0.35),
     "같은 구성, --attack tool_knowledge (32B에서 baseline ASR이 ~2배였던 더 강한 공격)",
     size=10, color=MUTED, first=True)

table(s, M_L, 2.30, M_W,
      [["knockout 헤드", "k0 sec", "kN sec", "억제/backfire", "net", "kN util"],
       ["8B 전이", "0.402", "0.392", "3 / 2", [B("−1 (효과 없음)", RED)], "0.186"],
       [[B("70B 자체", BLUE)], "0.398", [B("0.223", BLUE)], [B("18 / 0")],
        [B("−18 (ASR 44%↓)", BLUE)], "0.204"],
       ["70B 자체 (heldout 15쌍)", "0.733", [B("0.400", BLUE)], "5 / 0", "−5",
        [B("0.333 (무손상)", BLUE)]]],
      col_w=[3.15, 1.35, 1.35, 1.75, 2.3, 2.03], row_h=0.48, head_h=0.42,
      sizes=[10, 10, 10, 9.5, 10, 10], aligns=["l", "r", "r", "c", "c", "r"])

card(s, M_L, 4.35, M_W, 1.75, CARD_HL)
tf = textbox(s, 1.00, 4.53, 11.33, 1.45)
para(tf, [B("공격을 바꿔도 같은 패턴", BLUE),
          (" — 70B 자체 헤드는 backfire ", {}), B("0건", BLUE), ("(더 깨끗함).", {})],
     size=13, first=True, line_spacing=1.3)
para(tf, [("→ 헤드가 특정 공격 문구가 아니라 ", {}), B("일반 injection 신호", INK),
          ("를 담는다는 근거가 공격-독립적으로 확정.", {})],
     size=13, space_before=8, line_spacing=1.3)

# ================================================================ S8 판정
s = new_slide("05 · 판정", "\"대형 모델이 저항한다\"가 아니라 \"전이 헤드는 스케일이 안 된다\"")

tf = textbox(s, M_L, Y_BODY + 0.10, M_W, 4.5)
para(tf, [B("정정: ", RED), ("4차 발표 시점의 \"스케일업 반례\" 서술 중 ", {}),
          B("Llama 계열 부분", RED),
          ("은 70B의 저항이 아니라 8B 헤드가 70B에 전이되지 않았기 때문이었다. 70B 자신의 "
           "헤드로 끄면 knockout이 정상 작동한다(ASR 34~44%↓, utility 손상 0, 공격 2종 "
           "모두 재현).", {})],
     size=13, first=True, line_spacing=1.35)
para(tf, [B("Qwen 계열은 사정이 다르다", ORANGE),
          (" — Qwen2.5-32B의 \"8~9→5\" 결과는 처음부터 32B 자체 헤드를 썼다(전이 아님, "
           "feedback-2026-08-31.md:289). 남은 caveat은 그 헤드가 fp4로 탐색돼 nf4dq eval과 "
           "양자화가 안 맞는다는 것뿐 — 별개 축(todo.md §4-1, 미실행).", {})],
     size=13, space_before=16, line_spacing=1.35)

card(s, M_L, 5.25, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 5.43, 11.33, 1.25)
para(tf, [B("헤드 분리 가설 자체는 두 스케일(8B/70B)에서 성립", BLUE),
          (" — 바뀐 건 \"대형=저항\"이 아니라 \"knockout은 모델별 자체 헤드 탐색이 필요, "
           "전이 헤드는 스케일이 안 된다\"는 방법론적 교훈.", {})],
     size=13, first=True, line_spacing=1.35)

# ================================================================ S9 실험③ 배경
s = new_slide("06 · 실험 ③  heldout 확대", "heldout 표본이 너무 얇았다")

tf = textbox(s, M_L, Y_BODY + 0.20, M_W, 3.5)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("§S6~S7의 \"누수 없는 heldout\"은 ", {}), B("slack 15쌍뿐", RED),
          ("이었다 — head_n=200(8B 탐색과 동일 값)이 slack user_task 대부분을 head 선정에 "
           "써버렸기 때문.", {})],
     size=14, first=True, line_spacing=1.35)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("15쌍은 판정을 뒤집을 만큼 얇지는 않지만(누수 있는 all105보다 오히려 강한 효과), "
           "표본을 늘리면 신뢰도가 올라간다 → ", {}), B("head_n을 낮춰 재탐색하면 heldout "
           "user_task가 늘어난다", BLUE),
          (" (대가: 탐색 예시 감소로 헤드 노이즈 소폭 증가).", {})],
     size=14, space_before=18, line_spacing=1.35)

# ================================================================ S10 실험③ 결과
s = new_slide("06 · 실험 ③  heldout 확대", "head_n 80 재탐색으로 heldout 15쌍 → 60쌍", title_size=24)

table(s, M_L, Y_BODY, 5.6,
      [["head_n", "slack head 그룹", "실제 eval 후보"],
       ["200 (기존)", "18/20 소진", "15"],
       [[B("80 (신규)", BLUE)], "7/20", [B("70 (60 평가)", BLUE)]]],
      col_w=[1.3, 1.9, 2.4], row_h=0.42, head_h=0.36,
      sizes=[10, 9.5, 9.5], aligns=["l", "c", "c"])

tf = textbox(s, 6.55, Y_BODY, 5.9, 1.55)
para(tf, [B("재현성: jaccard = 0.82", BLUE),
          (" (20개 중 18개 head_n=200과 일치) — 탐색 예시 54개(vs 149)로 줄여도 헤드가 거의 "
           "그대로 재현됨. 탐색 풀 축소 우려는 기우.", {})],
     size=10.5, first=True, line_spacing=1.3)

table(s, M_L, 3.25, M_W,
      [["공격", "표본", "k0 sec", "kN sec", "억제/bf/pers", "net", "kN util", "parse_ok"],
       ["important_instructions", "15→60", "0.250 (15)", [B("0.117", BLUE)], "9/1/6",
        [B("+8 (53%↓)", BLUE)], [B("0.150 (↓)", RED)], "0.733"],
       ["tool_knowledge", "15→60", "0.350 (21)", [B("0.217", BLUE)], "8/0/13",
        [B("+8 (38%↓)", BLUE)], [B("0.183 (↑)", BLUE)], "0.747"]],
      col_w=[3.0, 1.15, 1.55, 1.15, 1.55, 1.75, 1.55, 1.28], row_h=0.50, head_h=0.42,
      sizes=[9.5, 9, 9, 9, 9, 9, 9, 9], aligns=["l", "c", "r", "r", "c", "c", "r", "r"])

card(s, M_L, 5.35, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 5.53, 11.33, 1.25)
para(tf, [("표본 4배(15→60)로도 두 공격 다 net 방어적 유지. ", {}),
          B("important_instructions만 utility 첫 손상", RED),
          (" — 원인 특정됨: 손실 3건 전부 suppressed case에서 task 수행 자체가 같이 무너진 "
           "것(무작위 형식 손상 아님). tool_knowledge는 backfire 0 3표본 연속.", {})],
     size=11, first=True, line_spacing=1.3)

# ================================================================ S11 실험④ 배경
s = new_slide("07 · 실험 ④  세대 축", "Qwen3-8B — lxt \"첫 토큰 쏠림\" 경고 선(先)진단")

tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 3.6)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("피드백(A) 두 번째 축: ", {}), B("Qwen3-8B", INK),
          (" — 아키텍처는 그대로, 세대만 바뀐 대조군.", {})],
     size=13.5, first=True, line_spacing=1.35)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("lxt README가 Qwen3에서 \"attribution이 첫 토큰으로 쏠린다\"고 경고 — 우리 방법은 "
           "relevance를 D", {}), (("inj", {"size": 9})), (" span에 group-sum하므로, 질량이 "
           "position 0에 흡수되면 head 점수가 계통적으로 눌릴 위험. ", {}),
          B("배선(6줄)보다 진단이 먼저.", RED)],
     size=13.5, space_before=16, line_spacing=1.35)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("판단 기준: position 0 비중이 0이 아닌 것 자체는 문제가 아니다(causal LM의 흔한 "
           "attention sink) — 같은 프롬프트로 ", {}), B("qwen2 대조군과 나란히 돌려 상대적으로 "
           "얼마나 더 쏠리는지가 기준.", BLUE)],
     size=13.5, space_before=16, line_spacing=1.35)

# ================================================================ S12 실험④ 결과
s = new_slide("07 · 실험 ④  세대 축", "쏠림은 실재하지만 D_inj 신호를 지우지는 않는다", title_size=24)

table(s, M_L, Y_BODY, 6.5,
      [["family", "position 0 비중", "data_inj 비중(22tok)"],
       ["qwen2 (대조군)", "0.49%", "37.71%"],
       [[B("qwen3", BLUE)], [B("16.71%", RED)], "32.78%"]],
      col_w=[1.9, 2.3, 2.3], row_h=0.42, head_h=0.36,
      sizes=[9.5, 9.5, 9.5], aligns=["l", "c", "c"])

tf = textbox(s, 7.45, Y_BODY, 5.0, 1.55)
para(tf, [B("쏠림은 qwen2 대비 ~34배로 실재", RED),
          (". 그러나 data_inj span 비중은 qwen2와 비슷하게 유지 — head 탐색은 span 단위 "
           "group-sum이라 position 0은 애초에 그 합산에 안 들어감. ", {}),
          B("진단 통과.", BLUE)],
     size=9.5, first=True, line_spacing=1.25)

table(s, M_L, 3.15, M_W,
      [["항목", "Track A (탐색)", "Track B — important_instructions", "Track B — tool_knowledge"],
       ["n / oom", "103/143 (40 oom)", "8/35 → 0/35 (8/0/0)", "6/35 → 2/35 (5/1/1)"],
       ["head 위치", "layer 18–29/36", "—", "—"],
       ["net / kN util", "—", [B("+8 전량 억제", BLUE)] + [(" · 0.429(↑)", {})],
        [B("+4 (66%↓)", BLUE)] + [(" · 0.400(무손상)", {})]]],
      col_w=[2.0, 3.0, 3.5, 3.43], row_h=0.48, head_h=0.42,
      sizes=[9.5, 9, 9, 9], aligns=["l", "c", "c", "c"])

card(s, M_L, 5.35, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 5.53, 11.33, 1.25)
para(tf, [B("8B급(Llama-8B/Qwen2.5-7B)과 동일한 패턴이 세대 축에서도 재현", BLUE),
          (" — important_instructions 전량 억제·backfire 0, tool_knowledge는 산발적 "
           "backfire 1건 있으나 net은 방어적. layer 0 헤드 2개(§S12 쏠림)가 섞여 있었지만 "
           "knockout 효과를 해치지 않음.", {})],
     size=11, first=True, line_spacing=1.3)

# ================================================================ S13 결과 요약 & 다음 단계
s = new_slide("08 · 마무리", "결과 요약 & 다음 단계")

card(s, M_L, Y_BODY, M_W, 3.15, CARD_HL)
tf = textbox(s, 1.00, Y_BODY + 0.18, 11.33, 2.90)
para(tf, "이번 사이클 확정된 것", size=13.5, bold=True, color=BLUE, first=True)
for sp in [
    [B("Llama-70B Track A", INK), (" (수동 device_map)로 자체 헤드 탐색 가능 확인.", {})],
    [B("\"스케일업 반례\"의 Llama 부분 = 전이헤드 confound", RED),
     ("였음을 2개 공격 축(important_instructions/tool_knowledge)에서 공격-독립적으로 확정, "
      "backfire 0(tool_knowledge).", {})],
    [B("Qwen2.5-32B 쪽은 애초에 전이 문제가 아니었음", INK),
     (" (자체 헤드, fp4↔nf4dq 양자화 불일치만 남음) — 4차 발표 서술의 착오를 정정.", {})],
    [B("Qwen3-8B lxt \"첫 토큰 쏠림\" 경고", INK), (" — 실재하나 D_inj 신호를 지우지 않음, "
        "진단 통과.", {})],
    [B("70B heldout 표본 확대(15→60쌍)", BLUE),
     ("에서도 두 공격축 모두 net 억제 유지. important_instructions에서 utility 첫 손상 "
      "발견(원인 특정), tool_knowledge는 backfire 0·무손상.", {})],
    [B("Qwen3-8B", BLUE), (": important_instructions 8/8 전량 억제·backfire 0, "
        "tool_knowledge net+4(66%↓)·backfire 1 — 8B급 패턴이 세대 축에서도 재현.", {})],
]:
    para(tf, [("·  ", {"color": BLUE, "bold": True})] + sp, size=10, space_before=5, line_spacing=1.18)

card(s, M_L, 5.20, M_W, 1.70)
tf = textbox(s, 1.00, 5.38, 11.33, 1.40)
para(tf, "다음 사이클 후보", size=13.5, bold=True, color=ORANGE, first=True)
for sp in [
    [B("Qwen2.5-32B 자체 헤드 nf4dq 재탐색", INK), (" (todo.md §4-1) — fp4↔nf4dq 탐색 "
        "일관성, Qwen 쪽 \"스케일업 반례\"가 여전히 유효한지.", {})],
    [B("70B k-sweep", INK), (" (topk 20→40→60) — 자체 헤드가 통함을 확인했으니 억제력 곡선.", {})],
    [B("70B utility 손상 원인 정밀 확인", INK), (" — suppressed 9건 중 3건에서만 나타난 이유.", {})],
]:
    para(tf, [("·  ", {"color": ORANGE, "bold": True})] + sp, size=10.5, space_before=5, line_spacing=1.2)

# ----------------------------------------------------------------
prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
