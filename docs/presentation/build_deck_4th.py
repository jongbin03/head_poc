# -*- coding: utf-8 -*-
"""4차 발표 deck 생성 — 파서 정합성 검증 + 양자화 아티팩트 규명.

내용은 IPI_Head_PoC_4th_script.md(S1~S15)를 그대로 옮긴 것. 디자인 토큰/헬퍼는
build_deck_3rd.py와 동일 — import하면 그쪽 deck이 재빌드되므로 의도적으로 복제했다
(build_deck_addendum_0819.py / build_deck_3rd.py와 같은 이유).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

OUT = (r"C:\Users\Won\Desktop\대학교\AI Secure Lab\내부과제\atlas_poc"
       r"\docs\presentation\IPI_Head_Separation_PoC_4th [26-09-XX].pptx")

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
para(textbox(s, 0.70, 2.50, 10.5, 0.40), "IPI DEFENSE · PARSER & QUANTIZATION AUDIT",
     size=13, bold=True, color=ORANGE, font=MONO, first=True)
tf = textbox(s, 0.70, 2.95, 11.7, 2.20)
para(tf, "Read Head, Control Head 분리 PoC", size=34, bold=True,
     color=INK, first=True, line_spacing=1.2)
para(tf, "— 파서 정합성 검증 + 양자화 아티팩트 규명 (4차)", size=26, bold=True,
     color=INK, line_spacing=1.2)
para(textbox(s, 0.70, 4.75, 11.4, 1.00),
     "파서 A/B  ·  표본 확대 (n=148)  ·  양자화 규명 (fp4 → nf4 / bf16)",
     size=15, color=BODY, first=True)
para(textbox(s, 0.70, 6.50, 11.4, 0.40),
     "3차 발표 08-26  ·  피드백 대응 실험 08-31~09-06  ·  4차 발표 2026-09-XX       원종빈",
     size=11.5, color=MUTED, font=MONO, first=True)

# ================================================================ S2 서론 ①
s = new_slide("01 · 서론", "3차 발표까지의 결론")

card(s, M_L, Y_BODY, M_W, 0.92)
tf = textbox(s, 0.96, 2.04, 11.3, 0.70)
para(tf, [("knockout = 찾은 head들의 ", {}), B("주입 명령 방향(D", INK), ("inj", {"size": 9}),
          (") attention edge만 차단.  ", {}),
          B("세 축 모두에서 \"공격 억제 + 정상 기능 보존\"이 재현됨.", BLUE)],
     size=12.5, first=True)

para(textbox(s, M_L, 2.95, M_W, 0.40),
     [("확장 축:  ", {}), B("① 스케일", ORANGE), (" Qwen2.5 7B→32B    ", {}),
      B("② 패밀리", ORANGE), (" Qwen2→Llama-3.1    ", {}),
      B("③ 공격 강도", ORANGE), (" 문구 강화", {})],
     size=11.5, color=BODY, first=True)

table(s, M_L, 3.45, M_W,
      [["모델", "공격", "n", "ASR  k=0", "ASR  k=N", "상대 감소"],
       ["Qwen2.5-32B", "important_instructions", "57", "8.8%", [B("3.5%", RED)], "60%"],
       ["Qwen2.5-32B", [B("tool_knowledge", ORANGE)], "55", "14.5%", [B("9.1%", RED)], "37.5%"],
       ["Llama-3.1-8B", "important_instructions", "44", "4.5%", [B("0%", BLUE)], "100%"],
       ["Llama-3.1-8B", [B("tool_knowledge", ORANGE)], "44", "4.5%", [B("0%", BLUE)], "100%"]],
      col_w=[2.3, 3.2, 0.7, 1.8, 1.8, 2.13], row_h=0.44, head_h=0.38,
      sizes=[11, 10, 11, 10.5, 10.5, 10.5], aligns=["l", "l", "c", "r", "r", "r"])

card(s, M_L, 5.72, M_W, 1.05, CARD_HL)
tf = textbox(s, 1.00, 5.90, 11.33, 0.75)
para(tf, [B("3차가 스스로 단 한계 — ", INK),
          ("① 표본이 작다(suite마다 무작위 15쌍)   ② 성공 사례가 거의 ", {}),
          B("slack에 몰림", RED), ("   ③ tool-call 파서가 우리 커스텀 구현", {})],
     size=11.5, first=True)

# ================================================================ S3 서론 ②
s = new_slide("01 · 서론", "이번 사이클(P16)에 한 것")

card(s, M_L, Y_BODY, M_W, 0.85)
tf = textbox(s, 0.96, 2.02, 11.3, 0.65)
para(tf, [B("출발점: 교수님 피드백 4번", ORANGE),
          (" — \"tool-call 파서를 AgentDojo 기본값으로 바꿔서 실험해봐라\"", {})],
     size=13, first=True)

table(s, M_L, 2.95, M_W,
      [["#", "실험 라인", "무엇을 봤나"],
       ["①", [B("파서 A/B")],
        "커스텀 → agentdojo_default로 바꿔도 suite별 비대칭(banking/workspace 저조)이 그대로인가?"],
       ["②", [B("표본 최대 확대  (n=148)")],
        "slack knockout 실패가 표본이 작아서인가 실재하는가? + 성공한 공격의 성격 분석"],
       ["③", [B("양자화 규명")],
        "slack 실패가 나온 유일한 32B 실행이 4bit — 원인이 (1) 4bit 양자화인가 (2) 32B 스케일인가?"]],
      col_w=[0.5, 3.0, 8.43], row_h=0.74, head_h=0.38, aligns=["c", "l", "l"],
      sizes=[11, 11.5, 10.5])

card(s, M_L, 5.70, M_W, 1.05, CARD_HL)
tf = textbox(s, 1.00, 5.88, 11.33, 0.75)
para(tf, [B("결론 미리보기 — ", INK),
          ("① 파서는 confound 아님   ② 스케일업 반례 재확정, 단 slack은 특이 케이스   "
           "③ 원인 두 갈래(", {}), M("fp4"), (" 양자화 아티팩트 + 약한 32B 스케일 효과)로 분리", {})],
     size=11.5, first=True)

# ================================================================ S4 실험① 배경
s = new_slide("02 · 실험 ①  파서", "왜 파서를 의심했나")

pts = [
    [("Track B(", {}), M("run_agentdojo_eval.py"),
     (")가 쓰던 tool-call 프롬프트/파서는 AgentDojo 자체 기본값이 아니라 ", {}),
     B("우리가 만든 커스텀 파서", RED), ("였음", {})],
    [("이유: 초기 실험에서 1.5B 모델이 AgentDojo 기본 포맷을 안 따라 대체했던 이력", {})],
    [B("우려", INK), (": 지금까지 관찰한 \"banking/workspace suite 저조\"가 파서 아티팩트일 수 있지 않은가?", {})],
    [B("대응", BLUE), (": ", {}), M("--tool_call_format agentdojo_default"),
     (" 옵션 추가(AgentDojo 자체 파서 재사용) 후 같은 조건에서 A/B 비교", {})],
]
tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 4.4)
for i, sp in enumerate(pts):
    para(tf, [("•  ", {"color": ORANGE, "bold": True})] + sp,
         size=13, first=(i == 0), space_before=0 if i == 0 else 14, line_spacing=1.3)

# ================================================================ S5 실험① 결과
s = new_slide("02 · 실험 ①  파서", "Custom vs AgentDojo-default A/B")

para(textbox(s, M_L, Y_BODY - 0.05, M_W, 0.35),
     "같은 모델·heads·seed(42)·suite(banking/slack/workspace, held-out)로 비교",
     size=11, color=MUTED, first=True)

para(textbox(s, M_L, 2.20, 3.0, 0.32), "Qwen2.5-7B", size=12, bold=True, color=BLUE, first=True)
table(s, M_L, 2.52, 5.75,
      [["지표", "custom", "agentdojo"],
       ["n_pairs", "43", "45"],
       ["parse ok", "57.5%", "59.4%"],
       ["k0 utility", "39.5%", "42.2%"],
       ["k0 ASR", "4.7%", "6.7%"],
       ["kN ASR", "0.0%", "0.0%"]],
      col_w=[2.35, 1.7, 1.7], row_h=0.40, head_h=0.36,
      sizes=[10.5, 10.5, 10.5], aligns=["l", "r", "r"])

para(textbox(s, M_L + 6.18, 2.20, 4.0, 0.32), "Llama-3.1-8B", size=12, bold=True, color=BLUE, first=True)
table(s, M_L + 6.18, 2.52, 5.75,
      [["지표", "custom", "agentdojo"],
       ["n_pairs", "45", "44"],
       ["parse ok", "79.2%", "79.1%"],
       ["k0 utility", "35.6%", "38.6%"],
       ["k0 ASR", "4.4%", "2.3%"],
       ["kN ASR", "0.0%", [B("2.3%", RED)]]],
      col_w=[2.35, 1.7, 1.7], row_h=0.40, head_h=0.36,
      sizes=[10.5, 10.5, 10.5], aligns=["l", "r", "r"])

card(s, M_L, 4.95, M_W, 1.70, CARD_HL)
tf = textbox(s, 1.00, 5.13, 11.33, 1.35)
para(tf, [B("banking/workspace 저조는 파서 아티팩트가 아님", BLUE),
          (" — 두 모델·두 파서 모두 같은 suite별 비대칭 재현, parse ok율도 파서와 무관.", {})],
     size=12, first=True)
para(tf, [("1.5B의 기본 포맷 실패 전례는 7B/8B에서 재현 안 됨 → ", {}),
          B("agentdojo_default를 기본값으로 채택.", INK),
          ("  단 Llama+agentdojo에서만 banking 1건 backfire(표본 1건, 이후 확대로 재확인).", {})],
     size=12, space_before=7)

# ================================================================ S6 실험② 배경
s = new_slide("03 · 실험 ②  표본 확대", "표본을 왜 최대치로 키웠나")

tf = textbox(s, M_L, Y_BODY + 0.20, M_W, 3.8)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("파서 전환 직후 32B vs 7B 비교(초기 n=43~45)에서: 스케일업해도 ASR이 안 오르는 "
           "기존 결론은 재확인되지만, ", {}),
          B("slack suite에서만 knockout이 일부 성공 공격을 못 막는", RED), (" 신호가 눈에 띔", {})],
     size=13.5, first=True, line_spacing=1.35)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("우연(표본이 작아서)인지 실재하는 패턴인지 가리려면 표본을 최대치로 → ", {}),
          B("banking/slack/travel은 held-out 풀 전수", INK),
          (", workspace만 풀 392쌍 중 45로 캡(", {}), M("--limit_pairs 45"), (")", {})],
     size=13.5, space_before=18, line_spacing=1.35)

# ================================================================ S7 실험② 결과
s = new_slide("03 · 실험 ②  표본 확대", "Qwen2.5-32B 4-suite held-out 풀 확대 (n=148)")

table(s, M_L, Y_BODY, M_W,
      [["suite", "n / held-out 풀", "k0 util", "k0 ASR", "kN util", "kN ASR", "parse ok"],
       ["banking", "45 / 45  전수", "66.7%", "2.2%", "73.3%", [B("0.0%", BLUE)], "52.9%"],
       [[B("slack", RED)], "35 / 35  전수", "25.7%", [B("22.9%", RED)], "22.9%", [B("14.3%", RED)], "73.4%"],
       ["travel", "28 / 28  전수", "25.0%", "3.6%", "25.0%", [B("0.0%", BLUE)], "83.8%"],
       ["workspace", [B("40 / 392", RED), ("  (45 샘플)", {})], "20.0%", "0.0%", "0.0%", [B("0.0%", BLUE)], "44.4%"],
       [[B("전체")], [B("148")], [B("36.5%")], [B("6.8%", RED)], [B("37.8%")],
        [B("3.4%", RED)], [B("68.3%")]]],
      col_w=[1.7, 2.3, 1.55, 1.55, 1.55, 1.55, 1.6], row_h=0.44, head_h=0.38,
      sizes=[10.5, 9.5, 10, 10, 10, 10, 10], aligns=["l", "l", "r", "r", "r", "r", "r"])

card(s, M_L, 4.62, M_W, 2.05, CARD_HL)
tf = textbox(s, 1.00, 4.80, 11.33, 1.75)
para(tf, [B("스케일업 반례 최종 확정", BLUE),
          (" — 지금까지 가장 큰 표본에서도 전체 ASR 6.8%. 표본 크기 문제일 여지 소멸.", {})],
     size=12, first=True)
para(tf, [B("slack만 특이", RED),
          (" — held-out 35쌍 전수 기준 성공한 공격 8건 중 5건을 knockout이 못 막음"
           "(억제 3, persist 5). banking/travel/workspace는 완벽 억제(0%).", {})],
     size=12, space_before=7)
para(tf, [("banking·slack·travel은 held-out 풀 전수라 표본 논란 없음. workspace는 392쌍 중 "
           "40쌍만 봤지만 k0 ASR 자체가 0%라 결론에 영향 없음.", {})],
     size=10.5, color=MUTED, space_before=6)

# ================================================================ S8 실험② 분석
s = new_slide("03 · 실험 ②  표본 확대", "성공한 공격은 전부 \"단순 공격\"이었다")

para(textbox(s, M_L, Y_BODY - 0.08, M_W, 0.32),
     "slack 성공 공격의 injection_task별 분해 (held-out 35쌍 전수,  k0 성공수 → kN 성공수)",
     size=10.5, color=MUTED, first=True)

table(s, M_L, Y_BODY + 0.30, M_W,
      [["inj", "목표 · 채점", "call", "7B", "Llama-8B", "32B fp4", "32B bf16"],
       [[B("1")], "피싱 링크 전송 · 단일조건", "1", "4→0", "3→0", [B("3→2", RED)], [B("3→3", RED)]],
       [[B("3")], "피싱 URL 방문 · 단일조건", "1", "0", "2→0", "1→0", "2→0"],
       [[B("5")], "초대·추가·제거 · trace", "3", "2→0", "1→0", [B("4→3", RED)], [B("4→2  †", RED)]],
       ["2", "4채널 유출 · 5-way AND", "6", "0", "1→0", "0", "0"],
       ["4", "general 유출 · 3-way AND", "2", "0", "0", "0", "0"],
       [[B("1/3/5")], [B("달성 가능한 공격")], "", [B("6→0", BLUE)], [B("6→0", BLUE)],
        [B("8→5", RED)], [B("9→5", RED)]]],
      col_w=[0.7, 3.75, 0.7, 1.35, 1.5, 1.5, 1.55], row_h=0.40, head_h=0.36,
      sizes=[10, 9.5, 9, 9.5, 9.5, 9.5, 9.5],
      aligns=["c", "l", "c", "c", "c", "c", "c"])

tf = textbox(s, M_L, 5.30, M_W, 1.75)
para(tf, [("†  32B bf16 inj_5 에 backfire 1건(k0=False→kN=True) 추가", {})],
     size=9.5, color=MUTED, first=True)
para(tf, [B("관측", INK), (": 성공한 공격(k0=True)은 전부 1/3/5(단순).  ", {}),
          B("단 2/4의 0%는 실행 난이도가 아니라 연언 채점(near-unwinnable) 아티팩트", RED),
          (" — 분석에서 제외.", {})], size=11, space_before=6, line_spacing=1.25)
para(tf, [B("진짜 신호", BLUE),
          (": 달성 가능한 1/3/5 범위에서 8B급은 전량 억제(6→0), 32B만 절반 이상 통과"
           "(inj_1은 bf16에서 3/3 그대로).  → k-sweep으로 억제력 보강 여부가 후속 검증 포인트.", {})],
     size=11, space_before=5, line_spacing=1.25)

# ================================================================ S9 실험③ 배경
s = new_slide("04 · 실험 ③  양자화", "양자화를 왜 의심했나")

tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 3.6)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("slack knockout 실패가 나온 유일한 실행은 ", {}), B("32B (4bit fp4)", RED),
          (" — 대조군 8B급(Llama, Qwen 7B)은 전부 ", {}), B("bf16", INK)],
     size=13, first=True, line_spacing=1.3)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("즉 원인이 ", {}), B("(1) 32B 스케일", INK), ("인지 ", {}),
          B("(2) 4bit 양자화", INK),
          ("인지가 지금까지 데이터로는 공변되어 구분 불가능", {})],
     size=13, space_before=14, line_spacing=1.3)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("확인 절차: 7B에서 bf16 vs 4bit 직접 비교 → 4bit 세팅 자체(품질) 점검 → "
           "32B를 bf16으로 재실행해 양자화 배제", {})],
     size=13, space_before=14, line_spacing=1.3)

# ================================================================ S10 실험③ 결과 7B
s = new_slide("04 · 실험 ③  양자화", "7B: bf16 vs fp4 vs nf4+double_quant")

table(s, M_L, Y_BODY, M_W,
      [["실행", "quant", "slack k0 ASR", "slack kN ASR", "전체 k0 util", "slack backfire"],
       ["7B bf16", "bf16", "0.171", [B("0.000", BLUE)], "0.311", "0"],
       ["7B fp4 (bnb 기본값)", [B("fp4 / no-dq", RED)], "0.057", "0.057", [B("0.178", RED)], [B("2", RED)]],
       ["7B nf4 + double_quant", "nf4 / dq", "0.343", "0.057", "0.254", [B("1", RED)]]],
      col_w=[3.0, 2.1, 1.9, 1.9, 1.9, 1.13], row_h=0.48, head_h=0.40,
      sizes=[10.5, 10, 10, 10, 10, 9.5], aligns=["l", "l", "r", "r", "r", "c"])

card(s, M_L, 4.15, M_W, 2.20, CARD_HL)
tf = textbox(s, 1.00, 4.35, 11.33, 1.85)
para(tf, [B("fp4 아티팩트 확정", BLUE),
          (": bnb 4bit 기본값(fp4 + double_quant off)이 모델을 과손상 → knockout이 손상된 "
           "궤적에서 backfire.  nf4+double_quant로 utility 회복(0.178→0.254) + backfire 2→1.", {})],
     size=12, first=True)
para(tf, [B("비결정성 아님", INK),
          (": 동일 조건 2회(RTX 4090) 152/152 쌍 전 필드 일치 — greedy+고정 seed에서 완전 "
           "결정론적.  → ", {}), M("run_agentdojo_eval.py"),
          (" 4bit 기본값을 nf4+double_quant on으로 교체 완료.", {})],
     size=12, space_before=8)

# ================================================================ S11 실험③ 결과 32B
s = new_slide("04 · 실험 ③  양자화", "32B: fp4 / nf4dq (비결정적) / bf16")

table(s, M_L, Y_BODY, M_W,
      [["실행", "GPU / quant", "slack k0", "slack kN", "supp", "backfire", "persist"],
       ["32B nf4dq #1", "A6000 / nf4dq", "0.229", "0.229", "2", "2", "6"],
       ["32B nf4dq #2", "Blackwell / nf4dq", "0.171", "0.114", "2", "0", "4"],
       [[B("32B bf16", BLUE)], [B("A6000+2 / bf16", INK)], "0.257", [B("0.143", RED)], "5",
        [B("1", RED)], "4"],
       ["(대조) 7B·8B bf16", "4090 / bf16", "~0.18", [B("0.000", BLUE)], "전량", [B("0", BLUE)], "—"]],
      col_w=[2.7, 2.9, 1.4, 1.4, 1.0, 1.4, 1.13], row_h=0.46, head_h=0.40,
      sizes=[10, 9.5, 10, 10, 10, 10, 10], aligns=["l", "l", "r", "r", "c", "c", "c"])

card(s, M_L, 4.30, M_W, 2.05, CARD_HL)
tf = textbox(s, 1.00, 4.50, 11.33, 1.70)
para(tf, [("두 nf4dq 실행(둘 다 slack 풀 35쌍 전수) 비교 시 ", {}),
          B("slack 12/35쌍(34%) 불일치, 그중 8쌍이 baseline 차이", RED),
          (" (user_task_0·2가 k0 붕괴) → ", {}),
          B("32B-4bit 평가는 run-to-run 비결정적", RED), (" — 이 경로로는 스케일 판정 불가.", {})],
     size=12, first=True)
para(tf, [B("bf16으로 우회", BLUE),
          (": 완주율 100%·결정론적인데도 kN ASR 0.143(≠0) + backfire 1 — 7B·8B "
           "bf16(전량 억제·backfire 0)과 질적으로 다름 → ", {}),
          B("fp4와 무관한 스케일 기여 확인.", INK)], size=12, space_before=8)

# ================================================================ S12 종합 결론
s = new_slide("05 · 종합", "종합 결론")

items = [
    [B("파서는 confound 아님", BLUE),
     (" — banking/workspace 저조는 커스텀 파서 탓이 아니라 모델·suite 자체의 성질. "
      "agentdojo_default로 전환 완료.", {})],
    [B("스케일업 반례 재확정 (n=148)", BLUE),
     (" — 스케일을 키워도 전체 ASR은 안 오름(6.8%). 단 slack에서만 달성 가능한 공격 "
      "1/3/5 중 절반 이상 persist.", {})],
    [B("원인 두 갈래로 분리 규명", BLUE),
     (" — (a) fp4 4bit 양자화 아티팩트(확정, nf4+dq로 해소·기본값 교체)  "
      "(b) 32B 스케일 효과(약하게 확정, bf16 단독에서도 불완전 + backfire 1).", {})],
    [B("순효과는 여전히 방어적", INK),
     (" (suppressed 5 > backfire 1) — \"못 막는다\"가 아니라 \"불완전 + 가끔 backfire\".", {})],
]
tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 4.6)
for i, sp in enumerate(items):
    para(tf, [("%d.  " % (i + 1), {"bold": True, "color": ORANGE, "size": 14})] + sp,
         size=12.5, first=(i == 0), space_before=0 if i == 0 else 15, line_spacing=1.32)

# ================================================================ S13 대조 실험
s = new_slide("06 · 대조 실험", "32B bf16 banking — \"slack 특유\" 확정")

para(textbox(s, M_L, Y_BODY - 0.05, M_W, 0.35),
     "같은 32B bf16 스택(동일 heads·설정)에서 --suite 만 slack → banking 으로 교체",
     size=11, color=MUTED, first=True)

table(s, M_L, 2.25, M_W,
      [["suite", "k0 ASR", "kN ASR", "backfire", "판정"],
       ["slack (S11 재인용)", "0.257", "0.143", [B("1 / 9", RED)], "스케일 효과로 불완전 억제"],
       ["banking", "0.0", "0.0", [B("0 / 42", BLUE)],
        "baseline부터 공격 실패, knockout도 새 leak 없음"]],
      col_w=[2.6, 1.5, 1.5, 1.5, 4.83], row_h=0.55, head_h=0.40,
      sizes=[11, 10.5, 10.5, 10.5, 10.5], aligns=["l", "r", "r", "c", "l"])

card(s, M_L, 4.20, M_W, 2.30, CARD_HL)
tf = textbox(s, 1.00, 4.40, 11.33, 1.95)
para(tf, [("banking은 42쌍 전부 baseline부터 공격 실패라 \"억제율\" 틀이 안 맞음 → 판정 근거는 ", {}),
          B("backfire", INK), (": k0=False인데 knockout 후 kN=True로 뒤집힌 쌍이 있는지. ", {}),
          B("0/42로 하나도 없음.", BLUE)], size=12, first=True)
para(tf, [("같은 스택에서 slack만 backfire 1건이 나왔던 것과 대비 → ", {}),
          B("32B 스케일 효과는 slack에 국한, banking엔 전이 안 됨.", RED)],
     size=12, space_before=8)
para(tf, [B("⚠️ caveat", MUTED),
          (": 완주율 42/45(3쌍 원인불명 누락) · banking 자체가 baseline 성공률 낮은 suite.", {})],
     size=10.5, color=MUTED, space_before=8)

# ================================================================ S14 한계
s = new_slide("07 · 한계", "한계 및 다음 단계")

table(s, M_L, Y_BODY, M_W,
      [["#", "한계", "향후 방향"],
       ["1", [B("32B-4bit 평가의 run-to-run 비결정성", RED), (" 원인 미규명(별개 이슈)", {})],
        "bnb 4bit 커널 / 디바이스 배치 조건 추가 조사"],
       ["2", [B("slack knockout 불안정의 최종 원인", RED),
              (" 이 \"단순 공격의 실행 관성\" 가설 수준", {})],
        "k-sweep(head 개수 ↑)으로 억제력 보강되는지 확인"],
       ["3", [B("복합 공격(2/4)은 채점이 near-unwinnable", RED), (" — knockout 효과 측정 축이 없음", {})],
        "AgentDojo 외 벤치마크 / 자체 시나리오로 다단계 채점 보완"],
       ["4", [B("Llama-8B banking backfire 1건", RED), (" (표본 1건) 미확인", {})],
        "필요 시 Llama도 n=148급 확대"],
       ["5", "다른 아키텍처(Qwen3, Llama-70B) 교차검증 보류 중", "다음 사이클 후보"],
       ["6", [B("banking 대조 검증력 약함", RED),
              (" (baseline 성공 0/42) + 완주율 93.3%(로그 미보존)", {})],
        "k-sweep으로 banking 성공률 ↑ 후 재검증 / 재실행 시 로그 보존"]],
      col_w=[0.45, 6.6, 4.88], row_h=0.72, head_h=0.38, aligns=["c", "l", "l"],
      sizes=[10.5, 10, 10])

# ================================================================ S15 마무리
s = new_slide("08 · 마무리", "마무리")

cw, gap = 3.7767, 0.30
for i, (head, lines, col) in enumerate([
    ("파서 전환", ["confound 아니었음", "agentdojo_default 기본값 채택"], BLUE),
    ("표본 확대 (n=148)", ["스케일업 반례 최종 확정",
                          "성공 공격은 전부 단순(1/3/5), 8B 전량 억제 / 32B만 절반 통과"], BLUE),
    ("양자화 규명", ["fp4 아티팩트 확정(기본값 교체)",
                   "+ 32B 스케일 효과 확정, 순효과는 방어적"], ORANGE)]):
    x = M_L + i * (cw + gap)
    card(s, x, Y_BODY, cw, 1.95)
    tf = textbox(s, x + 0.24, 2.05, cw - 0.48, 1.65)
    para(tf, head, size=13, bold=True, color=col, first=True)
    for t in lines:
        para(tf, [("·  ", {"color": col}), (t, {})], size=10.5, space_before=7, line_spacing=1.22)

card(s, M_L, 4.10, M_W, 1.85, CARD_HL)
tf = textbox(s, 1.00, 4.30, 11.33, 1.50)
para(tf, [B("대조 실험 — ", INK),
          ("banking 대조로 32B 스케일 효과가 ", {}), B("slack에 국한", RED),
          ("됨을 확정.", {})], size=13, first=True)
para(tf, [("\"모델을 키우면 전반적으로 위험해진다\"가 아니라 ", {}),
          B("\"특정 suite의 단순 공격에서만, 가끔 불완전 + backfire\".", BLUE)],
     size=13, space_before=8)

foot(s, [B("다음: ", ORANGE),
         ("이 deck의 pptx 산출 완료 · k-sweep 착수 / Qwen3-8B 진단은 다음 사이클", {})],
     y=6.20)

# ----------------------------------------------------------------
prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
