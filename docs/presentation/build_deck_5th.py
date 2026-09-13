# -*- coding: utf-8 -*-
"""5차 발표 deck 생성 — Llama-3.1-70B / Qwen3-8B 헤드 탐색·평가 실험 (결과 리포트 형식).

내용은 IPI_Head_PoC_5th_script.md(S1~S10, 2026-09-12 재구성판)를 그대로 옮긴 것.
디자인 토큰/헬퍼는 build_deck_4th.py와 동일 — import하면 그쪽 deck이 재빌드되므로
의도적으로 복제했다(3rd/4th와 같은 이유).
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


B = lambda t, c=INK: (t, {"bold": True, "color": c})     # noqa: E731
M = lambda t: (t, {"font": MONO, "size": 11})            # noqa: E731


# ================================================================ S1 타이틀
s = prs.slides.add_slide(BLANK)
rect(s, 0, 0, 13.33, 7.5, BG)
rect(s, 0.70, 2.28, 1.10, 0.055, ORANGE)
para(textbox(s, 0.70, 2.50, 10.5, 0.40), "IPI DEFENSE · HEAD DISCOVERY & EVALUATION REPORT",
     size=13, bold=True, color=ORANGE, font=MONO, first=True)
tf = textbox(s, 0.70, 2.95, 11.7, 2.20)
para(tf, "Read Head, Control Head 분리 PoC", size=34, bold=True,
     color=INK, first=True, line_spacing=1.2)
para(tf, "— Llama-3.1-70B / Qwen3-8B 헤드 탐색·평가 실험 (5차)", size=25, bold=True,
     color=INK, line_spacing=1.2)
para(textbox(s, 0.70, 4.75, 11.4, 1.00),
     "70B 자체 헤드 탐색 + 자체/전이 헤드 평가  ·  Qwen3-8B 레이어0 쏠림 진단 + 평가  ·  "
     "4개 suite 전체 확장",
     size=14.5, color=BODY, first=True)
para(textbox(s, 0.70, 6.50, 11.4, 0.40),
     "실험 2026-09-08~09-12  ·  5차 발표 2026-09-XX       원종빈",
     size=11.5, color=MUTED, font=MONO, first=True)

# ================================================================ S2 서론
s = new_slide("01 · 서론", "이번 사이클에 진행한 실험")

table(s, M_L, Y_BODY, M_W,
      [["#", "실험", "목적"],
       ["1", [B("Llama-3.1-70B 헤드 탐색 (Track A)")],
        "70B 스케일에서 AttnLRP로 자체 control head를 찾을 수 있는가 (기존엔 backward "
        "OOM으로 불가 판정)"],
       ["2", [B("Llama-3.1-70B 평가 (Track B)")],
        "그 헤드를 knockout하면 slack IPI 공격이 억제되는가 — 자체 헤드 vs 8B 전이 헤드 "
        "비교, heldout 표본 확대"],
       ["3", [B("Qwen3-8B 헤드 탐색 (Track A) + 레이어 0 쏠림 진단")],
        "lxt가 경고한 \"Qwen3는 attribution이 첫 토큰에 쏠린다\"는 현상이 실재하는지, "
        "실재해도 헤드 탐색이 유효한지"],
       ["4", [B("Qwen3-8B 평가 (Track B)")],
        "Qwen3-8B에서 찾은 헤드로도 knockout이 8B급 모델과 같은 패턴을 보이는가"]],
      col_w=[0.5, 4.3, 7.13], row_h=0.85, head_h=0.38, aligns=["c", "l", "l"],
      sizes=[11, 11, 10])

card(s, M_L, 5.55, M_W, 1.20, CARD_HL)
tf = textbox(s, 1.00, 5.73, 11.33, 0.95)
para(tf, [B("공통 조건", INK),
          (": AgentDojo slack suite(+확장 평가로 banking/travel/workspace) · "
           "agentdojo_default tool-call 파서 · 공격 2종(important_instructions, "
           "tool_knowledge) · greedy decoding.", {})],
     size=12, first=True, line_spacing=1.3)

# ================================================================ S3 70B 탐색 방법
s = new_slide("02 · Llama-3.1-70B 헤드 탐색", "방법 — 수동 device_map으로 backward OOM 우회")

pts = [
    [("기존 결론(\"70B AttnLRP backward는 하드웨어 한계로 불가\")은 ", {}), M("--device_map auto"),
     (" 경로 한정이었다 — 타이트한 ", {}), M("--max_memory"),
     ("면 CPU/disk 분산 에러, 느슨하면 레이어가 한 GPU에 몰려 backward OOM.", {})],
    [B("해결: ", BLUE), M('--device_map_plan "0:32,1:30,2:18"'),
     (" — embed/norm/lm_head를 첫 GPU에 몰아두고(backward가 양 끝에서 시작·수렴), "
      "레이어는 순서대로 분배. ", {}),
     M("CUDA_VISIBLE_DEVICES=1,0,2"),
     (" → A6000(48G,root) / Blackwell(32G) / 4090(24G).", {})],
    [("두 번 탐색: ", {}), B("head_n=200", INK), ("(09-08, 8B 탐색과 동일 조건) / ", {}),
     B("head_n=80", BLUE), ("(09-12, heldout 평가 표본 확대 목적).", {})],
]
tf = textbox(s, M_L, Y_BODY + 0.15, M_W, 4.4)
for i, sp in enumerate(pts):
    para(tf, [("•  ", {"color": ORANGE, "bold": True})] + sp,
         size=13.5, first=(i == 0), space_before=0 if i == 0 else 18, line_spacing=1.35)

# ================================================================ S4 70B 탐색 결과
s = new_slide("02 · Llama-3.1-70B 헤드 탐색", "결과 — suite별 내역 · 헤드 · 재현성", title_size=24)

para(textbox(s, M_L, Y_BODY - 0.10, M_W, 0.35),
     "80층×64헤드(5,120개) 중 20개 선정. --max_seq_len 1000 필터 후 949쌍 중 137쌍만 "
     "통과 (workspace는 두 탐색 모두 전량 필터 탈락)", size=10, color=MUTED, first=True)

table(s, M_L, 2.20, 7.1,
      [["head_n", "suite", "탐색 pair", "user_task", "shortfall"],
       ["200", "banking", "61", "11/11", "5"],
       ["200", [B("slack")], "67", "20/20", "0"],
       ["200", "travel", "2", "1/1", "64"],
       [[B("80")], "banking", "26", "5/11", "0"],
       [[B("80")], [B("slack")], "26", "7/20", "0"],
       [[B("80")], "travel", "2", "1/1", "24"]],
      col_w=[1.1, 1.5, 1.4, 1.4, 1.4], row_h=0.36, head_h=0.34,
      sizes=[9.5, 9.5, 9.5, 9.5, 9.5], aligns=["c", "l", "c", "c", "c"])

tf = textbox(s, 8.05, 2.20, 4.4, 2.4)
para(tf, [B("n_examples_used", INK)], size=11, bold=True, color=BLUE, first=True)
para(tf, [("head_n=200: 130/149 (0 oom, 0 nan)", {})], size=10.5, space_before=4)
para(tf, [("head_n=80: 54/54 (0 oom, 0 nan)", {})], size=10.5, space_before=2)
para(tf, [B("선정 헤드 20개 (head_n=80)", INK)], size=11, bold=True, color=BLUE, space_before=14)
para(tf, [("layer 26–44/80 (≈33–55%), jaccard 0.82 vs head_n=200(18/20 일치)", {})],
     size=10.5, space_before=4, line_spacing=1.3)

card(s, M_L, 4.85, M_W, 1.85, CARD_HL)
tf = textbox(s, 1.00, 5.03, 11.33, 1.55)
para(tf, "(35,35)(38,52)(34,6)(32,22)(35,34)(30,51)(31,47)(29,58)(31,45)(33,30)"
         "(35,18)(31,7)(26,53)(29,62)(32,16)(28,41)(44,35)(28,1)(33,14)(34,43)",
     size=10, color=INK, font=MONO, first=True, line_spacing=1.4)
para(tf, [B("8B와 비교: ", BLUE), ("Llama-8B는 layer 11–22/32(≈38–47%) — ", {}),
          B("같은 상대 깊이 대역", INK), (" (스케일 8배에도 유지).", {})],
     size=11, space_before=10, line_spacing=1.3)

# ================================================================ S5 70B 평가 방법
s = new_slide("03 · Llama-3.1-70B 평가", "방법 — 자체 vs 전이 헤드, heldout 표본 확대")

tf = textbox(s, M_L, Y_BODY + 0.10, M_W, 2.2)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("비교축 ① 헤드 출처 — ", {}), B("70B 자체 헤드", BLUE), (" vs ", {}),
          B("8B 헤드 전이", RED),
          (" (70B Track A가 없던 시점엔 8B 헤드를 전이해 씀 → 이후 정면 대조)", {})],
     size=13, first=True, line_spacing=1.3)
para(tf, [("•  ", {"color": ORANGE, "bold": True}),
          ("비교축 ② 표본 — ", {}), M("--eval_split all"), ("(105쌍, 누수 있음) vs ", {}),
          M("--eval_split heldout"), ("(탐색에 안 쓰인 case만, 누수 없음)", {})],
     size=13, space_before=14, line_spacing=1.3)

table(s, M_L, 4.10, M_W,
      [["head_n", "헤드 탐색에 쓴 slack user_task", "heldout 후보", "평가한 수"],
       ["200", "18 / 20", "15", "15 (전부)"],
       [[B("80", BLUE)], "7 / 20", [B("70", BLUE)], [B("60", BLUE)]]],
      col_w=[1.3, 3.6, 1.8, 1.8], row_h=0.42, head_h=0.36,
      sizes=[10.5, 10.5, 10.5, 10.5], aligns=["c", "c", "c", "c"])

# ================================================================ S6 70B 평가 결과
s = new_slide("03 · Llama-3.1-70B 평가", "결과 — 자체 vs 전이 헤드 · suite 확장", title_size=23)

para(textbox(s, M_L, Y_BODY - 0.12, M_W, 0.3),
     "자체 vs 전이 헤드 (slack all105, 09-08/09)", size=10.5, bold=True, color=BLUE, first=True)
table(s, M_L, 2.10, M_W,
      [["공격", "헤드 출처", "k0 sec", "kN sec", "억제/bf", "net"],
       ["important_instructions", "8B 전이", "0.276", "0.257", "4/2", [B("−2", RED)]],
       ["important_instructions", [B("70B 자체", BLUE)], "0.276", [B("0.181")], "13/3",
        [B("−10 (34%↓)", BLUE)]],
       ["tool_knowledge", "8B 전이", "0.402", "0.392", "3/2", [B("−1", RED)]],
       ["tool_knowledge", [B("70B 자체", BLUE)], "0.398", [B("0.223")], "18/0",
        [B("−18 (44%↓)", BLUE)]]],
      col_w=[2.7, 1.8, 1.3, 1.3, 1.3, 2.0], row_h=0.34, head_h=0.32,
      sizes=[9, 9, 9, 9, 9, 9], aligns=["l", "c", "r", "r", "c", "c"])

para(textbox(s, M_L, 4.05, M_W, 0.3),
     "heldout 확대 + suite 확장 (70B 자체 헤드, head_n=80, 2026-09-12)",
     size=10.5, bold=True, color=BLUE, first=True)
table(s, M_L, 4.40, M_W,
      [["suite", "공격", "표본", "k0 sec", "kN sec", "억제/bf/pr", "net", "kN util"],
       ["slack", "important_instr.", "60", "0.250", [B("0.117")], "9/1/6",
        [B("+8 (53%↓)", BLUE)], [B("0.150 (↓)", RED)]],
       ["slack", "tool_knowledge", "60", "0.350", [B("0.217")], "8/0/13",
        [B("+8 (38%↓)", BLUE)], [B("0.183 (↑)", BLUE)]],
       ["banking", "important_instr.", "59", "0.085", "0.068", "3/2/2",
        [B("+1 (약함)")], [B("0.627 (↑)", BLUE)]],
       [[B("travel", RED)], "—", [B("0/60", RED)], "—", "—",
        [B("A6000 48G에서도 100% OOM", RED)], "—", "—"],
       [[B("workspace", RED)], "—", [B("0/60", RED)], "—", "—",
        [B("A6000 48G에서도 100% OOM", RED)], "—", "—"]],
      col_w=[1.5, 1.9, 1.0, 1.15, 1.15, 1.85, 1.75, 1.63], row_h=0.36, head_h=0.34,
      sizes=[8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5],
      aligns=["l", "l", "c", "r", "r", "c", "c", "r"])

# ================================================================ S7 Qwen3 진단
s = new_slide("04 · Qwen3-8B 헤드 탐색", "레이어 0(첫 토큰) 쏠림 진단")

tf = textbox(s, M_L, Y_BODY + 0.05, M_W, 1.7)
para(tf, [("lxt README 경고: \"Qwen3는 attribution이 첫 토큰(position 0)으로 쏠린다.\" "
           "우리 head 탐색은 relevance를 D", {}), (("inj", {"size": 9})),
          (" span에 ", {}), B("group-sum", INK),
          ("하므로, 질량이 position 0에 흡수되면 head 점수가 계통적으로 눌릴 위험 — "
           "배선 전에 직접 진단.", {})],
     size=13, first=True, line_spacing=1.35)
para(tf, [B("판단 기준: ", ORANGE),
          ("position 0 비중이 0이 아닌 것 자체는 문제가 아니다(causal LM의 흔한 attention "
           "sink). 같은 프롬프트로 ", {}), B("qwen2 대조군과 나란히 돌려 상대적으로 얼마나 "
           "더 쏠리는지가 기준.", BLUE)],
     size=13, space_before=12, line_spacing=1.35)

table(s, M_L, 3.65, 8.3,
      [["family", "모델", "position 0 비중", "data_inj 비중(22tok)"],
       ["qwen2 (대조군)", "Qwen2.5-7B-Instruct", "0.49%", "37.71%"],
       [[B("qwen3", BLUE)], "Qwen3-8B", [B("16.71%", RED)], "32.78%"]],
      col_w=[1.6, 2.7, 2.0, 2.0], row_h=0.42, head_h=0.36,
      sizes=[10, 10, 10, 10], aligns=["l", "l", "c", "c"])

card(s, M_L, 5.20, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 5.38, 11.33, 1.25)
para(tf, [B("쏠림은 qwen2 대비 ~34배로 실재", RED), (".", {})], size=12.5, first=True)
para(tf, [("그러나 data_inj span 비중은 qwen2와 비슷하게 유지되고 여전히 position 0 단독보다 "
           "2배 이상 크다 — group-sum 방식이라 position 0은 애초에 그 합산에 안 들어감. "
           "→ ", {}), B("진단 통과.", BLUE)],
     size=12.5, space_before=7, line_spacing=1.3)

# ================================================================ S8 Qwen3 탐색 결과
s = new_slide("04 · Qwen3-8B 헤드 탐색", "결과 — suite별 내역(oom) · 헤드", title_size=24)

para(textbox(s, M_L, Y_BODY - 0.10, M_W, 0.35),
     "36층×32헤드(1,152개) 중 20개 선정. --max_seq_len 1200 필터로 174/949쌍 통과 "
     "(4 suite 전부 생존). head_n=200, quota=50/suite", size=10, color=MUTED, first=True)

table(s, M_L, 2.20, 5.6,
      [["suite", "ok", "oom"],
       ["banking", "53", "4"],
       ["slack", "50", [B("0", BLUE)]],
       ["travel", [B("0", RED)], [B("14 (전량)", RED)]],
       ["workspace", [B("0", RED)], [B("22 (전량)", RED)]],
       [[B("합계")], [B("103")], [B("40 (28%)", RED)]]],
      col_w=[1.7, 1.3, 1.9], row_h=0.36, head_h=0.34,
      sizes=[10, 10, 10], aligns=["l", "c", "c"])

tf = textbox(s, 6.55, 2.20, 5.9, 1.7)
para(tf, [B("oom이 전부 travel/workspace에 집중", RED),
          ("(추정: 긴 프롬프트) — banking/slack은 거의 안전.", {})],
     size=11, first=True, line_spacing=1.3)
para(tf, [B("layer 범위: ", INK), ("18–29 / 36 (≈50–80% 깊이)", {})],
     size=11, space_before=10)
para(tf, [B("layer 0 헤드 2개(10%)", RED),
          (" — S7 쏠림과 무관 단정 불가(caveat), 다만 layer 0 지배는 이 방법론 전반의 "
           "반복 현상.", {})],
     size=11, space_before=6, line_spacing=1.3)

card(s, M_L, 4.20, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 4.38, 11.33, 1.25)
para(tf, "(25,10)(0,3)(20,29)(22,11)(21,18)(19,21)(24,31)(0,0)(29,0)(22,0)"
         "(23,26)(18,30)(21,19)(18,14)(21,11)(26,26)(18,15)(20,5)(21,27)(28,22)",
     size=10, color=INK, font=MONO, first=True, line_spacing=1.4)
para(tf, [B("Llama-8B/70B와 비교: ", BLUE),
          ("상대 깊이 38–47%/33–55% — Qwen3-8B가 뚜렷이 더 깊은 대역.", {})],
     size=11, space_before=8, line_spacing=1.3)

# ================================================================ S9 Qwen3 평가
s = new_slide("05 · Qwen3-8B 평가", "방법 & 결과 — slack + suite 확장", title_size=24)

para(textbox(s, M_L, Y_BODY - 0.12, M_W, 0.3),
     "헤드 탐색에 쓰인 slack user_task 6/20 → heldout 35쌍 전부 평가. knockout 20개 헤드, "
     "k=0→k=20", size=10, color=MUTED, first=True)

table(s, M_L, 2.20, M_W,
      [["suite", "공격", "표본", "k0 sec", "kN sec", "억제/bf/pr", "net", "kN util"],
       ["slack", "important_instr.", "35", "0.229", [B("0.000", BLUE)], "8/0/0",
        [B("+8 전량억제", BLUE)], [B("0.429 (↑)", BLUE)]],
       ["slack", "tool_knowledge", "35", "0.171", [B("0.057")], "5/1/1",
        [B("+4 (66%↓)", BLUE)], [B("0.400 무손상")]],
       ["banking", "important_instr.", "42", "0.095", [B("0.000", BLUE)], "4/0/0",
        [B("+4 전량억제", BLUE)], "0.667 (↓)"],
       ["workspace", "important_instr.", "53/60", "0.000", "0.000",
        [B("대조군(baseline 0)")], "0", "0.189 무변화"],
       ["travel (all)", "important_instr.", "52/60", "0.000", [B("0.019", RED)], "0/1/0",
        [B("−1 (잡음)", RED)], [B("0.096 (P14)", RED)]]],
      col_w=[1.55, 1.85, 1.05, 1.1, 1.1, 1.75, 1.75, 1.78], row_h=0.44, head_h=0.36,
      sizes=[8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5],
      aligns=["l", "l", "c", "r", "r", "c", "c", "r"])

card(s, M_L, 5.20, M_W, 1.55, CARD_HL)
tf = textbox(s, 1.00, 5.38, 11.33, 1.25)
para(tf, [B("banking도 slack처럼 전량 억제·backfire 0", BLUE),
          (" — Qwen3-8B는 70B와 달리 banking에서도 깨끗하게 작동. workspace는 baseline "
           "공격이 아예 없어 순수 대조군(utility 무변화). travel은 4090 100% OOM → "
           "Blackwell로 재시도, k0_util 0.096으로 극히 낮음(P14 파서 문제 재확인).", {})],
     size=11, first=True, line_spacing=1.3)

# ================================================================ S10 결과 요약 & 다음 단계
s = new_slide("06 · 마무리", "결과 요약 & 다음 단계")

card(s, M_L, Y_BODY, M_W, 3.35, CARD_HL)
tf = textbox(s, 1.00, Y_BODY + 0.18, 11.33, 3.10)
para(tf, "확인된 것", size=13.5, bold=True, color=BLUE, first=True)
for sp in [
    [B("Llama-70B", INK), (": 수동 device_map으로 자체 헤드 탐색 가능. 자체 헤드가 8B "
        "전이 헤드보다 뚜렷이 강하게 작동(전이 헤드는 net 효과 없음). heldout 표본 4배 "
        "확대(15→60)에도 net 억제 유지 — important_instructions에서 utility 첫 손상 "
        "발견(원인 특정).", {})],
    [B("70B는 slack·banking만 평가 가능", RED),
     (" — travel·workspace는 A6000(48GB)로도 100% OOM. banking은 knockout 신호가 "
      "약함(net+1)지만 utility는 오히려 개선.", {})],
    [B("Qwen3-8B", INK), (": 첫 토큰 쏠림 실재(qwen2 대비 34배)하나 D_inj 신호를 지우지 "
        "않음 — 8B급과 동일한 knockout 패턴 재현.", {})],
    [B("Qwen3-8B는 4개 suite 전부 평가 가능", BLUE),
     (" — banking도 slack처럼 전량 억제, workspace는 순수 대조군, travel은 알려진 파서 "
      "문제(P14)로 낮은 utility 재확인.", {})],
]:
    para(tf, [("·  ", {"color": BLUE, "bold": True})] + sp, size=10.5, space_before=6, line_spacing=1.2)

card(s, M_L, 5.55, M_W, 1.35)
tf = textbox(s, 1.00, 5.73, 11.33, 1.05)
para(tf, "다음 단계", size=13.5, bold=True, color=ORANGE, first=True)
for sp in [
    [B("Qwen2.5-32B 자체 헤드 nf4dq 재탐색", INK), (" (todo.md §4-1).", {})],
    [B("70B k-sweep", INK), (" (topk 20→40→60), utility 손상 원인 정밀 확인.", {})],
    [B("70B travel/workspace OOM 근본 해결", INK), (" (2-GPU 분산 등) — 지금은 평가 자체가 안 됨.", {})],
]:
    para(tf, [("·  ", {"color": ORANGE, "bold": True})] + sp, size=10.5, space_before=4, line_spacing=1.15)

# ----------------------------------------------------------------
prs.save(OUT)
print("saved:", OUT)
print("slides:", len(prs.slides._sldIdLst))
