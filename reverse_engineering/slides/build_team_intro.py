"""RE, REA 팀 입문 deck 생성기.

실행: python reverse_engineering/slides/build_team_intro.py [추가 슬라이드.json]
SLIDES 하나에서 re-rea-team-intro.pptx, .bento.html, .md 를 함께 만든다.
세 파일을 직접 고치지 말고 SLIDES 를 고친 뒤 다시 실행한다.
내용의 근거는 이 폴더(reverse_engineering/)의 .md 문서다.

회사에서 얻은 REA 결과는 SLIDES 에 넣지 않는다(공개 저장소다). slides/extra_사내용.json 에
적으면 사이사이에 끼워 넣은 re-rea-team-intro_사내용.pptx, .md 를 따로 만든다.
형식과 규칙은 team-intro-editing-guide.md 에 있다.
"""
import json
import math
import sys
import re
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

from team_intro_bento import build_bento

HERE = Path(__file__).parent
NAME = "re-rea-team-intro"

# 회사 정보는 저장소에 넣지 않는다. 회사 PC에서 이 세 값을 바꾼 뒤 다시 생성한다.
LOGO, SLOGAN, DEPT = "⟨로고⟩", "⟨슬로건⟩", "⟨부서⟩"

KR, MONO = "맑은 고딕", "Consolas"
RED, ORANGE, BLUE, LBLUE = "D7263D", "F38B1F", "1F5AA6", "E8F0FA"
BLACK, DGRAY, GRAY, LGRAY, WHITE = "262626", "595959", "D9D9D9", "F2F2F2", "FFFFFF"
PINK, YELLOW, ARROW = "FCE4E4", "FFF2CC", "A6A6A6"
X, W = 0.5, 12.33  # 본문 왼쪽 여백과 너비(inch)
BODY_Y, BODY_H, FOOT_H = 2.15, 4.6, 0.62

P1, P2, P3, P4 = "1부 리버스 엔지니어링이란", "2부 작업 절차", "3부 Ghidra와 REA", "4부 회사 PC에서 써 보기"

SETUP = """# 1. 준비물 확인 (REA가 대신 설치해 주지 않는다)
#    Node.js 22.23.4 이상(22.x) 또는 24.22.1 이상(24.x)
#    Ghidra 12.1 이상의 12.x
#    JDK 21 이상 (Ghidra 12.2는 JDK 25 이상)

# 2. REA 설치 (npm: Node.js에 딸린 설치 도구)
npm install --global rea-agents

# 3. Ghidra와 Java의 위치를 알려 준다
$env:GHIDRA_INSTALL_DIR = "C:\\tools\\ghidra_12.1.4_PUBLIC"
$env:JAVA_HOME = "C:\\Program Files\\Java\\jdk-21"
$env:REA_ANALYSIS_PROVIDER = "ghidra"

# 4. 진단
rea doctor --provider ghidra --json

# 5. 에이전트(pi)에 등록하고 확인
rea setup --client pi --dry-run --json
rea setup --client pi --yes --json
rea doctor --client pi --json"""

SLIDES = [
    dict(kind="cover", title="리버스 엔지니어링 입문과 REA 활용법",
         msg="소스 없이 프로그램을 이해하는 방법: 개념, 절차, Ghidra와 REA",
         sub="팀 공유 세미나  |  2026-11",
         notes="오늘은 리버스 엔지니어링이 무엇인지, 그리고 회사 PC에서 Ghidra와 REA로 어떻게 해 볼 수 있는지 소개합니다. "
               "프로그래밍 전공이 아니어도 이해할 수 있게 준비했습니다."),
    dict(kind="draw_agenda", part="시작하며", title="오늘 다룰 내용",
         msg="전공과 상관없이 따라올 수 있도록, 새 용어는 나올 때마다 풀어서 설명합니다",
         items=[("1부  RE란 무엇인가", "무엇이고 왜 필요한가\n꼭 알아야 할 용어\n정적 분석과 동적 분석"),
                ("2부  작업 절차", "허가 확인부터 보고까지 7단계\n시작 전의 안전 수칙\n좋은 질문을 만드는 법"),
                ("3부  Ghidra와 REA", "실행 파일을 읽는 도구 Ghidra\n그 도구를 AI에게 맡기는 REA\nAI 쪽 용어 정리"),
                ("4부  회사 PC에서 쓰기", "Windows에서 되는 것과 안 되는 것\n설치와 연결\n질문하는 법과 결과 읽는 법")],
         time=[("시작", 3), ("1부 개념과 용어", 17), ("2부 절차", 6), ("3부 도구", 8), ("4부 써 보기", 16), ("정리와 질문", 10)],
         md="시간 배분(모두 60분): 시작 3분, 1부 17분, 2부 6분, 3부 8분, 4부 16분, 정리와 질문 10분.",
         notes="네 부분으로 진행합니다. 끝나고 나면 세 가지를 얻어 가시면 됩니다. 리버스 엔지니어링이 어떤 일인지 설명할 수 있고, "
               "분석을 시작하기 전에 무엇을 확인해야 하는지 알고, REA에게 좋은 질문을 쓸 수 있게 됩니다. 한 장에 2분 안쪽으로 진행합니다."),

    # 1부
    dict(kind="draw_reverse", part=P1, title="리버스 엔지니어링이란 무엇인가",
         msg="완성된 결과물에서 거꾸로 설계와 동작을 읽어내는 일입니다",
         nodes=[("설계와 의도", "무엇을 어떻게 만들지 정한 것\n\n비유: 요리 구상"),
                ("소스 코드", "사람이 읽고 쓰는 프로그램 원문\n\n비유: 레시피"),
                ("실행 파일", "우리 손에 있는 완성품 (.exe)\n\n비유: 완성된 요리")],
         md="보통의 개발(Forward)은 설계에서 제품으로 간다. 리버스 엔지니어링(Reverse)은 같은 길을 제품에서 설계 쪽으로 거슬러 올라간다.",
         foot="줄여서 RE라고 부른다. 완성품의 단서로 동작을 **추정**하는 일이다. 원래 설계와 작성 방식까지 똑같이 복원되지는 않는다. "
              "목적은 프로그램을 뚫는 것이 아니라 이해하는 것이다.",
         notes="리버스는 '거꾸로'라는 뜻입니다. 보통은 설계에서 제품으로 가는데, 리버스 엔지니어링은 제품에서 설계 쪽으로 거슬러 올라갑니다. "
               "완성된 요리를 맛보고 재료와 조리 순서를 되짚는 것과 같습니다. 다만 원래 레시피를 그대로 되찾는 것은 아니고, 단서로 추정하는 것입니다. "
               "해킹과 같은 말도 아닙니다. 사내 도구가 실제로 무엇을 하는지 이해하려는 일입니다."),
    dict(kind="draw_why", part=P1, title="왜 우리에게 필요한가",
         msg="상황은 달라도 답해야 할 질문은 하나입니다",
         question="이 프로그램은 실제로 무엇을 하는가?",
         md="네 상황의 공통 질문: \"이 프로그램은 실제로 무엇을 하는가?\" 리버스 엔지니어링은 이 질문에 근거를 붙여 답하는 방법이다.",
         items=[("소스가 없다", "만든 사람이 떠났거나, 저장소 없이 실행 파일만 공유 폴더에 남았다"),
                ("소스와 배포본이 다르다", "저장소의 코드와 실제로 돌고 있는 프로그램이 같은 버전인지 아무도 모른다"),
                ("연동해야 한다", "웹 앱이 어떤 주소로 어떤 형식의 데이터를 주고받는지 문서가 없다"),
                ("새로 만들어야 한다", "기존 도구의 계산식과 판정 기준을 그대로 옮겨야 한다")],
         foot="소스 코드(Source Code): 사람이 작성한 프로그램 원문.  배포본: 사용자 PC에 실제로 설치되어 돌고 있는 프로그램.  "
              "저장소: 소스 코드를 보관하는 곳.",
         notes="사내 도구는 만든 사람이 자리를 옮기면 실행 파일만 남는 경우가 많습니다. 고치거나, 다른 시스템과 잇거나, 새로 만들려면 "
               "그 프로그램이 실제로 무엇을 하는지부터 알아야 합니다. 리버스 엔지니어링은 그 질문에 근거를 붙여 답하는 방법입니다."),
    dict(kind="draw_chain", part=P1, title="용어 1: 프로그램은 어떻게 만들어지는가",
         msg="사람이 읽는 글(소스 코드)을 컴퓨터가 읽는 숫자(실행 파일)로 바꾼 것이 프로그램입니다",
         nodes=[("소스 코드 (Source Code)", "사람이 읽고 쓰는 글. 이름, 설명(주석), 구조가 모두 들어 있다.\n비유: 레시피",
                 "if (value > 100)\n    return -1;\nreturn value * 2;"),
                ("컴파일 (Compile)", "소스 코드를 기계어로 번역하는 과정. 번역하는 프로그램이 컴파일러다.\n비유: 조리",
                 "소스 코드  →  기계어"),
                ("실행 파일 (Binary)", "컴퓨터가 실행하는 숫자 묶음. Windows에서는 .exe, .dll 파일.\n비유: 완성된 요리",
                 "8d 14 09 83 f9 65\nb8 ff ff ff ff\n0f 4c c2 c3")],
         back="리버스 엔지니어링은 이 화살표를 거꾸로 거슬러 올라간다",
         foot="이 그림은 C, C++처럼 기계어로 바로 만드는(네이티브) 프로그램의 예다. 웹, .NET, Python 프로그램은 남는 정보와 분석 방법이 다르다(12장).",
         notes="용어부터 정리하겠습니다. 개발자가 쓰는 글이 소스 코드이고, 그것을 컴퓨터가 알아듣는 숫자로 번역하는 과정이 컴파일입니다. "
               "결과물이 우리가 더블클릭하는 exe 파일입니다. 우리 손에는 보통 오른쪽 끝의 실행 파일만 있습니다. "
               "왼쪽의 세 줄짜리 코드는 뒤에서 계속 예시로 씁니다. 값이 100보다 크면 -1, 아니면 2배를 돌려주는 코드입니다."),
    dict(kind="table", part=P1, title="컴파일은 정보를 버린다",
         msg="사라진 정보가 많을수록 분석이 어렵습니다. 그래서 남은 단서에서 출발합니다",
         head=["소스 코드에 있던 것", "실행 파일에서는", "분석할 때 보이는 모습과 쓰임"], widths=[3.7, 1.9, 6.7],
         tint={"사라짐": PINK, "바뀜": YELLOW, "남음": LBLUE},
         rows=[["변수 이름, 설명(주석)", "사라짐", "local_18, param_1 같은 기계적인 이름으로 보인다"],
               ["함수 이름", "자주 사라짐", "FUN_00401a20처럼 위치(주소)로만 보인다"],
               ["값의 종류(타입)", "자주 사라짐", "숫자 계산과 위치 계산으로만 보인다"],
               ["코드의 원래 모양", "바뀜", "더 빨리 돌도록 고쳐져서(최적화) 조건문과 반복문이 달라진다"],
               ["화면 문구, 에러 메시지(문자열)", "남음", "분석의 가장 좋은 출발점이 된다"],
               ["외부 기능 호출 목록(임포트)", "남음", "파일, 네트워크 등 무엇을 쓰는지 짐작하는 단서"],
               ["실제 동작(기계어)", "남음", "도구로 다시 글로 바꿔서 읽는다"]],
         foot="비유: 이름표가 떨어진 서류함. 내용은 남아 있지만 각 칸의 역할은 찾아봐야 안다.  배포 설정에 따라 이름과 타입 정보가 남기도 한다. "
              "남아 있으면 더 많이 알 수 있다.  함수(Function): 한 가지 일을 하는 코드 묶음.",
         notes="컴퓨터는 변수 이름이나 주석이 필요 없기 때문에 번역할 때 버립니다. 그래서 실행 파일을 다시 글로 바꿔도 이름이 사라져 있습니다. "
               "대신 화면에 보이는 문구나 에러 메시지 같은 글자는 그대로 남아서, 분석의 출발점이 됩니다. 빨간 칸이 잃는 것, 파란 칸이 남는 것입니다."),
    dict(kind="draw_chain", part=P1, title="같은 함수의 세 가지 모습",
         msg="의사 코드로 빨리 뜻을 파악하고, 중요한 판단은 어셈블리로 확인합니다",
         nodes=[("기계어 (Machine Code)", "파일 안에 실제로 들어 있는 숫자. 이것이 원본이다.",
                 "8d 14 09 83 f9 65\nb8 ff ff ff ff\n0f 4c c2 c3"),
                ("어셈블리 (Assembly)", "숫자에 명령어 이름을 붙인 표기. 정확하지만 읽기 어렵다.",
                 "lea   edx, [rcx + rcx]\ncmp   ecx, 0x65\nmov   eax, 0xffffffff\ncmovl eax, edx\nret"),
                ("의사 코드 (Pseudocode)", "읽기 좋게 다시 쓴 추정본. 원본이 아니다.\nparam_1: 첫 입력값, 0x65: 101",
                 "iVar1 = -1;\nif (param_1 < 0x65) {\n  iVar1 = param_1 * 2;\n}\nreturn iVar1;")],
         labels=["디스어셈블", "디컴파일"],
         back="왼쪽일수록 정확하다 (원본 그대로)          오른쪽일수록 읽기 쉽지만 추정이 섞인다", back_shape="both",
         foot="앞 장의 소스 \"value가 100보다 크면 -1, 아니면 2배\"와 같은 동작이다. 입력 100 → 200, 입력 101 → -1. "
              "이름 value는 param_1이 되었고, \"100보다 크다\"는 \"101보다 작다\"의 반대 모양으로 바뀌었다.",
         notes="같은 함수 하나를 세 가지 높이에서 본 모습입니다. 왼쪽이 파일에 실제로 든 숫자이고, 가운데가 그 숫자에 이름을 붙인 어셈블리, "
               "오른쪽이 도구가 사람이 읽기 좋게 다시 써 준 의사 코드입니다. 오른쪽으로 갈수록 읽기 쉽지만 도구의 추정이 섞입니다. "
               "어셈블리를 읽을 줄 몰라도 됩니다. 의사 코드가 원본이 아니라는 점만 기억하시면 됩니다."),
    dict(kind="table", part=P1, title="용어 2: 분석할 때 쓰는 말",
         msg="용어는 많아 보여도, 모두 \"단서를 찾고 따라가는\" 동작에 붙인 이름입니다",
         head=["용어", "쉬운 뜻", "비유"], widths=[2.9, 6.0, 3.4],
         rows=[["디스어셈블 (Disassemble)", "기계어 숫자를 어셈블리 명령어로 바꾼다", "모스 부호를 글자로 옮겨 적기"],
               ["디컴파일 (Decompile)", "남은 단서를 읽기 쉬운 코드로 다시 쓴다. 원문과 같다는 보장은 없다", "요리를 맛보고 레시피 다시 쓰기"],
               ["심볼 (Symbol)", "함수와 변수의 이름 정보. 배포할 때 지워지는 경우가 많다", "서류함 칸에 붙은 이름표"],
               ["임포트 (Import)", "프로그램이 Windows 같은 외부에서 빌려 쓰는 기능의 목록", "요리에 쓴 시판 소스 목록"],
               ["문자열 (String)", "프로그램 안에 든 글자. 화면 문구, 에러 메시지, 파일 경로", "제품에 인쇄된 문구"],
               ["상호 참조 (Xref)", "\"이 문자열과 함수를 누가 쓰는가\"의 목록", "책 뒤의 찾아보기(색인)"],
               ["진입점 (Entry Point)", "분석을 시작하는 지점. 버튼, 에러 문구, 화면 이름", "미로의 입구"]],
         notes="일곱 개를 다 외울 필요는 없습니다. 다음 장에서 이 용어들이 실제로 어떤 순서로 쓰이는지 그림으로 보겠습니다. "
               "임포트는 프로그램이 파일을 여는지, 네트워크를 쓰는지 짐작하게 해 주는 단서입니다. "
               "다만 임포트에 있다고 해서 실제로 호출했다는 증명은 아닙니다."),
    dict(kind="draw_chain", part=P1, title="단서를 따라가는 길: 화면 문구에서 코드까지",
         msg="눈에 보이는 문구 하나가 가장 좋은 출발점입니다",
         nodes=[("1  화면", "눈에 보이는 버튼이나 문구에서 시작한다", "[ Export ]"),
                ("2  문자열", "그 문구를 파일 안에서 찾는다", "\"Export completed\""),
                ("3  상호 참조", "그 문자열을 쓰는 함수를 찾는다", "누가 쓰는가?"),
                ("4  함수 (진입점)", "이름이 없어 주소로 불린다. 여기서부터 읽는다", "FUN_140012340"),
                ("5  의사 코드", "디컴파일해서 읽고, 여기서 부르는 함수를 따라간다", "if (...) { ... }")],
         gap=0.55, sample_size=11, ch=2.45, back="REA도 속으로는 이 순서로 움직인다. 사람은 출발점(문구)을 알려 주면 된다", back_shape="box",
         foot="주소와 문구는 설명용 예시다.  임포트도 출발점이 된다. 예: 파일을 여는 Windows 기능을 빌려 쓰는 곳을 찾는다.",
         notes="가장 흔한 탐색 방법입니다. 화면에 보이는 문구를 문자열에서 찾고, 상호 참조로 그 문구를 쓰는 함수를 찾고, "
               "그 함수를 디컴파일해서 읽습니다. 앞 장의 용어가 이 순서로 쓰입니다. 뒤에서 볼 REA도 속으로는 이 순서로 움직입니다."),
    dict(kind="draw_loop", part=P1, title="정적 분석과 동적 분석",
         msg="정적 분석으로 후보를 좁히고, 동적 분석으로 확인하는 순서가 기본입니다",
         groups=[("정적 분석 (Static): 실행하지 않고 읽는다", ["비유: 설계 도면을 펴 놓고 읽기", "하는 일: 디컴파일, 문자열 검색, 호출 관계 추적",
                                                    "장점: 안전하고 몇 번이든 반복할 수 있다", "한계: 실제로 어느 경로가 실행되는지는 모른다"]),
                 ("동적 분석 (Dynamic): 실행하면서 본다", ["비유: 장비를 돌려 보면서 계측하기", "하는 일: 화면 조작, 통신 기록, 출력 파일 확인",
                                                  "장점: 실제 값을 볼 수 있다", "한계: 실행해 본 경로만 보인다. 운영 환경에서는 하지 않는다"])],
         md="두 분석은 번갈아 쓴다. 정적 분석에서 가설을 세워 동적 분석으로 확인하고, 결과가 다르면 다시 읽는다.",
         banner="읽어서 가설을 세우고, 실행해서 확인한다. 결과가 가설과 다르면 실패가 아니라 정보다",
         notes="도면만 봐서는 실제로 그 경로가 쓰이는지 알 수 없고, 돌려 보기만 해서는 돌려 본 경우만 알 수 있습니다. 그래서 둘을 같이 씁니다. "
               "읽어서 가설을 세우고, 실행해서 확인합니다. 결과가 가설과 다르면 실패가 아니라 정보입니다."),
    dict(kind="draw_evidence", part=P1, title="증거의 등급: 관찰, 추론, 미확인",
         msg="보고서가 틀리는 가장 흔한 이유는 추론을 관찰처럼 쓰는 것입니다",
         nodes=[("관찰 (Observation)", "도구가 직접 본 사실\n\n예: 이 함수가 문자열 'Recipe saved'를 참조한다"),
                ("추론 (Inference)", "관찰을 묶어서 내린 해석\n\n예: 이 함수가 레시피 저장 함수일 것이다"),
                ("실행해서 확인", "테스트 환경에서 저장을 실행해 본다"),
                ("관찰 (Observation)", "확인된 사실\n\n예: 저장할 때 이 함수가 호출되었다")],
         unknown="**미확인 (Unknown)**: 도구가 볼 수 없었던 것.  예: 간접 호출이라 누가 이 함수를 부르는지 실행 전에는 찾지 못했다",
         md="미확인(Unknown): 도구가 볼 수 없었던 것. 예: 간접 호출이라 누가 이 함수를 부르는지 실행 전에는 찾지 못했다.",
         foot="디컴파일 결과를 읽고 \"이 함수는 A를 한다\"고 쓰면 추론이다. 실행해서 A가 일어나는 것을 봐야 관찰이 된다. "
              "REA는 결과를 이 세 가지로 나눠서 돌려준다.  간접 호출: 부를 대상이 실행 중에 정해지는 호출.",
         notes="실험 보고서에서 측정값과 해석을 구분하는 것과 같습니다. 그림은 한 가지 사례입니다. 문자열 참조를 본 것은 관찰, "
               "그래서 저장 함수일 것이라는 것은 추론, 실행해서 확인하면 다시 관찰이 됩니다. "
               "REA도 결과를 이 세 가지로 나눠서 주기 때문에, 기록할 때 이 구분을 지우면 안 됩니다."),
    dict(kind="draw_zones", part=P1, title="대상에 따라 난이도가 다르다",
         msg="무엇으로 만들었는지에 따라 남는 정보가 다릅니다. 쉬운 대상부터 시작합니다",
         groups=[("쉬움", ["**웹 앱의 화면 쪽**: 화면 쪽 코드가 이미 브라우저에 내려와 있다",
                        "**웹 기술로 만든 설치형 앱 (Electron)**: 압축을 풀면 웹 앱과 같다",
                        "**일반 .NET 프로그램 (C#)**: 클래스와 메서드 이름이 남는다"]),
                 ("보통", ["**Python 프로그램을 exe로 묶은 것 (PyInstaller)**: 내부 파일 목록이 남는다. 소스 복원은 별도 도구",
                        "**Go, Rust 프로그램**: 모듈과 의존성 정보가 남는다"]),
                 ("어려움", ["**C, C++ 프로그램 (네이티브)**: 기계어, 임포트, 문자열만 남는다. Ghidra가 필요하다",
                         "**웹 앱의 서버 쪽**: 밖에서는 요청과 응답만 보인다. 접근 권한을 받는 편이 빠르다"])],
         foot="소스 맵(Source Map): 압축된 웹 코드를 원본 파일과 줄 번호로 되돌리는 지도 파일. 원본 내용이 들어 있으면 원본에 가깝게 읽을 수 있다.  "
              "네이티브(Native): CPU가 바로 실행하는 기계어로 만든 프로그램.",
         notes="같은 프로그램이라도 만든 언어에 따라 난이도가 크게 다릅니다. 웹 앱은 코드가 이미 브라우저에 내려와 있어서 쉽습니다. "
               "C나 C++로 만든 프로그램이 가장 어렵고, 이때 Ghidra가 필요합니다. 서버 쪽은 접근 권한을 받는 것이 리버스 엔지니어링보다 빠릅니다."),

    # 2부
    dict(kind="draw_steps", part=P2, title="작업 절차 7단계",
         msg="코드를 읽기 전에 허가, 질문, 대상부터 고정합니다",
         nodes=[("범위와 허가", "소유자, 목적, 동의, 실행 환경 확인"), ("질문 정하기", "답할 수 있는 크기로 쪼갠다"),
                ("대상 식별", "복사하고 해시를 기록한다"), ("정찰", "문자열, 임포트에서 진입점 후보 찾기"),
                ("정적 분석", "진입점에서 답까지 경로를 따라간다"), ("동적 확인", "테스트 환경에서 실행해 확인한다"),
                ("기록과 보고", "관찰, 추론, 미확인을 구분해 남긴다")],
         md="1~3단계는 준비, 4~6단계는 분석, 7단계는 정리다. 6단계 결과가 가설과 다르면 5단계로 돌아간다.",
         foot="해시(Hash): 파일의 지문. 내용이 한 글자만 달라도 값이 바뀐다. \"그 파일이 이 파일인가\"를 확인하는 데 쓴다.",
         notes="새 분석을 시작할 때마다 이 순서를 따릅니다. 앞의 세 단계는 코드를 한 줄도 읽지 않는 단계인데, 여기를 건너뛰면 "
               "엉뚱한 파일을 분석하거나 끝이 없는 분석을 하게 됩니다. 폴더에 분석 기록 양식이 있으니 복사해서 단계마다 채우면 됩니다."),
    dict(kind="cards", part=P2, title="시작 전에: 허가와 안전",
         msg="동료가 만든 도구라도 바로 분석하지 않습니다",
         items=[("허가 범위 안에서만", "소유 부서와 목적을 먼저 확인한다.\n\n만든 사람에게 연락이 되면 묻는 편이 더 빠르다.\n\n"
                                "외부 상용 소프트웨어는 라이선스가 분석을 금지할 수 있다."),
                ("운영 환경에서 실행하지 않는다", "장비 제어, DB 쓰기, 파일 삭제를 하는 도구는 분석 중에도 그 동작을 실제로 한다.\n\n"
                                      "REA는 실행 환경을 격리해 주지 않는다. 운영망, 운영 계정 정보, 공유 폴더가 연결되지 않은 테스트 환경에서만 실행한다."),
                ("회사 밖으로 내보내지 않는다", "분석은 내 PC에서 하지만 결과는 AI 모델로 전달된다.\n\n"
                                         "사내 LLM에 연결된 에이전트에서만 쓴다. 외부 AI 서비스에 붙여 넣지 않는다.")],
         banner="확인한 내용(소유자, 목적, 허가, 실행 환경)은 분석 기록 양식에 먼저 적고 시작한다",
         notes="세 가지를 먼저 확인합니다. 첫째는 허가입니다. 둘째는 실행 환경입니다. 분석하려고 실행한 도구가 실제 장비나 운영 DB를 건드릴 수 있습니다. "
               "셋째는 반출입니다. 디컴파일한 코드나 경로 같은 결과가 AI 모델로 넘어가기 때문에 사내 LLM에 연결된 에이전트만 씁니다."),
    dict(kind="two", part=P2, title="좋은 질문이 분석의 절반이다",
         msg="\"분석해줘\"는 질문이 아닙니다. 시작점, 알고 싶은 결과, 끝낼 기준을 넣습니다",
         groups=[("나쁜 질문", ["\"이 도구 분석해줘\"", "\"이 웹 앱 API 알려줘\"", "\"이 계산 로직 뭐야\""]),
                 ("좋은 질문", ["\"Export 버튼을 누르면 어떤 파일이 어떤 형식으로 만들어지는가?\"",
                            "\"목록 화면이 열릴 때 어떤 주소를 어떤 조건으로 호출하는가?\"",
                            "\"결과 화면의 보정값은 어떤 입력으로 어떻게 계산되는가?\""])],
         foot="질문마다 \"어떤 증거가 있으면 답이 되는가\"를 같이 적는다. 예: 파일을 쓰는 함수의 디컴파일 결과와 실제로 만들어진 파일 한 개.  "
              "API: 프로그램끼리 데이터를 주고받는 창구.",
         notes="좋은 질문에는 세 가지가 들어 있습니다. 버튼이나 화면처럼 눈으로 볼 수 있는 시작점, 파일 형식이나 계산식처럼 알고 싶은 결과, "
               "그리고 여기까지 알면 끝이라는 기준입니다. 사람에게 일을 맡길 때와 똑같습니다."),

    # 3부
    dict(kind="two", part=P3, title="Ghidra: 실행 파일을 읽는 도구",
         msg="Ghidra(기드라)는 미국 NSA가 공개한 무료 분석 도구입니다. 회사 PC에 설치할 수 있습니다",
         groups=[("Ghidra가 해 주는 일", ["실행 파일을 열어 함수 목록을 만든다", "기계어를 어셈블리로 바꾼다 (디스어셈블)",
                                     "어셈블리를 의사 코드로 바꾼다 (디컴파일)", "문자열, 임포트, 상호 참조를 찾아 준다"]),
                 ("직접 쓰면 어려운 점", ["화면과 메뉴가 많아 익히는 데 시간이 든다", "프로젝트 생성, 파일 가져오기, 자동 분석 대기를 거쳐야 한다",
                                   "함수 수십 개를 클릭으로 따라가면 반나절이 걸린다", "결과를 읽으려면 C 언어와 어셈블리 지식이 필요하다"])],
         banner="REA는 오른쪽의 손작업을 AI 에이전트에게 넘긴다 (다음 장)",
         notes="Ghidra는 앞에서 본 디스어셈블과 디컴파일을 해 주는 프로그램입니다. 무료이고 공개 소프트웨어입니다. "
               "다만 전문가용 도구라서 화면이 복잡하고, 손으로 따라가는 작업이 많습니다. 이 손작업을 덜어 주는 것이 다음 장의 REA입니다."),
    dict(kind="draw_arch", part=P3, title="REA: Ghidra를 AI에게 맡긴다",
         msg="사람은 질문하고, AI 에이전트가 REA를 통해 Ghidra를 대신 조작합니다",
         md="1. **사람**: 질문을 고르고 결과의 등급을 따진다.\n"
            "2. **AI 에이전트 (pi)**: 질문을 받아 어떤 도구를 쓸지 정한다. 사내 LLM(AI 모델)과 연결되어 있고, 도구 결과가 모델로 전달된다.\n"
            "3. **REA**: 스킬(작업 지침서)과 MCP 서버(분석 도구 묶음). 에이전트가 MCP로 도구를 부르면 근거와 한계가 붙은 결과를 돌려준다.\n"
            "4. **분석 엔진**: Ghidra(C, C++ 프로그램), JavaScript와 Electron 분석기, .NET과 Python 분석기, Chrome(웹 페이지 관찰).\n\n"
            "REA와 분석 엔진은 내 PC에서 실행된다.",
         foot="REA(Reverse Engineer Anything): 공개 도구(github.com/morluto/rea). 분석은 내 PC에서 실행되고, "
              "결과는 에이전트가 쓰는 AI 모델로 전달된다. 그래서 사내 LLM에 연결된 에이전트만 쓴다.",
         notes="REA는 AI 에이전트에 붙여 쓰는 분석 도구 묶음입니다. 우리가 글로 질문하면 에이전트가 REA의 도구를 부르고, REA가 Ghidra를 화면 없이 돌려서 "
               "결과를 돌려줍니다. 중요한 점은 REA가 결과마다 근거와 한계를 붙인다는 것입니다. AI가 근거 없이 지어내는 것을 막는 장치입니다. "
               "REA는 Ghidra 전용이 아니어서 웹 앱이나 .NET, Python 프로그램은 Ghidra 없이 분석합니다."),
    dict(kind="table", part=P3, title="용어 3: AI와 REA 쪽 말",
         msg="REA 스킬은 '지침서', MCP는 '연결선', Ghidra는 '엔진'으로 기억하면 됩니다",
         head=["용어", "쉬운 뜻", "비유"], widths=[2.7, 6.6, 3.0],
         rows=[["AI 에이전트 (Agent)", "지시를 받으면 도구를 스스로 골라 여러 단계의 일을 하는 AI 프로그램. 회사에서는 pi", "일을 맡길 수 있는 조수"],
               ["MCP", "에이전트가 외부 도구를 쓰기 위한 연결 규격 (Model Context Protocol)", "장비와 PC를 잇는 표준 커넥터"],
               ["REA 스킬 (Skill)", "에이전트에게 주는 작업 지침서. 어떤 순서로 어떤 도구를 쓸지 적혀 있다", "작업 표준서"],
               ["프로바이더 (Provider)", "네이티브 실행 파일을 분석하는 엔진. Ghidra, Hopper, IDA 중 우리는 Ghidra", "계측 장비 본체"],
               ["세션 (Session)", "대상을 열고 닫을 때까지의 작업 한 묶음. 닫기 전에 근거를 파일로 저장한다", "장비 한 번 가동"],
               ["Evidence", "도구 결과 한 건의 기록. 번호(Evidence ID)로 가리킨다", "측정 기록지"]],
         notes="REA를 설치하면 두 가지가 함께 들어옵니다. 하나는 실제 분석 도구 묶음인 MCP 서버이고, 다른 하나는 에이전트가 그 도구를 어떻게 쓸지 "
               "적어 둔 지침서인 스킬입니다. 스킬 이름은 reverse-engineer-anything입니다. 우리는 스킬을 직접 읽을 필요는 없고, 에이전트가 읽고 따릅니다."),
    dict(kind="two", part=P3, title="직접 할 때와 REA로 할 때",
         msg="손작업은 에이전트에게 넘기고, 사람은 질문과 판단에 집중합니다",
         groups=[("Ghidra를 직접 쓸 때", ["프로젝트를 만들고 파일을 가져온다", "자동 분석이 끝나기를 기다린다",
                                     "함수 목록과 디컴파일 창을 오가며 클릭한다", "읽은 내용을 손으로 정리한다"]),
                 ("REA를 쓸 때", ["대상 경로와 질문을 글로 적는다", "에이전트가 필요한 함수만 찾아서 읽는다",
                               "결론마다 근거 번호와 등급이 붙어서 온다", "사람은 결과를 검토하고 다음 질문을 정한다"])],
         foot="질문에 답할 소스가 온전히 있으면 소스부터 읽는다. REA는 배포된 결과물에서만 확인할 수 있는 사실이 필요할 때 쓴다. "
              "예: 저장소의 코드와 배포된 .exe가 같은지 확인할 때.",
         notes="REA를 쓴다고 사람이 할 일이 없어지는 것은 아닙니다. 클릭하는 일이 없어지고, 질문을 고르는 일과 결과를 따지는 일이 남습니다. "
               "그리고 소스가 온전히 있으면 소스부터 읽습니다. 실제 배포본이 소스와 같은지 확인하고 싶을 때가 REA를 쓸 때입니다."),

    # 4부
    dict(kind="table", part=P4, title="회사 PC(Windows)에서 되는 것과 안 되는 것",
         msg="Windows의 Ghidra 연동은 실험 단계이고 읽기 전용입니다. 사내 도구 분석에는 충분합니다",
         head=["하려는 일", "Windows", "알아둘 점과 대처"], widths=[4.6, 1.5, 6.2], tint={"안 됨": PINK, "됨": LBLUE},
         rows=[[".exe, .dll 분석 (Ghidra)", "됨", "읽기 전용. 대상 파일은 로컬 디스크(C:\\re-work 등)에 복사해 둔다"],
               ["일반 .NET 프로그램 분석", "됨", "Ghidra 없이 별도 분석기로 본다"],
               ["Python을 exe로 묶은 프로그램", "됨", "내부 목록과 구조를 본다. 원래 Python 소스로 되돌려 주지는 않는다"],
               ["Electron 앱, 웹 페이지 관찰, HAR 분석", "됨", "Chrome을 분석 전용 프로필로 띄운다"],
               ["REA로 함수에 이름, 주석 붙이기", "안 됨", "에이전트에게 주소와 제안 이름을 표로 정리하게 해서 기록에 남긴다"],
               ["설치형 프로그램의 실행 관찰", "안 됨", "사람이 테스트 환경에서 직접 실행한다. 웹 페이지 관찰은 된다"],
               ["공유 폴더, USB의 파일을 Ghidra로 분석", "안 됨", "고정 로컬 디스크(NTFS)로 복사한 뒤 분석한다"]],
         foot="HAR: 브라우저의 통신 기록 파일. 로그인 상태를 유지하는 인증 정보가 들어 있을 수 있으므로 분석이 끝나면 지운다.",
         notes="REA 문서는 맥과 리눅스 기준으로 쓰인 부분이 많아서, Windows에서 안 되는 기능이 몇 가지 있습니다. "
               "실제로 아쉬운 것은 함수에 이름을 붙이는 기능 하나입니다. 에이전트에게 따라간 함수의 주소와 제안 이름을 표로 정리하라고 하면 됩니다. "
               "그리고 분석할 파일은 반드시 로컬 디스크에 복사해 둡니다."),
    dict(kind="code", part=P4, title="설치와 연결 순서",
         msg="새로 설치할 때는 사내 설치 금지 목록과 라이선스를 먼저 확인합니다",
         code=SETUP,
         side=["**PowerShell**: Windows의 명령 입력 창. 여기에 왼쪽 명령을 입력한다",
               "**JDK**: Ghidra를 실행하는 데 필요한 Java 도구",
               "**환경 변수**: 프로그램에게 알려 주는 설정값. 여기서는 Ghidra와 Java가 설치된 위치",
               "**--dry-run**: 바뀔 내용만 미리 보여 주고 아무것도 바꾸지 않는다",
               "등록한 뒤 pi에서 **/reload**를 실행한다",
               "경로는 실제 설치 위치로 바꾼다. 자세한 내용은 rea-windows-pi-setup.md"],
         notes="한 번만 하면 되는 작업이고, 오늘 외울 내용은 아닙니다. 이런 순서라는 것만 보시면 됩니다. 준비물 세 가지는 REA가 대신 설치해 주지 않으니 먼저 있어야 합니다. "
               "rea doctor는 건강 검진 같은 명령입니다. 문제가 있으면 어디가 문제인지 알려 줍니다. 막히면 설정 문서를 보거나 저에게 물어보시면 됩니다."),
    dict(kind="draw_prompt", part=P4, title="프롬프트에 넣을 네 가지",
         msg="프롬프트(AI에게 보내는 지시문)의 질이 결과의 질을 정합니다. 이 틀을 복사해서 씁니다",
         nodes=[("대상", "파일의 전체 경로. \"내 앱\"처럼 모호하게 쓰지 않는다",
                 "대상: C:\\re-work\\recipe-tool\\RecipeTool.exe"),
                ("질문", "눈에 보이는 시작점과 알고 싶은 결과. 진입점 후보를 주면 빨라진다",
                 "질문: \"Export\" 버튼을 누르면 만들어지는 CSV의\n      열 구성과 각 열의 계산 방법을 알고 싶다.\n진입점 후보: 문자열 \"Export completed\", \"csv\"."),
                ("실행 범위", "읽기만 할지, 실행해도 되는지. 적지 않으면 에이전트가 프로그램을 실행할 수 있다",
                 "방법: 정적 분석만 한다. 프로그램을 실행하지 않는다.\n      프로바이더는 ghidra."),
                ("근거 요구", "결론마다 근거 번호와 등급. 못 찾은 부분도 보고하게 한다",
                 "결과 형식: 결론마다 Evidence ID와\n          관찰/추론/미확인 구분을 붙여라.\n          확인하지 못한 부분은 어디까지 찾아봤는지 적어라.")],
         md="나쁜 예는 \"이 exe 분석해줘\"다. 에이전트가 함수 목록부터 훑기 시작하고 결과는 넓고 얕다. CSV: 쉼표로 열을 나눈 표 파일.",
         notes="에이전트에게 보내는 글에 이 네 가지를 넣습니다. 왼쪽이 실제 예시이고 오른쪽이 각 부분의 역할입니다. 특히 세 번째가 중요합니다. "
               "정적 분석만 하라고 적지 않으면 에이전트가 프로그램을 실행하는 도구를 고를 수 있습니다. "
               "나쁜 예는 '이 exe 분석해줘'입니다. 결과가 넓고 얕게 나옵니다. 질문이 여러 개면 나눠서 묻습니다."),
    dict(kind="draw_tree", part=P4, title="대상별로 REA가 가는 길",
         msg="Ghidra가 필요한 것은 기계어로 바로 만든(네이티브) 프로그램입니다",
         root="설치 폴더를 받았다. 무엇으로 만들었는지 모른다",
         check="먼저 형식을 판별한다  (inspect_desktop_application)",
         nodes=[("네이티브 프로그램\n(C, C++ 등)", "**Ghidra 필요**", "open_binary\n→ analyze_function"),
                ("일반 .NET 프로그램\n(C#)", "Ghidra 불필요", "inspect_managed_artifact"),
                ("Python을 exe로 묶은 것\n(PyInstaller)", "Ghidra 불필요", "open_binary\n→ inspect_artifact"),
                ("웹 기술로 만든 설치형 앱\n(Electron)", "Ghidra 불필요", "analyze_javascript_\napplication")],
         foot="도구 이름은 외울 필요가 없다. 에이전트가 고른다.  브라우저에 열린 웹 앱은 Chrome 관찰 도구(inspect_web_page)로 본다.  "
              ".NET이라도 기계어로 빌드한 것(NativeAOT)은 네이티브 경로로 간다.",
         notes="에이전트는 대상의 형식을 보고 첫 도구를 고릅니다. 우리가 알아 둘 것은 하나입니다. 설치 폴더를 받았는데 무엇으로 만든 프로그램인지 모르면 "
               "폴더째 넘기면 REA가 판별해 줍니다. exe 하나가 아니라 폴더째 넘기는 이유는 DLL, 설정, 리소스가 함께 있어야 실제 동작을 설명할 수 있기 때문입니다. "
               "Ghidra까지 가는 것은 맨 왼쪽 길뿐입니다."),
    dict(kind="draw_native", part=P4, title="네이티브 .exe 분석 흐름",
         msg="넓게 훑은 뒤 한 곳만 깊게 봅니다. 닫기 전에 근거를 저장합니다",
         nodes=[("정찰", "문자열과 임포트로 관련 함수 후보를 찾는다. 아직 디컴파일하지 않는다"),
                ("진입점 선택", "질문과 가장 가까운 함수 하나를 고른다"),
                ("깊게 읽기", "그 함수만 분석한다. 요약부터 받고 필요한 부분만 펼친다"),
                ("어셈블리 확인", "결과가 이상하면 디컴파일러를 거치지 않고 확인하게 한다"),
                ("근거 저장", "export_evidence_bundle로 근거를 파일에 저장한다"),
                ("세션 닫기", "close_binary. 현재 배포본(6.8.0)은 닫을 때 세션 기록을 지운다")],
         foot="큰 파일은 Ghidra가 처음 여는 데 몇 분이 걸린다. REA가 Ghidra 시작을 기다리는 기본 시간은 330초다.",
         notes="C나 C++ 프로그램을 분석할 때의 순서입니다. 왼쪽 깔때기처럼 처음에는 전체에서 후보를 좁히고, 함수 하나를 고른 뒤에야 깊게 읽습니다. "
               "그리고 세션을 닫으면 근거 기록이 지워질 수 있으니 닫기 전에 꼭 파일로 저장하게 합니다."),
    dict(kind="draw_result", part=P4, title="결과를 읽는 법",
         msg="에이전트의 설명을 그대로 믿지 않습니다. 세 가지를 확인합니다",
         rows=[("관찰", "함수 FUN_140012340이 문자열 \"Export completed\"를 참조한다  [Evidence ID]", LBLUE),
               ("추론", "이 함수가 CSV 파일을 저장하는 함수일 것이다", YELLOW),
               ("미확인", "간접 호출이라 누가 이 함수를 부르는지 찾지 못했다", GRAY),
               ("한계", "함수 목록의 일부 페이지만 읽었다  (커버리지: 일부)", PINK)],
         items=[("1  무엇을 못 봤는가", "한계와 미확인부터 읽는다. 간접 호출, 잘린 결과가 여기에 적힌다"),
                ("2  전체인가, 일부인가", "커버리지(Coverage)를 본다. 일부만 보고 \"없다\"고 결론 내리지 않는다"),
                ("3  어느 문장이 추론인가", "추론은 실행해서 확인할 대상이다")],
         md="REA 결과에는 관찰, 추론, 미확인, 한계가 나뉘어 적힌다(슬라이드의 예시는 설명용으로 단순화한 것이다).",
         foot="남은 질문이 있으면 그 질문 하나만 좁혀서 다시 묻는다. 같은 분석을 그대로 다시 돌리지 않는다.",
         notes="AI가 쓴 설명은 그럴듯하게 읽힙니다. 그래서 결과를 받으면 먼저 무엇을 못 봤는지부터 확인합니다. "
               "REA는 한계와 커버리지를 결과에 같이 적어 주니 그 부분을 건너뛰지 않습니다. 왼쪽은 형식을 단순하게 보인 예시입니다."),
    dict(kind="draw_uses", part=P4, title="이런 일에 쓰면 좋다",
         msg="배포된 결과물에서만 확인할 수 있는 사실이 필요할 때 씁니다",
         items=[("버전 비교", "지난달 배포본과 이번 배포본은 무엇이 다른가"),
                ("계산 로직 옮기기", "기존 도구의 계산식을 새 코드로 옮긴다"),
                ("연동 정보 파악", "웹 앱이 어떤 주소와 형식으로 데이터를 주고받는가"),
                ("인수인계 없는 도구 파악", "이 도구가 어떤 파일과 설정을 읽고 쓰는가")],
         md="계산 로직을 옮길 때의 검증: 같은 입력(정상값, 경계값인 0과 음수와 아주 큰 값, 빈 입력)을 원본 도구와 새로 만든 코드에 넣고 "
            "출력을 비교한다. 시험한 입력에서 같으면 통과이고(프로그램 전체가 같다는 뜻은 아니다), 다르면 다시 분석한다.",
         notes="우리 팀에서 쓸 만한 경우 네 가지입니다. 특히 계산 로직을 옮길 때는 아래 그림처럼 옮긴 코드와 원본 도구에 같은 입력을 넣어 비교하는 것까지 해야 합니다. "
               "0, 음수, 빈 값 같은 경계값도 꼭 넣습니다."),
    dict(kind="table", part=P4, title="자주 만나는 문제",
         msg="막히면 먼저 rea doctor로 진단하고, 그다음 rea update로 최신 버전을 받습니다",
         head=["증상", "원인", "해결"], widths=[3.2, 4.2, 4.9],
         rows=[["pi에 REA 도구가 안 보인다", "등록이 안 됐거나 다시 불러오지 않았다", "rea doctor --client pi 확인 후 /reload"],
               ["Ghidra를 못 찾는다", "Ghidra 위치 설정이 등록 정보에 없다", "등록 환경에 GHIDRA_INSTALL_DIR을 넣고 재시작"],
               ["큰 파일이 시간 초과된다", "Ghidra 시작이 330초를 넘었거나, 에이전트의 대기 시간이 먼저 끝났다", "어느 쪽인지 구분한 뒤 해당 제한 시간을 늘린다"],
               ["결과에 의사 코드가 없다", "기본 응답이 요약이다", "필요한 부분만 펼쳐 달라고 한다 (inspect_analysis_view)"],
               ["닫았더니 근거가 사라졌다", "현재 배포본은 세션을 닫을 때 기록을 지운다", "닫기 전에 export_evidence_bundle로 저장"]],
         notes="처음 쓸 때 자주 만나는 문제 다섯 가지입니다. 대부분 연결 설정 문제이고 rea doctor가 원인을 알려 줍니다. "
               "REA는 업데이트가 잦아서, 이상하면 최신 버전으로 올려 보는 것도 방법입니다."),
    dict(kind="two", part="마무리", title="정리: 세 가지 원칙과 첫 실습",
         msg="도구보다 원칙이 먼저입니다. 실습은 쉬운 대상부터 시작합니다",
         groups=[("꼭 지킬 세 가지", ["**허가 범위 안에서만** 분석한다", "**관찰, 추론, 미확인**을 구분해서 적는다",
                                "분석 결과를 **회사 밖으로 내보내지 않는다**. 사내 LLM에 연결된 에이전트만 쓴다"]),
                 ("처음 해 볼 실습", ["사내 웹 앱 하나를 고르고 소유 부서의 동의를 받는다",
                                "개발자 도구(F12)의 Sources 탭에서 소스 맵이 있는지 본다",
                                "Network 탭에서 버튼을 눌렀을 때 나가는 요청을 본다. 테스트 화면에서, 데이터를 바꾸지 않는 버튼으로 한다",
                                "익숙해지면 REA로 설치형 프로그램 하나에 질문 하나를 해 본다"])],
         banner="허가받은 대상에, 좁은 질문을 던지고, 근거의 등급을 따진다",
         notes="오늘 내용에서 세 가지만 기억하신다면 왼쪽입니다. 실습은 웹 앱부터 권합니다. 브라우저만 있으면 되고, 진입점 찾기, 코드 따라가기, "
               "실행으로 확인하기라는 기본 흐름을 한 번 경험할 수 있습니다."),
    dict(kind="table", part="마무리", title="더 읽을 자료",
         msg="문서는 저장소의 reverse_engineering 폴더에 있습니다. 질문 받겠습니다",
         head=["문서", "언제 읽나"], widths=[4.6, 7.7],
         rows=[["re-fundamentals.md", "처음 시작할 때. 오늘 1부의 자세한 설명"],
               ["re-process.md", "새 분석을 시작할 때마다. 7단계 절차"],
               ["rea-windows-pi-setup.md", "회사 PC에 REA를 처음 연결할 때"],
               ["rea-usage-guide.md", "에이전트에 일을 시킬 때. 프롬프트 형식과 대상별 경로"],
               ["installed-program-analysis.md", "설치형 Windows 프로그램을 분석할 때"],
               ["templates/investigation-log.md", "분석 하나를 시작할 때 복사해서 쓰는 기록 양식"]],
         foot="REA: github.com/morluto/rea    REA 가이드: rea.tools/guides    Ghidra: github.com/NationalSecurityAgency/ghidra",
         notes="오늘 다룬 내용은 모두 이 문서들에 더 자세히 있습니다. 처음이라면 위에서부터 순서대로 읽으시면 됩니다. 질문 받겠습니다."),
]


# ---------- pptx: 기본 도형 ----------

def rgb(c):
    return RGBColor.from_string(c)


def fill_tf(tf, s, size, bold=False, color=BLACK, align=PP_ALIGN.LEFT, mono=False, bullets=False):
    """Fill a text frame. '\\n' starts a paragraph, **x** is bold."""
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if bullets:
            p.space_after = Pt(9)
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", "228600")
            pPr.set("indent", "-228600")
            etree.SubElement(pPr, qn("a:buChar"), char="•")
        for j, part in enumerate(re.split(r"\*\*(.+?)\*\*", line) if line else [" "]):
            if not part:
                continue
            r = p.add_run()
            r.text = part
            f = r.font
            f.size, f.bold, f.name = Pt(size), bold or j % 2 == 1, MONO if mono else KR
            f.color.rgb = rgb(color)
            etree.SubElement(r._r.get_or_add_rPr(), qn("a:ea"), typeface=KR)  # Korean glyphs


def box(sl, x, y, w, h, s="", size=14, fill=None, anchor=MSO_ANCHOR.TOP, pad=0.12, shape=MSO_SHAPE.RECTANGLE,
        line=None, **kw):
    shp = sl.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill:
        shp.fill.solid()
        shp.fill.fore_color.rgb = rgb(fill)
    else:
        shp.fill.background()
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(1.25)
    else:
        shp.line.fill.background()
    shp._element.remove(shp._element.find(qn("p:style")))  # theme style adds shadows
    tf = shp.text_frame
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    if s:
        fill_tf(tf, s, size, **kw)
    return shp


def arrow(sl, x1, y1, x2, y2, color=ARROW, dash=False):
    c = sl.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c._element.remove(c._element.find(qn("p:style")))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(2)
    if dash:
        c.line.dash_style = MSO_LINE.DASH
    etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd"), type="triangle")


def card(sl, x, y, w, h, head, body="", sample=None, size=15, hf=GRAY, hc=BLACK, hh=0.55, sample_size=13, **kw):
    """Head strip + light body. Optional white monospace sample on top of the body."""
    box(sl, x, y, w, hh, head, min(size + 1, 16), hf, MSO_ANCHOR.MIDDLE, bold=True, color=hc)
    box(sl, x, y + hh, w, h - hh, fill=LGRAY)
    top = y + hh + 0.08
    if sample:
        ch = 0.25 * (sample.count("\n") + 1) + 0.22
        box(sl, x + 0.12, top + 0.04, w - 0.24, ch, sample, sample_size, WHITE, MSO_ANCHOR.MIDDLE, mono=True)
        top += ch + 0.1
    if body:
        box(sl, x, top, w, y + h - top, body, size, pad=0.16, **kw)


def bar(sl, y, grad_w):
    box(sl, X, y, W, 0.06, fill=GRAY)
    g = box(sl, X, y, grad_w, 0.06)
    g.fill.gradient()
    g.fill.gradient_angle = 0
    g.fill.gradient_stops[0].color.rgb = rgb(RED)
    g.fill.gradient_stops[1].color.rgb = rgb(ORANGE)


def footer(sl, n):
    box(sl, X, 6.93, W, 0.012, fill=GRAY)
    box(sl, X, 6.98, 4, 0.34, LOGO, 10, color=DGRAY, pad=0, anchor=MSO_ANCHOR.MIDDLE)
    box(sl, X + 4, 6.98, W - 8, 0.34, SLOGAN, 10, color=DGRAY, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    box(sl, X + W - 4, 6.98, 4, 0.34, f"{DEPT}   |   {n}", 10, color=DGRAY, pad=0, align=PP_ALIGN.RIGHT,
        anchor=MSO_ANCHOR.MIDDLE)


# ---------- pptx: 슬라이드 종류 (kind 이름 = 함수 이름) ----------

def cards(sl, y, h, s):
    n, gap = len(s["items"]), 0.25
    w = (W - gap * (n - 1)) / n
    for i, (head, body) in enumerate(s["items"]):
        card(sl, X + i * (w + gap), y, w, h, head, body, size=16 if n < 4 else 15, hh=0.6)


def two(sl, y, h, s, x=X, w=None, gap=0.4):
    w = w or (W - gap) / 2
    for i, ((head, items), hf, hc) in enumerate(zip(s["groups"], (GRAY, BLUE), (BLACK, WHITE))):
        card(sl, x + i * (w + gap), y, w, h, head, "\n".join(items), size=16 if w > 5.5 else 14, hf=hf, hc=hc, hh=0.6,
             bullets=True)


def table(sl, y, h, s):
    rows, tint = [s["head"]] + s["rows"], s.get("tint", {})
    rh = min(h / len(rows), 0.7)
    tbl = sl.shapes.add_table(len(rows), len(s["head"]), Inches(X), Inches(y), Inches(W), Inches(rh * len(rows))).table
    for j, wd in enumerate(s["widths"]):
        tbl.columns[j].width = Inches(W * wd / sum(s["widths"]))
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(rh)
        for j, val in enumerate(row):
            hit = next((c for k, c in tint.items() if i and k in val), None)
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(hit or (GRAY if i == 0 else LGRAY))
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = cell.margin_right = Inches(0.12)
            fill_tf(cell.text_frame, val, 15 if i == 0 else 14, bold=i == 0 or j == 0 or bool(hit),
                    align=PP_ALIGN.CENTER if hit else PP_ALIGN.LEFT)


def code(sl, y, h, s):
    box(sl, X, y, 7.3, h, s["code"], 13, LGRAY, MSO_ANCHOR.MIDDLE, pad=0.25, mono=True)
    box(sl, X + 7.6, y, W - 7.6, h, "\n".join(s["side"]), 14, pad=0.05, bullets=True)


def draw_agenda(sl, y, h, s):
    cards(sl, y, 2.5, s)
    box(sl, X, y + 2.75, W, 0.4, "시간 배분 (모두 60분)", 14, pad=0, anchor=MSO_ANCHOR.MIDDLE, bold=True)
    x = X
    for i, (label, mins) in enumerate(s["time"]):
        w = W * mins / 60
        big = mins >= 16
        box(sl, x, y + 3.2, w - 0.05, 0.85, f"{label}\n{mins}분", 12, BLUE if big else (LBLUE if i % 2 else GRAY),
            MSO_ANCHOR.MIDDLE, pad=0.02, align=PP_ALIGN.CENTER, bold=True, color=WHITE if big else BLACK)
        x += w
    box(sl, X, y + 4.1, W, 0.4, "한 장에 2분 안쪽으로 진행한다. 용어 표와 명령어 장은 훑고 지나간다.", 12, pad=0, color=DGRAY)


def draw_reverse(sl, y, h, s):
    box(sl, X, y, W, 0.75, "보통의 개발 (Forward): 설계에서 제품으로", 16, GRAY, MSO_ANCHOR.MIDDLE,
        shape=MSO_SHAPE.RIGHT_ARROW, align=PP_ALIGN.CENTER, bold=True)
    w, gap = 3.3, (W - 3 * 3.3) / 2
    for i, (head, body) in enumerate(s["nodes"]):
        card(sl, X + i * (w + gap), y + 0.95, w, h - 1.9, head, body, size=14)
        if i < 2:
            box(sl, X + i * (w + gap) + w + 0.15, y + h / 2 - 0.2, gap - 0.3, 0.4, "↔", 20, anchor=MSO_ANCHOR.MIDDLE,
                align=PP_ALIGN.CENTER, color=DGRAY, pad=0)
    box(sl, X, y + h - 0.75, W, 0.75, "리버스 엔지니어링 (Reverse): 제품에서 설계 쪽으로", 16, BLUE, MSO_ANCHOR.MIDDLE,
        shape=MSO_SHAPE.LEFT_ARROW, align=PP_ALIGN.CENTER, bold=True, color=WHITE)


def draw_chain(sl, y, h, s):
    """Cards in a row joined by arrows. Optional arrow labels and a wide arrow underneath."""
    nodes, gap, back = s["nodes"], s.get("gap", 0.9), s.get("back")
    ch = s.get("ch", h - 0.7 if back else h)
    w = (W - gap * (len(nodes) - 1)) / len(nodes)
    for i, (head, body, sample) in enumerate(nodes):
        x = X + i * (w + gap)
        card(sl, x, y, w, ch, head, body, sample, size=13, sample_size=s.get("sample_size", 13))
        if i < len(nodes) - 1:
            box(sl, x + w + 0.08, y + ch / 2 - 0.2, gap - 0.16, 0.4, fill=ARROW, shape=MSO_SHAPE.RIGHT_ARROW)
            if s.get("labels"):
                box(sl, x + w - 0.2, y + ch / 2 - 0.65, gap + 0.4, 0.4, s["labels"][i], 12, align=PP_ALIGN.CENTER,
                    color=BLUE, bold=True, pad=0)
    if back:
        fill, shape = {"both": (GRAY, MSO_SHAPE.LEFT_RIGHT_ARROW), "box": (LBLUE, MSO_SHAPE.RECTANGLE)}.get(
            s.get("back_shape"), (BLUE, MSO_SHAPE.LEFT_ARROW))
        box(sl, X, y + h - 0.55, W, 0.55, back, 14, fill, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, shape=shape, bold=True,
            color=WHITE if fill == BLUE else BLUE if fill == LBLUE else BLACK)


def draw_loop(sl, y, h, s):
    w, mid = 4.9, W - 2 * 4.9
    two(sl, y, h, s, w=w, gap=mid)
    cx = X + w
    box(sl, cx, y + 0.55, mid, 0.4, "가설을 세운다", 14, align=PP_ALIGN.CENTER, bold=True, color=BLUE, pad=0)
    box(sl, cx + 0.2, y + 1.0, mid - 0.4, 0.55, fill=BLUE, shape=MSO_SHAPE.RIGHT_ARROW)
    box(sl, cx + 0.2, y + 1.9, mid - 0.4, 0.55, fill=ARROW, shape=MSO_SHAPE.LEFT_ARROW)
    box(sl, cx, y + 2.5, mid, 0.7, "결과가 다르면\n다시 읽는다", 14, align=PP_ALIGN.CENTER, bold=True, color=DGRAY, pad=0)


def draw_evidence(sl, y, h, s):
    n, gap, ch = 4, 0.55, h - 1.05
    w = (W - gap * (n - 1)) / n
    for i, ((head, body), hf) in enumerate(zip(s["nodes"], (LBLUE, YELLOW, GRAY, LBLUE))):
        x = X + i * (w + gap)
        card(sl, x, y, w, ch, head, body, size=13, hf=hf)
        if i < n - 1:
            box(sl, x + w + 0.08, y + ch / 2 - 0.2, gap - 0.16, 0.4, fill=ARROW, shape=MSO_SHAPE.RIGHT_ARROW)
    box(sl, X, y + h - 0.85, W, 0.85, s["unknown"], 14, WHITE, MSO_ANCHOR.MIDDLE, line=ARROW, pad=0.2).line.dash_style = MSO_LINE.DASH


def draw_zones(sl, y, h, s):
    gap = 0.2
    w = (W - gap * 2) / 3
    for i, ((head, items), hf) in enumerate(zip(s["groups"], (LBLUE, YELLOW, PINK))):
        x = X + i * (w + gap)
        box(sl, x, y, w, 0.55, head, 16, hf, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True,
            shape=MSO_SHAPE.PENTAGON if i < 2 else MSO_SHAPE.RECTANGLE)
        ih = (h - 0.55 - 0.12 * 3) / 3
        for k, it in enumerate(items):
            box(sl, x, y + 0.67 + k * (ih + 0.12), w, ih, it.replace("**: ", "**\n"), 13, LGRAY, MSO_ANCHOR.MIDDLE, pad=0.15)


def draw_steps(sl, y, h, s):
    w = W / 7
    for (label, a, b) in (("준비: 코드를 읽기 전에", 0, 3), ("분석", 3, 6), ("정리", 6, 7)):
        box(sl, X + a * w, y, (b - a) * w - 0.12, 0.38, label, 13, anchor=MSO_ANCHOR.MIDDLE, color=DGRAY, bold=True, pad=0.05)
        box(sl, X + a * w, y + 0.38, (b - a) * w - 0.12, 0.03, fill=ARROW)
    for i, (head, body) in enumerate(s["nodes"]):
        fill, color = (GRAY, BLACK) if i < 3 else (LBLUE, BLACK) if i < 6 else (BLUE, WHITE)
        box(sl, X + i * w, y + 0.55, w - 0.02, 0.85, f"{i + 1}\n{head}", 13, fill, MSO_ANCHOR.MIDDLE, pad=0.02,
            shape=MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON, align=PP_ALIGN.CENTER, bold=True, color=color)
        box(sl, X + i * w, y + 1.55, w - 0.12, 1.15, body, 13, LGRAY, pad=0.1)
    box(sl, X + 4.5 * w, y + 2.85, w, 0.4, fill=BLUE, shape=MSO_SHAPE.LEFT_ARROW)
    box(sl, X + 5.6 * w, y + 2.8, 1.4 * w, 0.5, "가설과 다르면 5단계로 돌아간다", 12, anchor=MSO_ANCHOR.MIDDLE, color=BLUE, bold=True, pad=0)


def draw_arch(sl, y, h, s):
    my = y + 1.75  # 가운데 줄의 중심선
    box(sl, X + 6.2, y, W - 6.2, h, line=ARROW).line.dash_style = MSO_LINE.DASH
    box(sl, X + 6.3, y + 0.02, 4, 0.35, "내 PC에서 실행된다", 12, color=DGRAY, bold=True, pad=0.05)
    box(sl, X, my - 0.6, 1.7, 1.2, "**사람**\n질문을 고르고\n결과를 따진다", 13, LGRAY, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    arrow(sl, X + 1.75, my, X + 2.55, my)
    box(sl, X + 1.7, my - 0.45, 0.9, 0.35, "질문", 12, align=PP_ALIGN.CENTER, color=DGRAY, pad=0)
    box(sl, X + 2.6, my - 0.6, 2.3, 1.2, "**AI 에이전트 (pi)**\n어떤 도구를 쓸지\n정한다", 13, GRAY, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
    box(sl, X + 2.6, y, 2.3, 0.7, "**사내 LLM** (AI 모델)", 13, WHITE, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, line=ORANGE)
    arrow(sl, X + 3.75, my - 0.62, X + 3.75, y + 0.74, ORANGE)
    box(sl, X, y + 0.72, 3.65, 0.42, "도구 결과가 모델로 전달된다", 12, align=PP_ALIGN.RIGHT, color=ORANGE, bold=True, pad=0.05)
    arrow(sl, X + 4.95, my - 0.25, X + 6.45, my - 0.25, BLUE)
    arrow(sl, X + 6.45, my + 0.25, X + 4.95, my + 0.25)
    box(sl, X + 4.9, my - 1.0, 1.3, 0.65, "MCP로\n도구 호출", 12, anchor=MSO_ANCHOR.BOTTOM, align=PP_ALIGN.CENTER, color=BLUE, bold=True, pad=0)
    box(sl, X + 4.9, my + 0.35, 1.3, 0.65, "근거와 한계가\n붙은 결과", 12, align=PP_ALIGN.CENTER, color=DGRAY, bold=True, pad=0)
    box(sl, X + 6.5, my - 0.95, 2.3, 1.9, "**REA**\n\n스킬: 작업 지침서\nMCP 서버:\n분석 도구 묶음", 13, BLUE, MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER, color=WHITE)
    eh = (h - 0.5 - 3 * 0.12) / 4
    for i, t in enumerate(("**Ghidra**\nC, C++ 프로그램", "**JavaScript 분석기**\n웹 앱, Electron 앱", "**.NET, Python 분석기**",
                           "**Chrome**\n웹 페이지 관찰")):
        ey = y + 0.42 + i * (eh + 0.12)
        box(sl, X + 9.8, ey, W - 9.8 - 0.12, eh, t, 12, LBLUE if i == 0 else LGRAY, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER)
        arrow(sl, X + 8.85, my, X + 9.75, ey + eh / 2)


def draw_why(sl, y, h, s):
    cards(sl, y, 1.95, s)
    w = (W - 0.75) / 4
    for i in range(4):
        arrow(sl, X + i * (w + 0.25) + w / 2, y + 2.0, X + W / 2 + (i - 1.5) * 1.3, y + 2.5)
    box(sl, X + 2.2, y + 2.55, W - 4.4, 0.75, s["question"], 22, BLUE, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True, color=WHITE)
    box(sl, X, y + 3.35, W, 0.4, "리버스 엔지니어링은 이 질문에 근거를 붙여 답하는 방법이다", 14, align=PP_ALIGN.CENTER, color=DGRAY, pad=0)


def draw_prompt(sl, y, h, s):
    yy = y
    for i, (head, body, sample) in enumerate(s["nodes"]):
        sh = 0.3 * (sample.count("\n") + 1) + 0.3
        box(sl, X, yy, 7.5, sh, sample, 13, LGRAY, MSO_ANCHOR.MIDDLE, pad=0.2, mono=True)
        arrow(sl, X + 8.05, yy + sh / 2, X + 7.55, yy + sh / 2, BLUE)
        box(sl, X + 8.1, yy + sh / 2 - 0.25, 0.5, 0.5, str(i + 1), 16, BLUE, MSO_ANCHOR.MIDDLE, pad=0, shape=MSO_SHAPE.OVAL,
            align=PP_ALIGN.CENTER, bold=True, color=WHITE)
        box(sl, X + 8.7, yy, W - 8.7, sh, f"**{head}**\n{body}", 13, anchor=MSO_ANCHOR.MIDDLE, pad=0.05)
        yy += sh + 0.14


def draw_tree(sl, y, h, s):
    cx = X + W / 2
    box(sl, cx - 3.2, y, 6.4, 0.55, s["root"], 15, GRAY, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True)
    arrow(sl, cx, y + 0.57, cx, y + 0.73)
    box(sl, cx - 3.6, y + 0.75, 7.2, 0.5, s["check"], 14, LBLUE, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True)
    n, gap, ty = len(s["nodes"]), 0.3, y + 1.6
    w = (W - gap * (n - 1)) / n
    for i, (head, body, sample) in enumerate(s["nodes"]):
        x = X + i * (w + gap)
        arrow(sl, cx, y + 1.27, x + w / 2, ty - 0.03)
        card(sl, x, ty, w, h - 1.6, head, body, sample, size=13, hf=BLUE if i == 0 else GRAY, hc=WHITE if i == 0 else BLACK,
             hh=0.7, sample_size=11, align=PP_ALIGN.CENTER)


def draw_native(sl, y, h, s):
    lw, lh, gap = 5.3, (h - 2 * 0.3) / 3, 0.3
    levels = (("프로그램 안의 전체 함수", lw, GRAY, BLACK), (f"**1  {s['nodes'][0][0]}**\n{s['nodes'][0][1]}", lw - 0.9, LBLUE, BLACK),
              (f"**2  {s['nodes'][1][0]}**\n{s['nodes'][1][1]}", lw - 1.8, BLUE, WHITE))
    for i, (t, w, fill, color) in enumerate(levels):
        yy = y + i * (lh + gap)
        box(sl, X + (lw - w) / 2, yy, w, lh, t, 13, fill, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, color=color)
        if i < 2:
            box(sl, X + lw / 2 - 0.25, yy + lh + 0.03, 0.5, gap - 0.06, fill=ARROW, shape=MSO_SHAPE.DOWN_ARROW)
    box(sl, X + lw + 0.1, y + h / 2 - 0.25, 0.6, 0.5, fill=ARROW, shape=MSO_SHAPE.RIGHT_ARROW)
    x0, cw, chh = X + lw + 0.8, (W - lw - 0.8 - 0.25) / 2, (h - 0.25) / 2
    for k, (head, body) in enumerate(s["nodes"][2:]):
        r, c = divmod(k, 2)
        card(sl, x0 + c * (cw + 0.25), y + r * (chh + 0.25), cw, chh, f"{k + 3}  {head}", body, size=13)


def draw_result(sl, y, h, s):
    lw = 7.3
    box(sl, X, y, lw, 0.5, s.get("head", "REA 결과의 구성 (설명용으로 단순화한 예시)"), 14, GRAY, MSO_ANCHOR.MIDDLE, bold=True)
    rh = (h - 0.5 - 0.08 * 4) / 4
    for i, (tag, text, fill) in enumerate(s["rows"]):
        yy = y + 0.58 + i * (rh + 0.08)
        box(sl, X, yy, 1.2, rh, tag, 14, fill, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True)
        box(sl, X + 1.2, yy, lw - 1.2, rh, text, 13, LGRAY, MSO_ANCHOR.MIDDLE, pad=0.15)
    ch = (h - 0.15 * 2) / 3
    for i, (head, body) in enumerate(s["items"]):
        card(sl, X + lw + 0.4, y + i * (ch + 0.15), W - lw - 0.4, ch, head, body, size=13, hh=0.45)


def draw_uses(sl, y, h, s):
    cards(sl, y, 1.75, s)
    by, bh = y + 2.45, h - 2.45
    box(sl, X, y + 1.95, W, 0.4, "계산 로직을 옮길 때는 검증까지 한다", 14, pad=0, anchor=MSO_ANCHOR.MIDDLE, bold=True, color=BLUE)
    box(sl, X, by, 3.4, bh, "**같은 입력**\n정상값\n경계값 (0, 음수, 아주 큰 값)\n빈 입력", 13, LGRAY, MSO_ANCHOR.MIDDLE, pad=0.2)
    for k, t in enumerate(("원본 도구", "새로 만든 코드")):
        yy = by + k * (bh / 2 + 0.075)
        box(sl, X + 4.4, yy, 3.2, bh / 2 - 0.075, t, 14, GRAY, MSO_ANCHOR.MIDDLE, align=PP_ALIGN.CENTER, bold=True)
        arrow(sl, X + 3.45, by + bh / 2, X + 4.35, yy + bh / 4)
        arrow(sl, X + 7.65, yy + bh / 4, X + 8.55, by + bh / 2)
    box(sl, X + 8.6, by, W - 8.6, bh, "**출력 비교**\n시험한 입력에서 같으면 통과\n다르면 다시 분석한다", 14, BLUE, MSO_ANCHOR.MIDDLE,
        align=PP_ALIGN.CENTER, color=WHITE)


def image(sl, y, h, s):
    """A screenshot fitted into the body area, centered. Path is relative to this folder."""
    pic = sl.shapes.add_picture(str(HERE / s["image"]), Inches(X), Inches(y), height=Inches(h))
    if pic.width > Inches(W):
        pic.height = int(pic.height * Inches(W) / pic.width)
        pic.width = Inches(W)
    pic.left = Inches(X) + (Inches(W) - pic.width) // 2


def with_extras(path):
    """Base slides with the extra slides from a JSON file placed after the base slide numbers they name."""
    extras = json.loads(path.read_text(encoding="utf-8"))
    for e in extras:
        assert 1 <= e["after"] <= len(SLIDES) and e["kind"] in EXTRA_KINDS, (e.get("title"), e["kind"], e["after"])
        e.setdefault("part", "REA 실행 결과")
        e.setdefault("notes", "")
    return [x for n, s in enumerate(SLIDES, 1) for x in [s] + [e for e in extras if e["after"] == n]]


EXTRA_KINDS = ("cards", "two", "table", "code", "image", "draw_result", "draw_chain")


def build_pptx(path, slides):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for n, s in enumerate(slides, 1):
        sl = prs.slides.add_slide(prs.slide_layouts[6])
        sl.notes_slide.notes_text_frame.text = s["notes"]
        if s["kind"] == "cover":
            box(sl, X, 2.3, W, 1.1, s["title"], 44, anchor=MSO_ANCHOR.BOTTOM, pad=0, bold=True)
            bar(sl, 3.55, 4.0)
            box(sl, X, 3.8, W, 0.6, s["msg"], 20, pad=0, color=BLUE, bold=True)
            box(sl, X, 4.5, W, 0.5, s["sub"], 16, pad=0, color=DGRAY)
            footer(sl, n)
            continue
        box(sl, X, 0.3, 9.9, 0.7, s["title"], 28, anchor=MSO_ANCHOR.MIDDLE, pad=0, bold=True)
        box(sl, X + 9.9, 0.3, W - 9.9, 0.7, s["part"], 12, anchor=MSO_ANCHOR.BOTTOM, pad=0, color=DGRAY, align=PP_ALIGN.RIGHT)
        bar(sl, 1.04, 2.4)
        box(sl, X, 1.27, W, 0.68, s["msg"], 18, LBLUE, MSO_ANCHOR.MIDDLE, bold=True, color=BLUE, align=PP_ALIGN.CENTER)
        h = BODY_H
        if s.get("foot"):
            h -= FOOT_H + 0.2
            box(sl, X, BODY_Y + h + 0.2, W, FOOT_H, s["foot"], 12, anchor=MSO_ANCHOR.MIDDLE, color=DGRAY, pad=0.05)
        elif s.get("banner"):
            h -= 0.95
            box(sl, X, BODY_Y + h + 0.25, W, 0.7, s["banner"], 16, WHITE, MSO_ANCHOR.MIDDLE, line=BLUE, bold=True, color=BLUE,
                align=PP_ALIGN.CENTER)
        globals()[s["kind"]](sl, BODY_Y, h, s)
        footer(sl, n)
    prs.save(path)


# ---------- md ----------

MD_HEAD = f"""---
tags: [reverse-engineering, rea, ghidra, slides, beginner]
level: beginner
last_updated: 2026-11-14
document_type: learning
category_major: "리버스 엔지니어링"
category_middle: "발표 자료"
category_minor: "팀 입문 deck"
note_kind: "발표 원고"
---

# 리버스 엔지니어링 입문과 REA 활용법: 팀 공유 발표 원고

> 전공이 다양한 팀원에게 리버스 엔지니어링(RE)과 REA 활용법을 소개하는 {len(SLIDES)}장짜리 발표 자료. 새 용어는 나올 때마다 풀어서 설명한다.

## 왜 필요한가? (Why)

팀원 대부분은 전기 공학이나 컴퓨터 공학 전공이 아니다. 기존 문서는 용어를 안다고 가정하고 쓰였으므로, 발표에서는 용어 설명과 비유를 먼저 두고 회사 환경(Windows PC, Ghidra 설치 가능)에서 REA를 쓰는 방법으로 이어 간다.

## 핵심 개념 (What)

| 파일 | 역할 |
|---|---|
| `{NAME}.pptx` | 발표용 슬라이드. 슬라이드마다 발표자 노트가 있다 |
| `{NAME}.bento.html` | 같은 내용의 HTML deck(Bento 형식). 브라우저로 열어 Slideshow로 발표한다 |
| `{NAME}.md` | 이 문서. 슬라이드 내용과 발표자 노트를 글로 옮긴 원고 |
| `build_team_intro.py` | 세 파일을 만드는 스크립트. 내용은 이 스크립트의 `SLIDES`에 있다 |

구성은 4부이고 발표 시간은 질문을 포함해 60분이다. 1부 리버스 엔지니어링이란(3~12장), 2부 작업 절차(13~15장), 3부 Ghidra와 REA(16~19장), 4부 회사 PC에서 써 보기(20~27장), 마무리(28~29장). 슬라이드의 도식은 이 원고에서 목록과 표로 옮겨 적었다.

## 어떻게 사용하는가? (How)

pptx, HTML deck, 이 문서는 직접 고치지 않는다. `build_team_intro.py`의 `SLIDES`를 고친 뒤 저장소 루트에서 다시 만든다.

```bash
python reverse_engineering/slides/build_team_intro.py
```

footer의 로고, 슬로건, 부서 이름은 자리 표시다. 회사 PC에서 스크립트 위쪽의 `LOGO`, `SLOGAN`, `DEPT` 값을 바꾼 뒤 다시 만든다. 슬라이드의 사실은 이 폴더의 문서에 근거가 있어야 한다. 문서를 고치면 해당 슬라이드도 함께 고친다.

아래부터 슬라이드 원고다.
"""

MD_TAIL = """## 참고 자료 (References)

- REA 저장소: <https://github.com/morluto/rea>
- REA 가이드 사이트: <https://rea.tools/guides/>
- Ghidra: <https://github.com/NationalSecurityAgency/ghidra>

## 관련 문서

- [리버스 엔지니어링 기초 개념](../re-fundamentals.md)
- [리버스 엔지니어링 작업 절차](../re-process.md)
- [회사 Windows + pi 환경 설정](../rea-windows-pi-setup.md)
- [REA + Ghidra로 분석하는 방법](../rea-usage-guide.md)
- [설치형 Windows 프로그램을 Ghidra로 분석하는 방법](../installed-program-analysis.md)
- [분석 기록 양식](../templates/investigation-log.md)
"""


def one_line(s, sep=". "):
    """Join slide lines into one markdown line; lines without end punctuation get a period."""
    parts = [p.strip() for p in re.split(r"\n+| {3,}", s) if p.strip()]
    return "".join(p + (" " if p[-1] in ".,:?" or sep == " " else sep) for p in parts).rstrip(". ") + ("." if sep != " " and parts[-1][-1] == "." else "")


def build_md(path, slides):
    out = [MD_HEAD.replace(f"{len(SLIDES)}장짜리", f"{len(slides)}장짜리")]
    for n, s in enumerate(slides, 1):
        out += [f"### {n}. {s['title']}\n", f"> {s['msg']}\n"]
        if "sub" in s:
            out.append(s["sub"] + "\n")
        if "md" in s:
            out.append(s["md"] + "\n")
        if "image" in s:
            out.append(f"![{s['title']}]({s['image']})\n")
        for head, items in s.get("groups", []):
            out += [f"**{head}**\n", *[f"- {it}" for it in items], ""]
        for key, mark in (("items", "-"), ("nodes", "1.")):  # cards are bullets, flows are numbered
            for head, body, *sample in s.get(key, []):
                out.append(f"{mark} **{one_line(head, ' ')}**: {one_line(body)}")
                if sample:
                    out.append("\n\t```text\n" + "\n".join("\t" + ln for ln in sample[0].split("\n")) + "\n\t```\n")
        if "rows" in s and "head" in s:
            out += ["| " + " | ".join(s["head"]) + " |", "|" + "---|" * len(s["head"])]
            out += ["| " + " | ".join(r) + " |" for r in s["rows"]]
        elif "rows" in s:
            out += [f"- **{tag}**: {text}" for tag, text, _ in s["rows"]]
        if "code" in s:
            out += [f"```text\n{s['code']}\n```\n", *[f"- {it}" for it in s["side"]]]
        for key in ("back", "banner"):
            if key in s:
                out.append(f"\n핵심: {one_line(s[key])}")
        out.append("")
        if s.get("foot"):
            out.append(f"참고: {s['foot']}\n")
        out.append(f"**발표자 노트**: {s['notes']}\n")
    out.append(MD_TAIL)
    path.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    assert 20 <= len(SLIDES) <= 30, len(SLIDES)
    for s in SLIDES:  # 저장소 문장 부호 규칙: em dash와 가운데 점을 쓰지 않는다
        assert not re.search("[—·]", str(s)), s["title"]
    build_pptx(HERE / f"{NAME}.pptx", SLIDES)
    build_md(HERE / f"{NAME}.md", SLIDES)
    build_bento(HERE / f"{NAME}.bento.html", SLIDES)
    print(f"{len(SLIDES)} slides -> {NAME}.pptx, {NAME}.md, {NAME}.bento.html")
    extra = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "extra_사내용.json"
    if extra.is_file():  # company-only results: separate output files, kept out of git by .gitignore
        slides = with_extras(extra)
        build_pptx(HERE / f"{NAME}_사내용.pptx", slides)
        build_md(HERE / f"{NAME}_사내용.md", slides)
        build_bento(HERE / f"{NAME}_사내용.bento.html", slides, shell=HERE / f"{NAME}.bento.html")
        print(f"{len(slides)} slides -> {NAME}_사내용.pptx, .md, .bento.html  (git 제외, 올리지 않는다)")
