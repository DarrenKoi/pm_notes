---
tags: [cd-sem, metrology, tiff, sem, file-format, semi-eda, vendor]
level: intermediate
last_updated: 2026-07-10
reviewed_on: 2026-10-04
review_status: reviewed_with_limits
document_type: learning_note
---

# CD-SEM / 계측 파일 포맷

> CD-SEM 장비 raw 출력을 읽기 위한 벤더별 포맷·TIFF private tag·기존 오픈소스 reader·SEMI EDA 정리. **손으로 파서 짜기 전에 이 문서를 확인**한다 — 지원되는 이미지/표준 형식은 기존 reader로 조사할 수 있지만 실제 CD-SEM 판본의 지원 비율은 미확인이다.

> 표기: 1차/권위 있는 출처로 확인 못 한 항목은 **[미확인]** — 사실이 아니라 "벤더에 확인할 것" 항목으로 취급한다.

## TL;DR 결정 순서

1. **벤더에 먼저 요청** — 포맷 스펙 또는 export(CSV/XML/DB). 권한·계약·관할과 제공 범위를 확인한다. 다른 방법보다 빠르거나 항상 허용된다는 보장은 없다 → [03-legal-and-first-moves](./03-legal-and-first-moves.md).
2. **이미지**: 실제 container를 확인한다. `tifffile`의 FEI/Thermo·Zeiss tag 구현(§2·§3)을 먼저 대조하되 필요한 픽셀/값/단위까지 UI/export와 맞춰 확인한다.
3. **CD/측정결과**: 실제 binary/DB/export는 장비별 미확인이다. **SEMI EDA/Interface A** 또는 **SECS/GEM**(§4)은 인터페이스 후보이며 필요한 필드/권한/설치 판본을 확인해야 한다.
4. **테스트/파라메트릭**: **STDF**와 공개 parser 후보(현 판본/지원 범위 미확인) — "표준 먼저 확인"의 모범 사례(§4).

---

## 1. 벤더별 파일 출력

현실 점검: 아래는 원래 수집한 벤더/모델/출력 후보를 보존한 표다. **확인한 특정 reader 구현을 모든 벤더·CD-SEM·펌웨어의 출력 형식으로 확대하지 않는다.** 실제 DB/host(SECS/GEM,EDA)·CSV/XML export와 공개 스펙의 존재/제공 범위는 벤더/설치 환경별 미확인이다.

| 벤더 / 툴 | (a) SEM 이미지 | (b) 측정/CD 결과 | (c) Recipe |
|---|---|---|---|
| **Hitachi High-Tech** (CG/CV/GS CD-SEM; S-4800, SU8000) | `.tif`(baseline) + **append/side 텍스트 메타 블록** 또는 **`.txt` sidecar**; 랩툴은 `.bmp`/`.jpg`. Bio-Formats에 Hitachi reader(`.txt` 기반) | 독점; CSV/텍스트 export 또는 SECS/GEM host **[미확인]** | 독점 binary **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **Applied Materials** (PROVision CD-SEM, SEMVision) | TIFF(독점 tag **[미확인]**) | 독점; host/EDA·DB export **[미확인]** | 독점 **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **KLA** (eSL/eDR review SEM; Archer overlay; SpectraShape OCD) | TIFF | 독점 DB; host + KLA SW **[미확인, 공개 스펙 없음]** | 독점 **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **Thermo Fisher / FEI** (Helios, Verios, Apreo, Scios; TIA) | `.tif` + **FEI_SFEG34680/FEI_HELIOS34682 tag의 INI 스타일 메타데이터**(`[User]`,`[System]`,`[Beam]`,`[Scan]`…). TEM/STEM: `.ser`+`.emi`, `.emd` | 독점 **[미확인]** | 독점 **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **JEOL** (JSM SEM; F200) | `.tif` — 신형은 **TIFF tag에 XML** 임베드; 다수는 **`.txt`/`.par` sidecar**(pixel size). Bio-Formats JEOL reader는 `.dat/.img/.par` | 독점; sidecar 텍스트로 스케일 **[CD결과 미확인]** | 독점 **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **Zeiss** (SmartSEM: Merlin, Gemini, Crossbeam, EVO) | `.tif` + **private tag34118=CZ_SEM**(확인한 tifffile 구현;34119 연결 설명 정정)의 key/value(`AP_*`,`SV_*`,`DP_*`) | 독점 **[미확인]** | 독점 recipe **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |
| **TESCAN** (MIRA, CLARA, VEGA) | `.tif` + **`.hdr` sidecar**(INI) 또는 TIFF tag; 커뮤니티 reader 존재 **[일부 미확인]** | 독점 **[미확인]** | 독점 **[미확인]**  **[실제 모델/펌웨어·결과/recipe 미확인]** |

요약: TIFF container와 벤더 메타데이터 구현은 구분한다. 확인한 FEI/Zeiss reader 소스는 일부 형식의 근거이며 JEOL/TESCAN/Hitachi 전체 모델이나 모든 결과/recipe의 독점성·공개 문서 부재를 증명하지 않는다.

---

## 2. TIFF 기반 SEM 이미지와 private tag

- baseline TIFF는 tagged field(IFD)에 메타데이터 저장. private tag와 vendor payload가 있을 수 있다. 원래 tag ID≥32768 private 범위 설명은 TIFF 규격 전문을 이번 검토에서 대조하지 못해 **[미확인]**으로 남긴다. 확인한 FEI/Zeiss 구현은 tag 값을 읽으며 일반적인 EOF append 설명과 다르다. 일부 파일의 평문 tail 여부는 실제 bytes로 확인해야 한다.
- 검토할 두 조사 방법(항상 성공하지 않음): **모든 IFD tag 열거**(tag 기반 벤더), **파일 꼬리의 ASCII 후보 확인**(형식/경계/encoding 미확인, tag와 혼동하지 않음).

### tag 열거 3종

**Python `tifffile`** (`pip install tifffile`):
```python
import tifffile
with tifffile.TiffFile("image.tif") as tif:
    print("is_fei:", tif.is_fei, "| is_sem:", tif.is_sem)  # 벤더 자동감지
    if tif.is_fei: print(tif.fei_metadata)   # FEI/Thermo [User]/[Beam]... dict
    if tif.is_sem: print(tif.sem_metadata)   # Zeiss CZ_SEM {key:(idx,value,unit)}
    for pi, page in enumerate(tif.pages):     # 모든 private tag 무차별 덤프
        for tag in page.tags:
            print(pi, tag.code, tag.name, repr(tag.value)[:120])  # code>=32768 주목
```
공식 소스/문서 대조(설치/실제 장비 미실행): `tifffile`은 `fei_metadata`·`sem_metadata`·`tvips_metadata`와 `is_fei`/`is_sem`/`is_tvips` 플래그를 노출.

**exiftool** (`brew install exiftool`):
```bash
exiftool -a -u -g1 image.tif        # -u: Unknown/private tag 표시(핵심), -a: 전부, -g1: 그룹
exiftool -htmlDump image.tif > d.html   # 바이트 단위 구조 덤프 — RE에 최적
```

**ImageJ/Fiji**: `Image ▸ Show Info…`(ImageDescription·known tag). **Bio-Formats** importer(`Plugins ▸ Bio-Formats ▸ Importer`, "Display metadata")가 훨씬 많이 보여준다.

### 벤더별 추출

**Zeiss SmartSEM 후보** — 확인한 tifffile 소스는 private tag **34118(0x8546)=CZ_SEM**의 값을 `sem_metadata`로 읽는다. 원래34118→sub-IFD/34119=CZ_SEM 연결은 이 구현과 달라 정정했다. 실제 SmartSEM 모델/판본 지원은 미확인이다. 키 접두: `AP_`(analog, `AP_MAG`·`AP_BRIGHTNESS`), `SV_`(string, `SV_USER_NAME`·`SV_SERIAL_NUMBER`), `DP_`(digital), `AP_DATE`/`AP_TIME`. 키/튜플 형태와 단위는 실제 metadata/판본에서 확인한다. 접두 의미와 모든 키 존재는 별도 미확인이다.
```python
import tifffile
with tifffile.TiffFile("zeiss.tif") as tif:
    sem = tif.sem_metadata
    if sem is None:
        print("미확인: CZ_SEM metadata 없음")
    else:
        print(sem.get("ap_mag"), sem.get("ap_image_pixel_size"))
```
`sem_metadata`가 없으면 미확인으로 남기고 실제 tag/code/type/경계와 파서 지원을 조사한다. 임의34119 split을 공통 fallback으로 쓰지 않는다. 참고 후보 `ks00x/zeiss_tiff_meta`의 현재 구현은 별도 미확인이다.

**FEI / Thermo 후보** — 확인한 tifffile의 FEI_SFEG34680/FEI_HELIOS34682 tag에서 읽는 INI 스타일 metadata(`[User]`,`[System]`,`[Beam]`,`[EBeam]`,`[Scan]`,`[Stage]`,`[Image]`…).
```python
import tifffile
with tifffile.TiffFile("helios.tif") as tif:
    m = tif.fei_metadata
    if m is None:
        print("미확인: FEI metadata 없음")
    else:
        scan = m.get("Scan", {})
        beam = m.get("Beam", {})  # 일부 파일은 EBeam 등 다른 section일 수 있음
        pixel_w = float(scan["PixelWidth"]) if "PixelWidth" in scan else None
        hv = float(beam["HV"]) if "HV" in beam else None
        print(pixel_w, hv)  # 필요 키/단위/UI 대조 전에는 미확인 값
```
`tifffile`이 못 알아보는 파일의 **가상 ASCII tail 조사 초안**이다. FEI 전체에 적용하는 parser가 아니다. 분석 사본의 tail 안에서 header와 경계를 별도 확인한다. 원래 find==-1/중복 section 무시를 정정했다:
```python
import configparser
from pathlib import Path

def parse_ascii_tail(raw: bytes) -> str | None:
    tail = raw[-8000:]  # 조사 범위 예; 경계/실제 encoding을 먼저 확인
    i = tail.find(b"[User]")
    if i < 0:
        return None
    text = tail[i:].decode("ascii", errors="strict")
    cp = configparser.ConfigParser(interpolation=None, strict=True)
    cp.read_string(text)
    return cp.get("Scan", "PixelWidth", fallback=None)

if __name__ == "__main__":
    print(parse_ascii_tail(Path("helios.tif").read_bytes()))
```

**Hitachi 조사 후보 [실제 형식 미확인]** — 원래 두 패턴: (1) 이미지 근처/뒤에 **평문 메타 블록**(`PixelSize`,`Magnification`,`AcceleratingVoltage`), (2) 동명 **`.txt` sidecar**. 옛 Bio-Formats5.2.0 공개 PDF의 Hitachi S-4800 항목은.txt/.tif/.bmp/.jpg를 열거한다.8.1.0 JavaAPI에는 HitachiReader가 있다. 현재 reader 소스 조회는 실패해 정확한 sidecar/키/판본은 미확인이다. 이를 CG/CV/GS 또는 SU8000 출력으로 확대하지 않는다.
```python
from pathlib import Path
raw = Path("hitachi.tif").read_bytes()
tail = raw[-8000:].decode("latin-1","ignore")   # 꼬리에서 key=value 라인 스캔
```
이 tail decode는 byte 후보를 표시할 뿐 유효 metadata 파싱이 아니다. 정확한 키명/헤더/encoding/경계 유무는 **[미확인]** — 실제 파일 꼬리를 `exiftool -htmlDump`로 확인.

**JEOL 후보 [실제 모델/판본 미확인]** — 원래 조사 항목은 신형 **TIFF tag 안 XML**(tag 값 얻어 XML 파싱), 구형은 **`.txt`/`.par` sidecar**. 참고: `rfwebster/jeoltiff`(`tifffile`+`untangle`). Bio-Formats JEOL reader는 `.dat/.img/.par`.

---

## 3. 이미 SEM/현미경 포맷을 파싱하는 오픈소스 (파서 짜기 전 확인)

| 라이브러리 | 설치 | 처리 |
|---|---|---|
| **tifffile** | `pip install tifffile` | 주력. **FEI/Thermo**(`fei_metadata`)·**Zeiss CZ_SEM**(`sem_metadata`)·TVIPS·ImageJ·OME-TIFF·LSM·STK·ScanImage·NDPI. private tag 열거 |
| **RosettaSciIO** (HyperSpy 백엔드) | `pip install rosettasciio` | TIFF 스케일: FEI·Zeiss·Olympus SIS·JEOL SightX·Hamamatsu. EM: FEI **TIA**(`.ser`/`.emi`)·Gatan **DM3/DM4**·**EMD**·Bruker·JEOL EDS·MRC (Zeiss/FEI TIFF 스케일은 "제한적") |
| **HyperSpy** | `pip install hyperspy` | 위의 고수준 API. `hs.load("x.tif")` → `.metadata`/`.original_metadata` |
| **Bio-Formats / python-bioformats** | `pip install python-bioformats` (JVM 필요) | **Hitachi**·**JEOL**·FEI·Zeiss 등 reader 후보. 현재 개수·Python binding/Java 의존성·지원/성능은 미확인 |
| **ncempy / openNCEM** (LBNL) | `pip install ncempy` | **SER(+EMI 메타)·DM3/DM4·EMD·MRC**. TEM/STEM 참고 후보;현재 API/쓰기 지원/유지 상태 미확인 |
| **imageio** | `pip install imageio` | tifffile/Pillow 래퍼. 픽셀엔 좋고 벤더 메타엔 약함 |
| **Fiji 플러그인** | Fiji | "SEM FEI metadata scale", EM-tool/IMBalENce, zeiss_tiff_meta, jeoltiff |
| **pyUSID / sidpy** | `pip install pyUSID sidpy` | Universal Spectroscopy/Imaging + HDF5 translator. HDF5 표준화 아니면 과함 |

조사 후보 요약(필요 값/판본의 실제 지원과 다름): **FEI/Thermo TIFF → tifffile 또는 RosettaSciIO. Zeiss TIFF → tifffile. Hitachi·JEOL → Bio-Formats(또는 jeoltiff/EM-tool). SER/DM3/EMD → ncempy 또는 RosettaSciIO.**

---

## 4. Fab 표준

**장비 인터페이스 / host 통신 (데이터가 파일이 아니라 스트림으로 나오는 경로):**
- **SECS-I (SEMI E4)** — 시리얼(RS-232). legacy.
- **HSMS (SEMI E37)** — SECS over TCP/IP. 현대 전송.
- **SECS-II (SEMI E5)** — 메시지 내용 semantics.
- **GEM (SEMI E30)** — 표준 동작/상태 모델·이벤트·알람·데이터 수집. **GEM300** = 300mm 세트(E40 process job, E87 carrier, E90 substrate tracking, E94 control job…).

**EDA / Interface A (장비 데이터 수집 인터페이스 후보):**
- 세트: **E120**(CEM), **E125**(자기기술 EqSD), **E132**(client 인증/인가), **E134**(Data Collection Management).
- 와이어: 원래 SOAP/XML over HTTP(S)와 gRPC/Protobuf 전환 설명은 현재 Freeze/장비 판본의 규격 전문을 대조하지 못해 **[미확인]**이다. SEMI 공개 목록의 E128 XML Message Structures는 특정 설치의 transport/필드 지원을 보증하지 않는다.
- **CD-SEM 데이터의 EDA 경로 후보:** 장비가 실제 지원하는 Data Collection Plan/report/trace·E125 모델·판본·권한을 확인한다. CD·sidewall·roughness·좌표·recipe 컨텍스트가 모두 노출된다는 보장은 없다.300mm 보급률과 특정 fab의 EES/FDC·Cimetrix/PEER/Agileo 종단 상태도 **[미확인]**이다. 지원과 필요한 값/단위/주기를 대조한 경우 파일 해석의 대안으로 검토한다.

**레이아웃 데이터:** GDSII·**OASIS(SEMI P39)**와 **mask-tool용 OASIS(SEMI P44)**·**OpenAccess**. CD 사이트를 설계 좌표에 묶을 때. 오픈 파서 `gdstk`·`gdspy`·`klayout`.

**테스트/파라메트릭 — "표준 먼저" 모범 사례:**
- **STDF (v4)** — ATE 표준 binary. CD-SEM은 아니지만, 문서화된 binary + 성숙한 오픈 파서의 전형:
  - `pystdf` (`pip install pystdf`) — STDF parser 후보;현 CSV/XLSX 기능/판본은 미확인.
  - `Semi-ATE/STDF` (`pip install Semi-ATE-STDF`) — read/write·NumPy/pandas 연동은 원래 참고 주장으로 현 API/설치 미확인.
- **교훈:** 역공학 전에 PyPI/GitHub에서 포맷명을 검색하라 — 이미 파서가 있을 수 있다(확인한 reader와 실제 모델 지원을 구분하며 CD-SEM 결과 parser의 공개 부재를 단정하지 않는다).

---

## 5. 커뮤니티 역공학 사례 (출발점 + 선례)

- **`ks00x/zeiss_tiff_meta`** — Zeiss SmartSEM TIFF 메타(tag 34118/34119), ImageJ용 DPI/스케일 보정.
- **`rfwebster/jeoltiff`** — JEOL TIFF tag의 XML 추출(F200), DM/ImageJ 스케일 tag 재작성.
- **`IMBalENce/EM-tool`** + IMBalENce Fiji 플러그인 — 멀티벤더 EM 메타/스케일, JEOL·**TESCAN** 지원.
- **"SEM FEI metadata scale"** Fiji 플러그인 — FEI `[User]`/`[Beam]` INI 블록 파싱.
- **`cgohlke/tifffile`** — FEI/Zeiss 파서 소스 자체가 그 독점 포맷의 문서화된 역공학.
- **`ercius/openNCEM`** — FEI SER/EMI·Gatan DM3 binary 레이아웃 공개 문서.
- **Bio-Formats** `HitachiReader`/`JEOLReader` — 오픈 Java 구현(샘플로부터 역공학, "공식 스펙 필요"라고 명시).
- **image.sc 포럼** — FEI/Zeiss/Hitachi/JEOL/TESCAN SEM TIFF 메타 추출 스레드의 사실상 Q&A 허브.

**CD-SEM 측정결과·recipe** 파일의 공개 스펙/reader(Hitachi CG/CV,AMAT PROVision,KLA)는 현재 검토 범위에서 **미확인**이다. 검색에서 찾지 못한 사실은 공개 부재/독점성/host·EDA·export만 제공한다는 증거가 아니다. 원래 "없음 확인" 단정은 철회한다.

## 미확인 사항 (명시)
- Hitachi append 텍스트/`.txt` sidecar의 정확한 키명·헤더 유무 → 실제 파일 확인.
- TESCAN `.hdr` 내부 구조, 메타가 `.hdr` vs TIFF tag 어디인지.
- AMAT PROVision·KLA Archer/eSL 온디스크 결과·recipe 포맷 → 공개 스펙과 EDA/host/export의 제공 여부·필드·권한 미확인.
- 특정 fab 툴의 TIFF 변종을 `tifffile` `is_fei`/`is_sem`가 자동감지하는지(펌웨어 의존) → tag/tail 조사는 후보 방법이며 항상 성공하지 않는다.

## 참고 자료 (References)

- tifffile: github.com/cgohlke/tifffile
- RosettaSciIO TIFF: hyperspy.org/rosettasciio/supported_formats/tiff.html
- Zeiss: solarchemist.se/2015/03/20/sem-tiffinfo/ · github.com/ks00x/zeiss_tiff_meta
- FEI: imagej.net/plugins/sem-fei-metadata-scale
- JEOL: github.com/rfwebster/jeoltiff · docs.openmicroscopy.org/bio-formats/…/formats/jeol.html
- Hitachi/EM-tool: forum.image.sc/t/…/24240 · github.com/IMBalENce/EM-tool
- ncempy: github.com/ercius/openNCEM · openncem.readthedocs.io
- SEMI EDA: semi.org/en/next-gen-semi-eda-standards · cimetrix.com/interfacea · peergroup.com/eda-semi-standards
- STDF: github.com/cmars/pystdf · github.com/Semi-ATE/STDF


## 확인 근거와 검토 결과

2026-10-04 [tifffile 공식 API](https://www.cgohlke.com/docs/tifffile/)·[소스](https://github.com/cgohlke/tifffile) 표시2026.9.20의 FEI34680/34682·Zeiss34118와 dict/None 계약, [RosettaSciIO TIFF](https://rosettasciio.readthedocs.io/en/stable/supported_formats/tiff.html)의 제한된 tag/scale, [SEMI 공개 목록](https://www.semi.org/en/products-services/standards/information_and_control)의 E4/E5/E37/E30·EDA 규격 명칭을 대조했다. P39는 [SEMI2021 소개](https://www.semi.org/en/standards-watch-2021sept/new-curvilinear-format-tf), P44 mask-tool 명칭은 [2023 SNARF7026](https://downloads.semi.org/web/wstdsbal.nsf/c8f1681362290e72882581800081edd5/b60f506e830c1c6d8825899000641ae8!OpenDocument)로 구분했다. SNARF는 재승인 제안이며 최신 규격 전문/승인 결과가 아니다.

Hitachi 자료는 [Bio-Formats8.1.0 API](https://downloads.openmicroscopy.org/bio-formats/8.1.0/api/loci/formats/in/HitachiReader.html)와 옛5.2.0 문서의 검색 색인 범위다.8.1.0/8.3.0 format URL와 Java 소스는 조회 오류였다. ExifTool 공식 pod도403으로 -a/-u/-g1/htmlDump의 현 옵션/설치·Fiji 메뉴는 미확인이다. 위 설치/커뮤니티/library 표와§5 사례는 탐색 후보이며 현재 API·가격·성능·모든 형식 지원을 개별 검증하지 않았다. References의 …를 포함한 URL은 원래 불완전 인용이며 유효 출처로 계산하지 않는다. 회사 자료/파일/벤더 SW를 외부로 업로드하거나 실행하지 않았다.

원래 모든 절·벤더/모델·reader/표준/커뮤니티 맥락·작성일을 보존하고 tag/append/범용 fallback/없음 확인을 정정했다. Python 예제는 dict/None·필수 키 부재를 구분하고 가상 tail 파서는 부재 None·ASCII/중복 오류를 숨기지 않는다. 실제 픽셀·메타데이터/단위·CD-SEM 판본·SEMI 유료 전문/설치 Freeze는 미확인이다. HERDR_ENV=1/pane_not_found로 Claude 의견이 없어 완전 통합·실제 회사 인터페이스 선택은 보류했다. [정리 기록](../organization-log.md)에 세 단계 검증을 남긴다.
