---
tags: [cd-sem, metrology, tiff, sem, file-format, semi-eda, vendor]
level: intermediate
last_updated: 2026-07-10
---

# CD-SEM / 계측 파일 포맷

> CD-SEM 장비 raw 출력을 읽기 위한 벤더별 포맷·TIFF private tag·기존 오픈소스 reader·SEMI EDA 정리. **손으로 파서 짜기 전에 이 문서를 확인**한다 — 이미지·표준 데이터는 이미 풀려 있는 경우가 대부분이다.

> 표기: 1차/권위 있는 출처로 확인 못 한 항목은 **[미확인]** — 사실이 아니라 "벤더에 확인할 것" 항목으로 취급한다.

## TL;DR 결정 순서

1. **벤더에 먼저 요청** — 포맷 스펙 또는 export(CSV/XML/DB). 법적으로 가장 깨끗하고 대개 가장 빠르다 → [03-legal-and-first-moves](./03-legal-and-first-moves.md).
2. **이미지**: 거의 TIFF. Python `tifffile`이 FEI/Thermo·Zeiss 메타데이터를 이미 파싱한다(§2·§3). 손으로 짜지 말 것.
3. **CD/측정결과**: 대개 벤더 독점 binary/DB. *표준* 경로는 **SEMI EDA/Interface A**(구조화 XML/SOAP) 또는 legacy **SECS/GEM**(§4).
4. **테스트/파라메트릭**: 오픈 표준 **STDF** + 성숙한 오픈 파서 — "표준 먼저 확인"의 모범 사례(§4).

---

## 1. 벤더별 파일 출력

현실 점검: **이미지 포맷은 잘 문서화됐고 대개 TIFF; 측정결과·recipe는 독점이고 공개 스펙이 거의 없다.** 벤더는 결과를 DB/host(SECS/GEM, EDA)·CSV/XML export로 노출하지 문서화된 파일 스펙으로 주지 않는다.

| 벤더 / 툴 | (a) SEM 이미지 | (b) 측정/CD 결과 | (c) Recipe |
|---|---|---|---|
| **Hitachi High-Tech** (CG/CV/GS CD-SEM; S-4800, SU8000) | `.tif`(baseline) + **append/side 텍스트 메타 블록** 또는 **`.txt` sidecar**; 랩툴은 `.bmp`/`.jpg`. Bio-Formats에 Hitachi reader(`.txt` 기반) | 독점; CSV/텍스트 export 또는 SECS/GEM host **[미확인]** | 독점 binary **[미확인]** |
| **Applied Materials** (PROVision CD-SEM, SEMVision) | TIFF(독점 tag **[미확인]**) | 독점; host/EDA·DB export **[미확인]** | 독점 **[미확인]** |
| **KLA** (eSL/eDR review SEM; Archer overlay; SpectraShape OCD) | TIFF | 독점 DB; host + KLA SW **[미확인, 공개 스펙 없음]** | 독점 **[미확인]** |
| **Thermo Fisher / FEI** (Helios, Verios, Apreo, Scios; TIA) | `.tif` + **이미지 뒤에 INI 스타일 텍스트 메타 블록**(`[User]`,`[System]`,`[Beam]`,`[Scan]`…). TEM/STEM: `.ser`+`.emi`, `.emd` | 독점 **[미확인]** | 독점 **[미확인]** |
| **JEOL** (JSM SEM; F200) | `.tif` — 신형은 **TIFF tag에 XML** 임베드; 다수는 **`.txt`/`.par` sidecar**(pixel size). Bio-Formats JEOL reader는 `.dat/.img/.par` | 독점; sidecar 텍스트로 스케일 **[CD결과 미확인]** | 독점 **[미확인]** |
| **Zeiss** (SmartSEM: Merlin, Gemini, Crossbeam, EVO) | `.tif` + **private tag 34118→IFD, 34119=CZ_SEM**의 key/value(`AP_*`,`SV_*`,`DP_*`) | 독점 **[미확인]** | 독점 recipe **[미확인]** |
| **TESCAN** (MIRA, CLARA, VEGA) | `.tif` + **`.hdr` sidecar**(INI) 또는 TIFF tag; 커뮤니티 reader 존재 **[일부 미확인]** | 독점 **[미확인]** | 독점 **[미확인]** |

요약: **TIFF 컨테이너는 표준, 메타데이터 payload는 벤더 독점이나 FEI/Zeiss/JEOL/TESCAN/Hitachi는 커뮤니티가 문서화. 결과·recipe 파일은 전 벤더 독점.**

---

## 2. TIFF 기반 SEM 이미지와 private tag

- baseline TIFF는 tagged field(IFD)에 메타데이터 저장. **tag ID ≥ 32768(0x8000)은 private 범위**로 벤더가 자체 메타에 사용. SEM 벤더는 (a) private tag에 텍스트/binary blob을 넣거나, (b) 이미지 strip 뒤에 formal tag가 아닌 평문 블록을 붙인다(FEI, 일부 Hitachi).
- 항상 통하는 두 폴백: **모든 IFD tag 열거**(tag 기반 벤더), **파일 꼬리에서 append된 ASCII 블록 grep**(FEI/Hitachi).

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
검증됨: `tifffile`은 `fei_metadata`·`sem_metadata`·`tvips_metadata`와 `is_fei`/`is_sem`/`is_tvips` 플래그를 노출.

**exiftool** (`brew install exiftool`):
```bash
exiftool -a -u -g1 image.tif        # -u: Unknown/private tag 표시(핵심), -a: 전부, -g1: 그룹
exiftool -htmlDump image.tif > d.html   # 바이트 단위 구조 덤프 — RE에 최적
```

**ImageJ/Fiji**: `Image ▸ Show Info…`(ImageDescription·known tag). **Bio-Formats** importer(`Plugins ▸ Bio-Formats ▸ Importer`, "Display metadata")가 훨씬 많이 보여준다.

### 벤더별 추출

**Zeiss SmartSEM** — private IFD tag **34118(0x8546)**→sub-IFD, 파싱 결과가 `tifffile.sem_metadata`(내부 CZ_SEM, tag 34119). 키 접두: `AP_`(analog, `AP_MAG`·`AP_BRIGHTNESS`), `SV_`(string, `SV_USER_NAME`·`SV_SERIAL_NUMBER`), `DP_`(digital), `AP_DATE`/`AP_TIME`. 값 튜플에 단위 포함.
```python
with tifffile.TiffFile("zeiss.tif") as tif:
    sem = tif.sem_metadata
    print(sem.get("ap_mag"), sem.get("ap_image_pixel_size"))
```
`sem_metadata`가 없으면 tag 34118/34119 값을 직접 읽어 개행/`=`로 split. 참고: `ks00x/zeiss_tiff_meta`.

**FEI / Thermo** — **이미지 데이터 뒤에 INI/configparser 스타일 텍스트 블록**(`[User]`,`[System]`,`[Beam]`,`[EBeam]`,`[Scan]`,`[Stage]`,`[Image]`…).
```python
with tifffile.TiffFile("helios.tif") as tif:
    if tif.is_fei:
        m = tif.fei_metadata
        pixel_w = float(m["Scan"]["PixelWidth"])   # meters/pixel
        hv = float(m["Beam"]["HV"])
```
`tifffile`이 못 알아보는 변종 폴백 — 꼬리에서 INI 블록 grep:
```python
import configparser
raw = open("helios.tif","rb").read()
i = raw.find(b"[User]")
cp = configparser.ConfigParser(strict=False)
cp.read_string(raw[i:].decode("latin-1","ignore"))
print(cp["Scan"]["PixelWidth"])
```

**Hitachi** — 두 패턴: (1) 이미지 근처/뒤에 **평문 메타 블록**(`PixelSize`,`Magnification`,`AcceleratingVoltage`), (2) 동명 **`.txt` sidecar**. Bio-Formats Hitachi reader는 그 `.txt` 기반(S-4800 테스트).
```python
raw = open("hitachi.tif","rb").read()
tail = raw[-8000:].decode("latin-1","ignore")   # 꼬리에서 key=value 라인 스캔
```
정확한 키명/헤더 유무는 **[미확인]** — 실제 파일 꼬리를 `exiftool -htmlDump`로 확인.

**JEOL** — 신형은 **TIFF tag 안 XML**(tag 값 얻어 XML 파싱), 구형은 **`.txt`/`.par` sidecar**. 참고: `rfwebster/jeoltiff`(`tifffile`+`untangle`). Bio-Formats JEOL reader는 `.dat/.img/.par`.

---

## 3. 이미 SEM/현미경 포맷을 파싱하는 오픈소스 (파서 짜기 전 확인)

| 라이브러리 | 설치 | 처리 |
|---|---|---|
| **tifffile** | `pip install tifffile` | 주력. **FEI/Thermo**(`fei_metadata`)·**Zeiss CZ_SEM**(`sem_metadata`)·TVIPS·ImageJ·OME-TIFF·LSM·STK·ScanImage·NDPI. private tag 열거 |
| **RosettaSciIO** (HyperSpy 백엔드) | `pip install rosettasciio` | TIFF 스케일: FEI·Zeiss·Olympus SIS·JEOL SightX·Hamamatsu. EM: FEI **TIA**(`.ser`/`.emi`)·Gatan **DM3/DM4**·**EMD**·Bruker·JEOL EDS·MRC (Zeiss/FEI TIFF 스케일은 "제한적") |
| **HyperSpy** | `pip install hyperspy` | 위의 고수준 API. `hs.load("x.tif")` → `.metadata`/`.original_metadata` |
| **Bio-Formats / python-bioformats** | `pip install python-bioformats` (JVM 필요) | **Hitachi**·**JEOL**·FEI·Zeiss 등 ~150 포맷 전용 reader. Java 기반, 무겁다 |
| **ncempy / openNCEM** (LBNL) | `pip install ncempy` | **SER(+EMI 메타)·DM3/DM4·EMD·MRC**. read-only, 잘 유지됨. TEM/STEM 중심 |
| **imageio** | `pip install imageio` | tifffile/Pillow 래퍼. 픽셀엔 좋고 벤더 메타엔 약함 |
| **Fiji 플러그인** | Fiji | "SEM FEI metadata scale", EM-tool/IMBalENce, zeiss_tiff_meta, jeoltiff |
| **pyUSID / sidpy** | `pip install pyUSID sidpy` | Universal Spectroscopy/Imaging + HDF5 translator. HDF5 표준화 아니면 과함 |

커버리지 요약: **FEI/Thermo TIFF → tifffile 또는 RosettaSciIO. Zeiss TIFF → tifffile. Hitachi·JEOL → Bio-Formats(또는 jeoltiff/EM-tool). SER/DM3/EMD → ncempy 또는 RosettaSciIO.**

---

## 4. Fab 표준

**장비 인터페이스 / host 통신 (데이터가 파일이 아니라 스트림으로 나오는 경로):**
- **SECS-I (SEMI E4)** — 시리얼(RS-232). legacy.
- **HSMS (SEMI E37)** — SECS over TCP/IP. 현대 전송.
- **SECS-II (SEMI E5)** — 메시지 내용 semantics.
- **GEM (SEMI E30)** — 표준 동작/상태 모델·이벤트·알람·데이터 수집. **GEM300** = 300mm 세트(E40 process job, E87 carrier, E90 substrate tracking, E94 control job…).

**EDA / Interface A (구조화 고용량 데이터 경로 — 풍부한 계측 데이터에 최적):**
- 세트: **E120**(CEM), **E125**(자기기술 EqSD), **E132**(인증), **E134**(Data Collection Management).
- 와이어: **SOAP/XML over HTTP(S)**. 차세대는 **gRPC + Protobuf**로 이동 중.
- **CD-SEM 데이터가 독점 binary 대신 EDA로 나오는 경우:** EDA 지원 툴(300mm에 흔함)이면 Data Collection Plan을 구독해 **구조화 XML report/trace**를 받는다. 파라미터명은 E125 자기기술 모델에서 온다 → CD·sidewall·roughness·좌표·recipe 컨텍스트를 named·typed 값으로, 온디스크 결과 파일을 건드리지 않고 얻는다. **MI 데이터 통합의 권장 경로.** 실무: fab의 EES/FDC나 미들웨어(Cimetrix/PEER/Agileo)가 이미 EDA를 종단한다 — 거기서 받아라.

**레이아웃 데이터:** GDSII·**OASIS(SEMI P39/P44)**·**OpenAccess**. CD 사이트를 설계 좌표에 묶을 때. 오픈 파서 `gdstk`·`gdspy`·`klayout`.

**테스트/파라메트릭 — "표준 먼저" 모범 사례:**
- **STDF (v4)** — ATE 표준 binary. CD-SEM은 아니지만, 문서화된 binary + 성숙한 오픈 파서의 전형:
  - `pystdf` (`pip install pystdf`) — STDF→CSV/XLSX.
  - `Semi-ATE/STDF` (`pip install Semi-ATE-STDF`) — read/write, numpy/pandas 친화.
- **교훈:** 역공학 전에 PyPI/GitHub에서 포맷명을 검색하라 — 이미 파서가 있을 수 있다(STDF·TIFF-SEM·SER/DM3는 있고, CD-SEM 결과 파일은 아직 없음).

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

**CD-SEM 측정결과·recipe** 파일의 공개 역공학 스펙(Hitachi CG/CV, AMAT PROVision, KLA)은 **찾지 못함** — host/EDA/export로만 노출되며 독점으로 유지되는 것과 일치. "안 찾아봤다"가 아니라 "없음 확인".

## 미확인 사항 (명시)
- Hitachi append 텍스트/`.txt` sidecar의 정확한 키명·헤더 유무 → 실제 파일 확인.
- TESCAN `.hdr` 내부 구조, 메타가 `.hdr` vs TIFF tag 어디인지.
- AMAT PROVision·KLA Archer/eSL 온디스크 결과·recipe 포맷 → 공개 스펙 없음, EDA/host/export로 획득.
- 특정 fab 툴의 TIFF 변종을 `tifffile` `is_fei`/`is_sem`가 자동감지하는지(펌웨어 의존) → 꼬리 grep + tag 덤프 폴백 항상 유지.

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
