# Orange Pi Zero 3 / H618 — adaptive dual-HackRF drone DOA

## Назначение

Это боевой CPU/NEON-конвейер для двух синхронизированных HackRF и Orange Pi Zero 3 (H618, 4× Cortex-A53, 1 GB RAM). Система рассчитана на последовательное перестраивание центральной частоты по диапазону примерно 800 MHz–6 GHz; один процесс обрабатывает одну центральную частоту за раз.

Главный принцип: **не жертвовать обнаружением или точностью ради sparse DFT**. Slave работает в `auto`-режиме, который выбирает exact-bin DFT только после измерения фактической занятости спектра. Для широкополосного источника Slave автоматически переводится на full FFT, а phase/DOA интегрируется по всему обнаруженному occupied band.

## Что исправлено после ревью

### 1. CW/OFDM/DSSS/FHSS: sparse DFT больше не является предположением

Master всегда строит полный FFT и обнаруживает кандидатов через локальный CA-CFAR плюс multi-scale band-energy detector.

Для каждого кандидата определяется реальная полоса:

```text
Master FFT
  -> local CFAR peaks
  -> объединение соседних пиков
  -> occupied band
  -> effective frequency + bandwidth
```

Дальше в `slave=auto`:

```text
occupied bins <= sparse cap  -> exact DFT только этих bins
occupied bins > sparse cap   -> full Slave FFT
```

Exact DFT математически совпадает с коэффициентами полного FFT на выбранных bin'ах. При этом phase оценивается не по фиксированным 5 bin'ам, а по всей фактически обнаруженной полосе.

Важно: даже явный `--slave-mode exact-bins` теперь **не имеет права испортить широкополосный сигнал** — при превышении sparse cap автоматически выполняется full FFT.

Для OFDM/DSSS/wideband-noise benchmark проверяет, что `auto` действительно выбирает `full-fft` и использует весь detected band.

### 2. Синхронизация stream: queue order не считается временной меткой

Каждая полностью набранная пачка получает `frame_id` на producer-стороне:

```text
Master producer: frame_id 0,1,2,...
Slave producer:  frame_id 0,1,2,...
```

Consumer сопоставляет **только одинаковые IDs**. Если одна сторона временно ушла вперёд, старые пачки только этой стороны отбрасываются до восстановления совпадающего ID. `--max-frame-gap` теперь является диагностическим soft-threshold и **не перезапускает HackRF**. Отдельный `--hard-frame-gap` оставляет аварийный предел для действительно патологического состояния. Это устраняет ложные аппаратные restart'ы из-за кратковременного планировщика/USB skew и не создаёт самоподдерживающийся loop от очистки обеих очередей.

Это также защищает от short reads: `read_exact_fd()` добирает неполный `os.readv()` до полной пачки, поэтому нормальная фрагментация `read()` не трактуется как потеря данных.

Ограничение, которое нельзя честно «залатать» программой: голый поток `hackrf_transfer -r -` не содержит sample timestamp/packet sequence. Поэтому **внутреннюю потерю нескольких семплов внутри уже принятого потока невозможно доказать одним frame_id**. Для этого добавлена независимая защита: межканальная coherence и автоматическое восстановление при устойчивом high-SNR incoherence. Для широкополосных сигналов рассогласование дополнительно проявляется как phase decorrelation.

### 3. Calibration теперь не падает из-за пустого эфира

Калибровка работает как контролируемый retry loop:

```text
слабый сигнал       -> ждать
неоднозначная корреляция -> ждать
надёжный peak       -> coarse lag + local refinement -> align
```

Прежний сценарий «нет сигнала в первые 50 ms -> RuntimeError» удалён. Исключение остаётся только для реального EOF/процесса HackRF или внешнего timeout, если его явно задано на уровне вызывающего кода.

Важная деталь: корреляция получает `lag_d` как абсолютную позицию найденного фрагмента в decimated Slave. Физический относительный сдвиг вычисляется как:

```text
delta_samples = (lag_d - start) * decim
```

а локальное уточнение выполняется около `start*decim + delta_samples`. Это исключает старую ошибку двойного прибавления `start`, которая могла увести уточняющее окно примерно в `2*start`. Положительное `delta_samples` означает, что совпадающий фрагмент находится позже в Slave; тогда именно Slave prefix отбрасывается для выравнивания.

### 4. Один `argmax` заменён на multi-target detection

Используются:

- local maxima;
- circular CA-CFAR;
- suppression близких пиков;
- clustering соседних spectral maxima;
- multi-scale integrated-energy detection для широкополосных источников;
- overlap-aware merge.

Таким образом, независимые разночастотные передатчики в одном FFT окне становятся отдельными `Detection`.

При полном перекрытии двух сигналов одной частотной полосой алгоритм не обещает невозможного: двухэлементный RF-фронтенд не получает дополнительную пространственную степень свободы только из-за software merge.

### 5. Вращение платформы используется как физическая модель раскрытия phase ambiguity

При `d > lambda/2` статическая двухэлементная фаза неоднозначна. Число `arcsin(...)` не объявляется истинным world bearing.

При подключенном encoder алгоритм использует:

```text
phi(t) = 2*pi*f*d/c * sin(theta_world - psi(t)) + noise
```

и ищет `theta_world` по **глобальному circular likelihood** на всём накопленном rotation arc. Это не эвристика «следи за производной»: spatial aliases выбираются по согласованности с физической моделью.

Encoder должен выдавать:

```text
angle_deg
```

или:

```text
timestamp_ns,angle_deg
```

`timestamp_ns` должен быть в той же monotonic time basis, что и timestamp потока, если он передаётся внешним контроллером. Для обычного UART с одним только углом `EncoderSource` ставит host `time.monotonic_ns()` при приёме строки, но читает char-device через `select()+os.read()`, а не через буферизованный Python `for line`, чтобы уменьшить host-side latency. `--encoder-zero-deg` задаёт механический нулевой сдвиг, а `--encoder-latency-ms` компенсирует **измеренную** задержку доставки телеметрии. Нельзя без проверки считать, что любой `/dev/ttyS*` имеет 16-мс задержку: такой latency timer характерен для некоторых USB-serial драйверов (например FTDI), тогда как встроенный UART не обязан иметь этот таймер. На драйверах, которые экспортируют `/sys/class/tty/<tty>/device/latency_timer`, режим `--encoder-low-latency auto` пытается выставить 1 ms; при наличии `setserial` он использует `low_latency` как fallback.

## Sparse DFT без trig-cache miss

Exact-bin режим больше не строит матрицу:

```text
nbins × N FFT
```

из `sin/cos` при каждом новом наборе частот. Вместо этого для каждого `N` один раз создаётся компактная таблица:

```text
root[k] = exp(-j 2*pi*k/N),  k=0..N-1
```

Например, для `N=32768` это около 256 KiB в `complex64`. Native kernel для произвольного/FHSS набора bins проходит эту таблицу шагом `k` и вычисляет те же exact DFT coefficients, что и полный FFT. Поэтому смена несущей не вызывает 30--45 ms Python `sin/cos` stall и не раздувает cache.

Для очень широкого occupied band `auto` всё равно выбирает full Slave FFT: compact root table не используется как оправдание для ухудшения чувствительности или точности.

## Численная устойчивость CFAR

Скользящие train-cell суммы в `cfar_noise_circular_f32()` накапливаются в `double`, хотя вход и выход остаются `float32`. Это сохраняет O(1) moving update и NEON-friendly доступ к данным, но исключает накопительный drift при длительной обработке больших FFT.

## NEON / H618 path

В `neon_kernels.c` native ARM64 path выполняет:

```text
int8 IQ -> complex64
DC removal
Hann window
power
cross-spectrum accumulation
CA-CFAR moving noise
wideband energy scoring
exact-bin DFT
```

Сборка:

```bash
./build_neon.sh
```

На ARM64 используется:

```text
-O3 -ffast-math -fomit-frame-pointer -mcpu=cortex-a53 -mtune=cortex-a53
```

Не следует считать комментарий «NEON» доказательством конкретного opcode: реальное SIMD исполнение зависит от ARM64 build. `NeonKernels().neon_enabled` показывает, был ли native NEON path собран.

## Установка

```bash
sudo apt update
sudo apt install -y python3 python3-numpy python3-scipy build-essential
cd /path/to/orange_pi_h618_doa
chmod +x build_neon.sh
./build_neon.sh
```

Проверка:

```bash
python3 - <<'PY'
from doa_h618_core import NeonKernels
k = NeonKernels()
print('native library:', k.available)
print('NEON:', k.neon_enabled)
PY
```

## Боевой запуск

Для старта на 2.437 GHz:

```bash
python3 drone_doa_h618.py \
  --freq 2437000000 \
  --sample-rate 5000000 \
  --fft 32768 \
  --batch 4 \
  --fft-workers 3 \
  --slave-mode auto \
  --sparse-max-bins 32 \
  --baseline-cm 5 \
  --threshold-snr 12 \
  --phase-occupancy-db 1.5 \
  --lna 16 \
  --vga 16 \
  --gain-auto
```

`auto` является безопасным режимом. `exact-bins` следует использовать только для контролируемого узкополосного режима; код всё равно принудительно откатится на full FFT при обнаружении широкой полосы.

При наличии энкодера:

```bash
python3 drone_doa_h618.py ... --encoder-path /dev/ttyS1 \
  --encoder-zero-deg 0.0 \
  --encoder-latency-ms 0.0
```

Для UART заранее задайте параметры терминала/MCU. Формат входа описан выше.

## Synthetic benchmark без HackRF

Benchmark полностью автономный: он генерирует IQ, физически моделирует межантенный delay как частотно-зависимую фазу, добавляет независимый receiver noise, квантует в int8 и прогоняет **тот же SDRCore**, что используется в боевом процессе.

Полный рекомендуемый тест:

```bash
python3 benchmark_sdr_synthetic.py \
  --fft-sizes 4096,8192,16384,32768,65536 \
  --workers 1,2,3,4 \
  --modes auto,full-fft \
  --waveforms tone,multitone,ofdm,wideband-noise,dsss \
  --sample-rates 2000000,5000000,10000000,20000000 \
  --snrs -6,0,6,12,18,24 \
  --freqs 800000000,2437000000,3000000000,6000000000 \
  --angles -60,-30,0,30,60 \
  --trials 20 \
  --warmup 3 \
  --repeats 25 \
  --multitarget-trials 20 \
  --json benchmark_results.json
```

Benchmark проверяет:

1. DSP throughput и real-time factor для разных FFT/Fs/workers;
2. synthetic dual-pipe ingest ceiling;
3. numerical equivalence exact-bin DFT vs full FFT;
4. Pdet/RMSE/coherence для CW, multitone, OFDM, DSSS и wideband;
5. автоматическое переключение `auto -> full-fft` по фактической occupied bandwidth;
6. multi-target recovery;
7. rotating-baseline ambiguity resolver на 6 GHz.

Абсолютные `ms/batch` с x86-хоста нельзя переносить на H618. Для Orange Pi benchmark надо запускать **непосредственно на плате** и сравнивать его с требуемой raw pair rate:

```text
pair MB/s = 4 * Fs / 1e6
```

Например, при 5 MSPS это 20 MB/s на два 8-bit IQ потока.

## RF-ограничение базы 5 см

Для `d = 5 cm` граница `d = lambda/2` находится примерно на:

```text
f = c/(2d) ~= 2.998 GHz
```

Поэтому на 800 MHz и 2.437 GHz статическая двухэлементная фаза однозначна, а около 6 GHz — нет. Rotating-baseline resolver может снять эту неоднозначность только при наличии достаточного углового раскрытия и надёжно синхронизированного encoder; без вращения аппаратная ambiguity остаётся фундаментальной.

## Практическая настройка H618

Сначала benchmark'ом найти лучший `fft-workers` для каждого FFT. Не считать `workers=4` автоматически самым быстрым.

Для непрерывного поля контролировать thermal throttling:

```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
cat /sys/class/thermal/thermal_zone*/temp
```

Для реального dual-HackRF потока желательно проверить USB topology и отсутствие shared bottleneck.

## Regression tests после сборки

Полностью синтетическая проверка исправлений:

```bash
python3 test_review_regressions.py
```

Она проверяет exact-bin/root-table equivalence, численную устойчивость CFAR, мягкое восстановление frame skew и формулу относительного calibration lag.
