# حلقهٔ مهندسی خودکار TEMP-DRIFT با OpenCode

این بسته را در ریشهٔ پروژهٔ زیر Extract کنید:

`C:\Users\jafari.h.SPADANACO\Desktop\ai_project\testing\qsnn_temp_drift_project`

سپس PowerShell را در همان مسیر باز کنید و فقط یک‌بار نصب اولیه را اجرا کنید:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\ResearchLoop\INSTALL.ps1
```

حالا OpenCode را از ریشهٔ پروژه باز کنید:

```powershell
opencode .
```

داخل محیط OpenCode این command را بزنید:

```text
/temp-drift-loop
```

تمام فرمان‌ها، تست‌ها، خطاها، تعمیرها و خلاصهٔ نتایج در همان TUI دیده می‌شوند. برای دیدن وضعیت بدون تغییر فایل‌ها:

```text
/temp-drift-status
```

اگر OpenCode یا سیستم قطع شد، دوباره `opencode .` را باز کنید و `/temp-drift-loop` را بزنید؛ از `Reports/status.json` ادامه می‌دهد.

حلقه از آخرین فاز معتبر ادامه می‌دهد، برای هر تلاش log می‌نویسد و فقط وقتی gate مستقل فاز PASS شود جلو می‌رود. برای توقف امن:

```powershell
.\ResearchLoop\STOP_LOOP.ps1
```

برای حالت headless/background هنوز می‌توانید `RUN_LOOP.ps1` را اجرا کنید، اما خروجی کامل آن در `Reports/logs/` ذخیره می‌شود. برای تجربهٔ موردنظر شما، حالت TUI و `/temp-drift-loop` پیشنهاد می‌شود. وضعیت خارج از TUI نیز با این فرمان دیده می‌شود:

```powershell
.\ResearchLoop\STATUS.ps1
```

## عامل کدنویسی: فقط OpenCode

این نسخه فقط از OpenCode استفاده می‌کند و هیچ fallback به Codex ندارد. ابتدا بررسی کنید OpenCode نصب و به OpenAI متصل است:

```powershell
opencode --version
opencode auth list
```

اگر OpenAI در فهرست نبود:

```powershell
opencode auth login
```

در حالت TUI، command سفارشی مستقیماً با agent نوع `build` اجرا می‌شود و subprocess دیگری از OpenCode نمی‌سازد. `RUN_LOOP.ps1` فقط برای اجرای non-interactive با `opencode run --auto` باقی مانده است.

مسیر پروژه و Python از قبل در `ResearchLoop/config.json` ثبت شده‌اند. اگر پروژه را جابه‌جا کردید فقط همان فایل را تغییر دهید.

## نکات علمی مهم

- نتایج مقاله در `ResearchLoop/reference/reference_results.csv` فقط مرجع‌اند و هرگز به‌عنوان نتیجهٔ مدل شما استفاده نمی‌شوند.
- مقایسه فقط برای تنظیمات دقیقاً همسان با مقاله مجاز است؛ مقایسهٔ بودجه‌های نامنطبق با برچسب `NON_COMPARABLE` ثبت می‌شود.
- attack فقط timestamp را تغییر می‌دهد. line/polarity/amplitude/count باید حفظ شود.
- ASR فقط روی نمونه‌های clean-correct محاسبه می‌شود.
- فایل‌های `Reports/status.json`، `Reports/logs/` و receiptهای هر فاز امکان Resume را فراهم می‌کنند.

منبع مرجع: https://arxiv.org/html/2602.03284v1
