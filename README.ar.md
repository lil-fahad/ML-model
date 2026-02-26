<div dir="rtl">

# نظام التنبؤ بأسعار الأسهم (Hybrid Stock Predictor)

واجهة محلية مبنية بـ **Streamlit** تستخدم نماذج تعلم آلي متقدمة على **ميزات تقنية مُهندسة** للتنبؤ باتجاه أسعار الأسهم.

## نبذة عن المشروع

يهدف هذا المشروع إلى بناء نموذج تنبؤي يُساعد في تحديد اتجاه سعر السهم (صعود/هبوط) بالاعتماد على مؤشرات تحليل تقني محسوبة من البيانات التاريخية. يدعم المشروع نموذجين:

- **النموذج الأصلي** — يعتمد على 10 ميزات أساسية (عوائد، تذبذب، تراجع).
- **النموذج المحسّن** — يضيف 8 مؤشرات تقنية متقدمة (RSI، MACD، Bollinger Bands، إلخ).

## هيكل المشروع

```
ML-model/
├── app/                        # تطبيق Streamlit
│   └── streamlit_app.py
├── data/                       # ملفات CSV للأسهم (تُحمَّل بواسطة download_data.py)
├── models/                     # ملفات النماذج المدرَّبة
│   ├── hybrid_model.pkl.b64       # النموذج الأصلي (10 ميزات)
│   ├── enhanced_model.pkl.b64     # النموذج المحسّن (18 ميزة)
│   └── *_meta.json                # بيانات وصفية للنماذج
├── scripts/                    # سكريبتات مساعدة
│   ├── download_data.py        # تحميل مجموعة بيانات Kaggle
│   ├── restore_models.py       # استعادة النماذج من base64
│   ├── train_model.py          # تدريب النموذج المحسّن
│   ├── run_simulator.py        # اختبار النماذج على محاكي التداول
│   └── inspect_model.py        # فحص بنية النموذج
├── src/                        # وحدات الكود المصدري
│   ├── features.py             # هندسة الميزات الأساسية
│   ├── hybrid_features.py      # ميزات النموذج الهجين
│   └── enhanced_features.py    # الميزات المحسّنة مع المؤشرات التقنية
├── tests/                      # اختبارات الوحدة
│   ├── test_enhanced_features.py
│   └── test_simulator.py
├── requirements.txt            # متطلبات Python
└── README.md
```

## البدء السريع

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/restore_models.py
streamlit run app/streamlit_app.py
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\restore_models.py
streamlit run app\streamlit_app.py
```

## الحصول على البيانات

### الخيار أ — خادم Kaggle MCP (موصى به)

يتضمن المشروع ملف `.mcp.json` يوصّل أي عميل MCP (مثل Claude Desktop أو Cursor أو VS Code) بخادم Kaggle البعيد:

```
https://www.kaggle.com/mcp
```

#### 1. توليد رمز المصادقة

مفتاح Kaggle API متاح في [https://www.kaggle.com/settings/account](https://www.kaggle.com/settings/account) تحت **API ← إنشاء رمز جديد**.
قم بترميزه كـ Basic-auth token:

```bash
# Linux / macOS
export KAGGLE_BASIC_AUTH_TOKEN=$(echo -n "YOUR_KAGGLE_USERNAME:YOUR_KAGGLE_KEY" | base64)
```

```powershell
# Windows PowerShell
$env:KAGGLE_BASIC_AUTH_TOKEN = [Convert]::ToBase64String(
    [Text.Encoding]::ASCII.GetBytes("YOUR_KAGGLE_USERNAME:YOUR_KAGGLE_KEY"))
```

#### 2. تحميل مجموعة البيانات عبر عميل MCP

بعد الاتصال بالخادم، أرسل الطلب التالي:

```
Download paultimothymooney/stock-market-data
```

سيقوم خادم Kaggle MCP بتدفق ملفات البيانات إلى جلستك.

### الخيار ب — سكريبت Python

```bash
# يتطلب متغيري البيئة KAGGLE_USERNAME و KAGGLE_KEY (أو ملف ~/.kaggle/kaggle.json)
python scripts/download_data.py
```

يقوم هذا السكريبت بتحميل مجموعة البيانات عبر `kagglehub` ونسخ ملفات CSV إلى مجلد `data/`.

## تدريب نموذج جديد

```bash
python scripts/train_model.py
```

سيقوم هذا السكريبت بـ:
1. تحميل جميع ملفات CSV من مجلد `data/`
2. بناء الميزات المحسّنة (RSI، MACD، Bollinger Bands، إلخ)
3. تدريب عدة نماذج (LogisticRegression، RandomForest، GradientBoosting، XGBoost، LightGBM)
4. تحسين المعاملات باستخدام Optuna
5. حفظ أفضل نموذج في `models/enhanced_model.pkl.b64`

## الميزات المستخدمة

### الميزات الأصلية (10 ميزات)
- `ret_1`، `ret_3`، `ret_5`، `ret_10`، `ret_20` — العوائد على فترات مختلفة
- `vol_5`، `vol_10`، `vol_20` — مقاييس التذبذب
- `dd_20` — أقصى تراجع خلال 20 يوم
- `range_pct` — النطاق اليومي كنسبة مئوية

### الميزات المحسّنة (18 ميزة)
جميع الميزات الأصلية بالإضافة إلى:
- `rsi_14` — مؤشر القوة النسبية (مُعيَّر)
- `macd_signal` — تقاطع خط إشارة MACD
- `bb_position` — موضع Bollinger Bands (من -1 إلى 1)
- `momentum_10` — زخم 10 أيام
- `obv_change` — تغيّر حجم التداول المتوازن (OBV)
- `atr_14` — متوسط المدى الحقيقي (مُعيَّر)
- `ema_ratio` — نسبة EMA 12/26 (مؤشر الاتجاه)
- `volume_sma_ratio` — حجم التداول نسبةً إلى المتوسط المتحرك البسيط لـ 20 يوم
- `stoch_k` — مذبذب Stochastic %K مُعيَّر (0–1)
- `adx_14` — مؤشر الاتجاه المتوسط مُعيَّر (قوة الاتجاه)

## أداء النماذج

| النموذج | الدقة | F1 Score | ملاحظات |
|---------|-------|----------|---------|
| الأصلي (RandomForest) | 52.03% | 52.17% | 10 ميزات |
| المحسّن (XGBoost + Optuna) | 49.19% | 53.60% | 18 ميزة، محسَّن بـ Optuna |

**ملاحظة:** التنبؤ بأسعار الأسهم أمر بالغ الصعوبة بطبيعته. النماذج مُحسَّنة لمقياس F1 لتحقيق توازن بين الدقة والاستدعاء.

## البيانات

- استخدم **خادم Kaggle MCP** (راجع "الحصول على البيانات" أعلاه) أو شغّل
  `python scripts/download_data.py` لملء مجلد `data/` بمجموعة البيانات
  `paultimothymooney/stock-market-data`.
- يمكنك أيضاً إضافة ملفات CSV خاصة بك إلى `data/` بصيغة `TICKER.csv`
  (الأعمدة: `date`، `open`، `high`، `low`، `close`، `volume`).
- خيار `yfinance` متاح في واجهة Streamlit عند توفر اتصال بالإنترنت.

## المتطلبات

- **الأساسية:** streamlit، pandas، numpy، scikit-learn، joblib، yfinance
- **ML المتقدم:** xgboost، lightgbm، optuna
- **محاكي التداول:** gymnasium، gym-anytrading
- **الاختبار:** pytest

## محاكي التداول

شغّل النماذج على محاكي التداول [gym-anytrading](https://github.com/AminHP/gym-anytrading) لتقييم أدائها:

```bash
python scripts/run_simulator.py
```

يقارن هذا السكريبت:
- **الخط الأساسي العشوائي** — قرارات شراء/بيع عشوائية
- **النموذج الهجين** — نموذج RandomForest بـ 10 ميزات
- **النموذج المحسّن** — نموذج XGBoost بـ 18 ميزة

مثال على النتائج:
```
============================================================
الملخص
============================================================
النموذج              متوسط الربح     معدل الفوز       
--------------------------------------------------
الخط الأساسي العشوائي   0.1471          0.00%          
النموذج الهجين         0.2364          0.00%          
النموذج المحسّن        0.6711          0.00%
```

**ملاحظة:** النموذج المحسّن يُظهر تحسناً بـ 4.5 أضعاف مقارنة بالتداول العشوائي!

## ملاحظات

- تُخزَّن النماذج كنصوص مشفرة بـ base64 (`.pkl.b64`) لتسهيل إدارة الإصدارات
- `scripts/restore_models.py` يُعيد إنشاء ملفات `.pkl` الثنائية محلياً
- شغّل `pytest tests/` للتحقق من صحة عملية هندسة الميزات

</div>
