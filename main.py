from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import shapiro

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# CORS設定，允許從http://localhost:3000的請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=
        ["http://localhost:3000",
        "https://shadytable-frontend.vercel.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定義請求的數據模型
class AnalysisRequest(BaseModel):
    data: List[dict]
    groupVar: Optional[str]
    catVars: List[str]
    contVars: List[str]
    fillNA: Optional[bool] = False

#p值回報格式設定
def format_p(p):
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"

#常態性檢定函數
def normality_test(data, cont_vars):
    is_normal = {}
    for var in cont_vars:
        values = data[var].dropna()
        if len(values) >= 3:
            _, p = shapiro(values)
            is_normal[var] = p > 0.05
        else:
            is_normal[var] = False  # 樣本太少 → 當作非常態
    return is_normal

#===============主分析函數========================
def analyze(data: pd.DataFrame, group_var, cat_vars, cont_vars, fillna: bool = False):
    result_rows = []
    groupCounts = {}

    #======== 檢查 catVars 與 contVars 型別是否合理 ========
    for col in cat_vars:
        if col not in data.columns:
            raise ValueError(f"❗類別變項 '{col}' 不存在於資料中")
        dtype = data[col].dtype
        n_unique = data[col].dropna().nunique()
        if pd.api.types.is_numeric_dtype(dtype) and n_unique > 10:
            raise ValueError(f"❗類別變項 '{col}' 為數值型且 unique 值數量為 {n_unique}，疑似連續變項")

    for col in cont_vars:
        if col not in data.columns:
            raise ValueError(f"❗連續變項 '{col}' 不存在於資料中")
        dtype = data[col].dtype
        if not pd.api.types.is_numeric_dtype(dtype):
            raise ValueError(f"❗連續變項 '{col}' 的資料型別為 {dtype}，不是數值型（int 或 float）")



#處理遺漏值填補（類別變項用眾數；連續變項用平均數）
    if fillna:
        for col in cat_vars:
            if data[col].isna().any():
                mode_val = data[col].mode(dropna=True)
                if not mode_val.empty:
                    data[col] = data[col].fillna(mode_val[0])
        for col in cont_vars:
            if data[col].isna().any():
                data[col] = data[col].fillna(data[col].mean())

#======== 自動檢查並轉換分組變項(group_var)為類別型（若合適） ========
    if group_var:
        series = data[group_var].dropna()
        n_unique = series.nunique()
        dtype = series.dtype

        is_categorical = (
            pd.api.types.is_object_dtype(dtype) or
            pd.api.types.is_categorical_dtype(dtype) or
            (pd.api.types.is_numeric_dtype(dtype) and n_unique <= 10)
        )

        if not is_categorical:
            raise ValueError(f"❗分組變項必須是類別變項")

        if pd.api.types.is_numeric_dtype(dtype) and n_unique <= 10:   #如果是數值型但唯一值數量少於等於10，轉換為類別型
            data[group_var] = data[group_var].astype("category")

#分組設定
    if group_var:
        groups = data[group_var].dropna().unique().tolist()
        for g in groups:
            groupCounts[g] = data[data[group_var] == g].shape[0]  #計算分組內樣本數
    else:
        groups = ["All"] #"All"表示沒有分組

    #結果表格的變項標題列
    missing_pct = data[group_var].isna().mean() * 100 if group_var else 0
    row_header = {
        "Variable": f"**{group_var}**",
        "Missing": f"{missing_pct:.1f}%",  #計算分組變項的遺漏值百分比
        "Method": "-",
        "P": "-"
    }

    for g in data[group_var].dropna().unique():
        count = data[data[group_var] == g].shape[0]
        row_header[g] = f"{count} (100%)"

    result_rows.insert(0, row_header)

    filtered_cat_vars = [v for v in cat_vars if v != group_var] #除了分組變項外的類別變項

    for var in filtered_cat_vars:
        missing_pct = round(data[var].isna().mean() * 100, 1) #計算類別變項的遺漏值百分比

        row_header = {
            "Variable": f"**{var}**",
            "Missing": f"{missing_pct}%",
            "Method": "-",
            "P": "-"
        }

        for g in groups:
            row_header[g] = "—"

#==============計算類別變項的列聯表和卡方檢定========================
        if group_var:
            ct = pd.crosstab(data[var], data[group_var])  #計算列聯表
            total_n = ct.values.sum()  # 計算總樣本數
            if (ct.shape == (2, 2)) and ((ct.values < 5).any() or total_n < 20): #如果是2x2列聯表且有任何單元格小於5或總樣本數小於20，使用Fisher精確檢定
                _, p = stats.fisher_exact(ct.values)
                method = "Fisher"
            else:
                _, p, _, _ = stats.chi2_contingency(ct)
                method = "Chi-square"

            row_header["Method"] = method
            row_header["P"] = format_p(p)

        result_rows.append(row_header)

        for level in data[var].dropna().unique():
            row = {"Variable": level}
            for g in groups:
                total = (data[group_var] == g).sum() if group_var else data.shape[0]
                count = data[(data[group_var] == g) & (data[var] == level)].shape[0] if group_var else data[data[var] == level].shape[0]
                percent = count / total * 100 if total > 0 else 0
                row[g] = f"{count} ({percent:.1f}%)"
            result_rows.append(row)

#==============計算連續變項的描述統計和常態檢定========================

    is_normal = normality_test(data, cont_vars)  # 先一次計算全部變項的常態檢定結果

    for var in cont_vars:
        missing_pct = round(data[var].isna().mean() * 100, 1) #計算連續變項的遺漏值百分比
        normal = is_normal.get(var, False)  # 查詢是否常態分布
        normal_text = "Yes" if normal else "No"
        method = "-"
        p_value = "-"

        row_header = {
            "Variable": f"**{var}**",
            "Missing": f"{missing_pct}%",
            "Normal": normal_text,
            "Method": "-",
            "P": "-"
        }

        for g in groups:
            row_header[g] = "—"

        if group_var:
            group_values = [data[data[group_var] == g][var].dropna() for g in groups]
            if all(len(gv) > 0 for gv in group_values):
                if len(groups) == 2: #如果只有兩個組別比較時，使用t檢定或Mann-Whitney U檢定
                    if normal:  #如果數據符合常態分佈
                        _, p = stats.ttest_ind(*group_values, nan_policy="omit") #使用獨立樣本t檢定
                        method = "t-test"
                    else:  #如果數據不符合常態分佈
                        _, p = stats.mannwhitneyu(*group_values)
                        method = "Mann–Whitney U"
                else: #多組比較
                    if normal:
                        _, p = stats.f_oneway(*group_values)
                        method = "ANOVA"
                    else:
                        _, p = stats.kruskal(*group_values)
                        method = "Kruskal-Wallis"
                row_header["Method"] = method
                row_header["P"] = format_p(p)

        #==============計算連續變項的均值和標準差========================
        for g in groups:
            values = data[data[group_var] == g][var].dropna() if group_var else data[var].dropna()

            if values.empty:
                row_header[g] = "—"
                continue

            if is_normal.get(var, False):  # 常態分布 → mean ± SD
                mean = values.mean()
                std = values.std()
                row_header[g] = f"{mean:.1f} ± {std:.1f}"

            else:  # 非常態分布 → median (min–max)
                median = values.median()
                min_val = values.min()
                max_val = values.max()
                row_header[g] = f"{median:.1f} ({min_val:.1f}–{max_val:.1f})"

        result_rows.append(row_header)


    all_keys = set(k for row in result_rows for k in row.keys())
    for row in result_rows:
        for k in all_keys:
            row.setdefault(k, "—")

    return {
        "table": pd.DataFrame(result_rows).astype(str).to_dict(orient="records"),
        "groupCounts": groupCounts
    }

@app.post("/analyze")
def run_analysis(req: AnalysisRequest):
    try:
        df = pd.DataFrame(req.data)
        print("✅ 轉換成 DataFrame 成功，欄位：", df.columns.tolist())
        result = analyze(df, req.groupVar, req.catVars, req.contVars, req.fillNA)
        return result
    except ValueError as ve:
        # 使用者輸入錯誤類（400 Bad Request）
        print("❌ 使用者指定錯誤：", str(ve))
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # 程式邏輯錯誤（500）
        print("❌ 系統錯誤：", traceback.format_exc())
        raise HTTPException(status_code=500, detail="伺服器內部錯誤，請聯絡系統管理者")
