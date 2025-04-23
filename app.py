from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)

# 加载数据
df = pd.read_excel("yearwater_converted.xlsx")

# 获取所有地区
regions = df["PR"].dropna().unique().tolist()

@app.route('/')
def index():
    return render_template("index.html", regions=regions)

@app.route('/predict_rainfall', methods=['POST'])
def predict_rainfall():
    try:
        data = request.get_json(silent=True)
        region = data.get('region') if data else request.form.get('region')
        year = data.get('year') if data else request.form.get('year')

        if not region or not year:
            return render_template("results.html", error="缺少必要参数: 地区或年份")

        try:
            year = int(year)
        except ValueError:
            return render_template("results.html", error="年份必须是整数")

        # ARIMA预测函数
        def arima_forecast(series, steps=1):
            try:
                model = ARIMA(series, order=(1, 1, 1))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=steps)
                return forecast[0] if steps == 1 else forecast.tolist()
            except Exception as e:
                print(f"ARIMA预测错误: {str(e)}")
                return np.nan

        # 获取地区数据
        target_row = df[df["PR"] == region]
        if len(target_row) == 0:
            return render_template("results.html", error=f"未找到地区 {region}")

        if year >= 2025:
            # 预测未来数据
            history = target_row[[str(y) for y in range(1950, 2023) if str(y) in df.columns]].values[0]
            history = pd.Series(history).dropna()

            if len(history) < 10:
                return render_template("results.html", error="历史数据不足，无法进行预测")

            steps = year - 2022
            forecast = arima_forecast(history, steps=steps)

            if year == 2025:
                rain = forecast[0] if isinstance(forecast, list) else forecast
            elif year == 2026:
                rain = forecast[1] if isinstance(forecast, list) and len(forecast) > 1 else np.nan
            else:
                return render_template("results.html", error="只支持预测2025或2026年")
        else:
            # 获取历史数据
            if str(year) not in df.columns:
                return render_template("results.html", error=f"无{year}年数据")

            rain = target_row[str(year)].values[0]

        return {
            "region": region,
            "year": year,
            "rain": rain,
            "predicted": (year >= 2025)  # 如果是未来年份，标记为预测
        }

        # return render_template("results.html", region=region, year=year, rain=rain, predicted=(year >= 2025))

    except Exception as e:
        return {
            "region": region,
            "year": year,
            "rain": rain,
            "predicted": (year >= 2025)  # 如果是未来年份，标记为预测
        }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
